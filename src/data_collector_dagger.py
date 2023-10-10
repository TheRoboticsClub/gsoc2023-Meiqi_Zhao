import argparse
import logging
import carla
from traffic_utils import spawn_vehicles, spawn_pedestrians
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils import carla_seg_to_array, carla_rgb_to_array,  road_option_to_int, int_to_road_option, read_routes, traffic_light_to_int, calculate_delta_yaw
from test_utils import model_control, calculate_distance
import numpy as np
import torch
from ModifiedDeepestLSTMTinyPilotNet.utils.ModifiedDeepestLSTMTinyPilotNet import PilotNetOneHot, PilotNetEmbeddingNoLight, PilotNetOneHotNoLight
import random
import h5py

has_collision = False
def collision_callback(data):
    global has_collision
    has_collision = True

def main(params):
    # load model
    if params.ignore_traffic_light:
        #model = PilotNetEmbeddingNoLight((288, 200, 6), 3, 4)
        model = PilotNetOneHotNoLight((288, 200, 6), 3, 4)
    else:
        model = PilotNetOneHot((288, 200, 6), 3, 4, 4)

    model.load_state_dict(torch.load(params.model, map_location=torch.device('cpu')))
    model.eval()
    # load episodes9
    episode_configs = read_routes(params.episode_file)
    #num_episodes = len(episode_configs)
    num_episodes = params.n_episodes

    # create carla client
    client = carla.Client(params.ip, params.port)
    client.set_timeout(200.0)

    # load map
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != params.map:
        world = client.load_world(params.map)
    map = world.get_map()

    # set the weather to clear
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)
    
    # set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # set up traffic manager
    traffic_manager = client.get_trafficmanager(params.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    # set red light duration
    list_actor = world.get_actors()
    for actor_ in list_actor:
        if isinstance(actor_, carla.TrafficLight):
            # actor_.set_state(carla.TrafficLightState.Green) 
            actor_.set_green_time(15.0)


    # configure hybrid mode
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    world.tick()
    
    episode_cnt = 0
    while episode_cnt < params.n_episodes:
        print(f"episode {episode_cnt}")

        # reset collision checker
        global has_collision
        has_collision = False
        
        # Get the specific spawn points
        spawn_points = map.get_spawn_points()
        episode_config = random.choice(episode_configs)
        #episode_config = episode_configs[i]
        start_point = spawn_points[episode_config[0][0]]
        end_point = spawn_points[episode_config[0][1]]
        print(f"from spawn point #{episode_config[0][0]} to #{episode_config[0][1]}")
        route_length = episode_config[1]
        route = episode_config[2].copy()

        # get the blueprint for this vehicle
        blueprint_library = world.get_blueprint_library()
        blueprint = blueprint_library.filter('model3')[0]
    
        # spawn the vehicle
        blueprint.set_attribute('role_name', 'hero')
        vehicle = world.spawn_actor(blueprint, start_point)

        # Add vehicle to the traffic manager
        vehicle.set_autopilot(True)
        traffic_manager.set_route(vehicle, route)
        if params.ignore_traffic_light:
            traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.ignore_signs_percentage(vehicle, 100)
        traffic_manager.distance_to_leading_vehicle(vehicle, 4.0)
        traffic_manager.set_desired_speed(vehicle, params.target_speed)

        # set spectator
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
        # carla.Rotation(pitch=-90)))
        offset = carla.Location(x=0.0, z=2)
        spectator.set_transform(carla.Transform(transform.location + offset, transform.rotation))
        
        for j in range(10):
            world.tick()

        all_id, all_actors, pedestrians_list, vehicles_list = [], [], [], []

        # Spawn vehicles
        if params.n_vehicles > 0:
            vehicles_list = spawn_vehicles(world, client, params.n_vehicles, traffic_manager)

        # Spawn Walkers
        if params.n_pedestrians > 0:
            all_id, all_actors, pedestrians_list = spawn_pedestrians(world, client, params.n_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=0.0)\
        
        print('spawned %d vehicles and %d walkers.' % (len(vehicles_list), len(pedestrians_list)))

        sensors = []
        # setup rgb camera
        rgb_cam = RGBCamera(world, vehicle)
        rgb_cam_sensor = rgb_cam.get_sensor()
        sensors.append(rgb_cam_sensor)

        # setup semantic segmentation camera
        seg_cam = SegmentationCamera(world, vehicle)
        seg_cam_sensor = seg_cam.get_sensor()
        sensors.append(seg_cam_sensor)

        collision_sensor = setup_collision_sensor(world, vehicle)
        collision_sensor.listen(collision_callback)
        sensors.append(collision_sensor)
        for i in range(10):
            world.tick()
    
        frame = 0
        prev_hlc = 0
        prev_yaw = 0
        delta_yaw = 0
        turning_infraction = False
        still_frames = 0
        episode_data = {
            'frame': [],
            'hlc': [],
            'light': [],
            'controls': [],
            'measurements': [],
            'rgb': [],
            'segmentation': []
        }
        cur_driver_count = 0
        autopilot = False
        while True:
            # update spectator's location to match the vehicle's
            transform = vehicle.get_transform()
            vehicle_location = transform.location
            # spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=50), carla.Rotation(pitch=-90)))
            offset = carla.Location(x=0.0, z=2)
            spectator.set_transform(carla.Transform(transform.location + offset, transform.rotation))

            if calculate_distance(vehicle_location, end_point.location) < 1.5 or frame > params.max_frames_per_episode or turning_infraction:
                print(f'Episode {episode_cnt} ending')
                vehicle.destroy()
                for sensor in sensors: sensor.destroy()

                print('\ndestroying %d vehicles' % len(vehicles_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                # stop walker controllers (list is [controller, actor, controller, actor ...])
                for i in range(0, len(all_id), 2):
                    all_actors[i].stop()
                print('\ndestroying %d walkers' % len(pedestrians_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
                # world.tick()

                break

            # calculate speed
            velocity = vehicle.get_velocity()
            speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed = 3.6 * speed_m_s #m/s to km/h 
            if speed < 1:
                still_frames += 1
            else:
                still_frames = 0
            if still_frames >= 100:
                print('skipping')

            
            # calculate next high-level command
            vehicle_location =vehicle.get_transform().location
            vehicle_waypoint = map.get_waypoint(vehicle_location)
            next_waypoint = vehicle_waypoint.next(10.0)[0]

            if vehicle_waypoint.is_junction or next_waypoint.is_junction:
                if prev_hlc == 0:
                    prev_yaw = vehicle.get_transform().rotation.yaw
                    if len(route) > 0:
                        hlc = road_option_to_int(route.pop(0))
                    else:
                        hlc = random.choice([1, 2, 3])
                else:
                    hlc = prev_hlc
                    # calculate turned angle
                    cur_yaw = vehicle.get_transform().rotation.yaw
                    delta_yaw += calculate_delta_yaw(prev_yaw, cur_yaw)
                    prev_yaw = cur_yaw
            else:
                hlc = 0
            
            # detect whether the vehicle made the correct turn
            if prev_hlc != 0 and hlc == 0:
                print(f'turned {delta_yaw} degrees')
                # if command is Left or Right but didn't make turn
                if 75 < np.abs(delta_yaw) < 180:
                    if delta_yaw < 0 and prev_hlc != 1:
                        turning_infraction = True
                    elif delta_yaw > 0 and prev_hlc != 2:
                        turning_infraction = True
                # if command is Go Straight but turned
                elif prev_hlc != 3:
                    turning_infraction = True
                if turning_infraction:
                    print('Wrong Turn!!!')
                delta_yaw = 0
            
            prev_hlc = hlc
            

            # save frame data
            frame_data = {
                'hlc': hlc,
                'measurements': speed,
                'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
                'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data()))
            }

            # get traffic light status
            if not params.ignore_traffic_light:
                light_status = -1
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    light_status = traffic_light.get_state()
                frame_data['light'] = np.array([traffic_light_to_int(light_status)])
            

            if autopilot:
                control = vehicle.get_control()
            else:
                control = model_control(model, frame_data, ignore_traffic_light=params.ignore_traffic_light)
            vehicle.apply_control(control)
            
            # switch between autopilot and model
            cur_driver_count += 1
            if cur_driver_count >= 20:
                autopilot = not autopilot
                cur_driver_count = 0

            # only save data when agent recovers from noise AND skip repetitive still frames
            if still_frames < 50 and autopilot:
                frame_data = {
                    'frame': np.array([frame]),
                    'hlc': np.array([hlc]),
                    'light': np.array([traffic_light_to_int(light_status)]),
                    'controls': np.array([control.throttle, control.steer, control.brake]),
                    'measurements': np.array([speed]),
                    'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
                    'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data())),
                }
                for key, value in frame_data.items():
                    episode_data[key].append(value)
            print(f'hlc: {hlc}')

            driver = 'autopilot' if autopilot else 'model'
            print(f'{frame} {driver} throttle: {control.throttle}, brake: {control.brake}, steer: {control.steer}')

            frame += 1
            world.tick()
        
        with h5py.File(f'{params.dataset_path}/dagger_episode_{episode_cnt + 40}.hdf5', 'w') as file:
            for key, data_list in episode_data.items():
                data_array = np.array(data_list)
                file.create_dataset(key, data=data_array, maxshape=(None,)+data_array.shape[1:])
        episode_cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town01')
    parser.add_argument('--tm_port', type=int, default=8000)
    parser.add_argument('--episode_file', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n_vehicles', type=int, default=80)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=4000)
    parser.add_argument('--target_speed', type=int, default=40)
    parser.add_argument('--ignore_traffic_light', action="store_true")

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)

# python data_collector_dagger.py --dataset_path ./data --episode_file Town01_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v8.0_30 epochs.pth" --n_episodes 5
# python data_collector_dagger.py --dataset_path ./data --episode_file Town01_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v8.2.1_10 epochs.pth" --n_episodes 5