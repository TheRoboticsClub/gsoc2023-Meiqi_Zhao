import argparse
import random
import logging
import carla
from traffic_utils import spawn_vehicles, spawn_pedestrians
from agents.navigation.behavior_agent import BehaviorAgent
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils import vehicle_control_to_dict, carla_seg_to_array, carla_rgb_to_array
import time
import numpy as np
import pickle

has_collision = False

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_start_end_points(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    points = [(int(line.split()[0]), int(line.split()[1])) for line in lines]
    return points

def collision_callback(data):
    global has_collision
    has_collision = True
    print('Collision detected!')

def main(params):
    episode_configs = read_start_end_points(params.episode_file)

    # create carla client
    client = carla.Client(params.ip, params.port)
    client.set_timeout(200.0)

    # load map
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != params.map:
        world = client.load_world(params.map)
    map = world.get_map()
    
    # set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # set up traffic manager
    traffic_manager = client.get_trafficmanager(params.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    # configure hybrid mode
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    episode_cnt = 0
    while episode_cnt < params.n_episodes:
        print(f"episode {episode_cnt}")

        # reset collision checker
        global has_collision
        has_collision = False
        
        # Get the specific spawn points
        spawn_points = map.get_spawn_points()
        episode_config = random.choice(episode_configs)
        start_point = spawn_points[episode_config[0]]
        end_point = spawn_points[episode_config[1]]
        print(f"from spawn point #{episode_config[0]} to #{episode_config[1]}")

        # get the blueprint for this vehicle
        blueprint_library = world.get_blueprint_library()
        blueprint = blueprint_library.filter('model3')[0]
    
        # spawn the vehicle
        blueprint.set_attribute('role_name', 'hero')
        vehicle = world.spawn_actor(blueprint, start_point)

        # spawn the vehicle
        blueprint.set_attribute('role_name', 'leading_vehicle')
        vehicle1 = world.spawn_actor(blueprint, spawn_points[33])
        vehicle1.set_autopilot(True)
        vehicle2 = world.spawn_actor(blueprint, spawn_points[173])
        vehicle2.set_autopilot(True)

        # set spectator
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)))

        for i in range(10):
            world.tick()

        # Spawn vehicles
        vehicles_list = spawn_vehicles(world, client, params.n_vehicles, traffic_manager)

        # Spawn Walkers
        all_id, all_actors, pedestrians_list = spawn_pedestrians(world, client, params.n_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=0.0)
        
        print('spawned %d vehicles and %d walkers.' % (len(vehicles_list), len(pedestrians_list)))

        # create and setup agent
        agent = BehaviorAgent(vehicle, behavior='normal')
        agent.ignore_vehicles(False)
        agent.ignore_traffic_lights(False)
        agent.follow_speed_limits(False)
        agent.ignore_stop_signs(False)
        agent.set_destination(end_point.location)

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
        world.tick()
    
        frame = 0
        episode_data = []
        while True:
            # Check if agent is done
            if agent.done() or frame >= params.max_frames_per_episode or has_collision:

                print("The target has been reached, episode ending")
                vehicle.destroy()
                for sensor in sensors: sensor.destroy()
                vehicle1.destroy()
                vehicle2.destroy()

                print('\ndestroying %d vehicles' % len(vehicles_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                # stop walker controllers (list is [controller, actor, controller, actor ...])
                for i in range(0, len(all_id), 2):
                    all_actors[i].stop()
                print('\ndestroying %d walkers' % len(pedestrians_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
                world.tick()

                break

            world.tick()

            # Apply control commands
            control = agent.run_step()
            vehicle.apply_control(control)
            velocity = vehicle.get_velocity()
            speed = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

            # Update spectator's location to match the vehicle's
            vehicle_location = vehicle.get_transform().location
            spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            # save frame data
            frame_data = {
                'frame': frame,
                'controls': vehicle_control_to_dict(control),
                'measurements': speed,
                'rgb': carla_rgb_to_array(rgb_cam.get_sensor_data()),
                'segmentation': carla_seg_to_array(seg_cam.get_sensor_data())
            }
            episode_data.append(frame_data)

            # print(f'{frame} throttle: {control.throttle}, brake: {control.brake}, steer: {control.steer}')
            # print(f'{frame} speed: {speed}')

            # Next frame
            frame += 1
        
        if not has_collision:
            # save data
            save_data(f'data/episode_{episode_cnt}.pkl', episode_data)

            episode_cnt += 1
        
        #TODO: save data in pickle format


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town01')
    parser.add_argument('--tm_port', type=int, default=8000)
    parser.add_argument('--episode_file', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--n_vehicles', type=int, default=80)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=4000)

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)
