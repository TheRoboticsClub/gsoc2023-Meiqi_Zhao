import argparse
import logging
import carla
from traffic_utils import spawn_vehicles, spawn_pedestrians
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils import carla_seg_to_array, carla_rgb_to_array,  road_option_to_int, int_to_road_option, read_routes, traffic_light_to_int, calculate_delta_yaw
from test_utils import model_control, calculate_distance, DistanceTracker, calculate_stats
import numpy as np
import torch
from ModifiedDeepestLSTMTinyPilotNet.utils.ModifiedDeepestLSTMTinyPilotNet import PilotNetOneHot, PilotNetEmbeddingNoLight, PilotNetOneHotNoLight, PilotNetOneHotEnhanced
import random
import pygame

has_collision = False
collision_type = None
def collision_callback(data):
    global has_collision, collision_type
    has_collision = True
    collision_type = type(data.other_actor)
    print(f'Collision detected with {type(data.other_actor)}')

def main(params):
    # load model
    if params.ignore_traffic_light:
        #model = PilotNetEmbeddingNoLight((288, 200, 6), 3, 4)
        model = PilotNetOneHotNoLight((288, 200, 6), 3, 4)
    else:
        # model = PilotNetOneHotEnhanced((288, 200, 6), 2, 4, 4)
        model = PilotNetOneHot((288, 200, 6), 3, 4, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.load_state_dict(torch.load(params.model))
    model.to(device)
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
    # # set red light duration
    # list_actor = world.get_actors()
    # for actor_ in list_actor:
    #     if isinstance(actor_, carla.TrafficLight):
    #         #actor_.set_state(carla.TrafficLightState.Green) 
    #         actor_.set_green_time(1000.0)


    # configure hybrid mode
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    # set up Pygame window
    pygame.init()
    WIDTH, HEIGHT = 1280, 720
    #WIDTH, HEIGHT = 640, 360
    FONT_SIZE = 20
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.font.init()
    myfont = pygame.font.SysFont('Arial', FONT_SIZE)

    
    info = []
    info.append('----------------------------------------------\n')
    success_record = []
    distances = []
    driving_scores = []
    total_infractions = {'red_light': 0,
                       'nonjunction_time_out': 0,
                       'junction_time_out': 0,
                       'collision_vehicle': 0,
                       'collision_walker': 0,
                       'collision_other': 0,
                       'turning': 0}
    for i in range(num_episodes):
        print(f"episode {i}")

        # reset collision checker
        global has_collision, collision_type
        has_collision = False
        
        # Get the specific spawn points
        spawn_points = map.get_spawn_points()
        episode_config = random.choice(episode_configs)
        #episode_config = episode_configs[i]
        start_point = spawn_points[episode_config[0][0]]
        start_point = carla.Transform(carla.Location(x=55.3, y=105.6, z=1.37), carla.Rotation(pitch=0.000000, yaw=180.0, roll=0.000000))
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

        # set up rgb camera for Pygame window
        pygame_rgb_cam = RGBCamera(world, vehicle, size_x='1280', size_y='720')
        pygame_rgb_cam_sensor = pygame_rgb_cam.get_sensor()
        sensors.append(pygame_rgb_cam_sensor)

        # setup semantic segmentation camera
        seg_cam = SegmentationCamera(world, vehicle)
        seg_cam_sensor = seg_cam.get_sensor()
        sensors.append(seg_cam_sensor)

        collision_sensor = setup_collision_sensor(world, vehicle)
        collision_sensor.listen(collision_callback)
        sensors.append(collision_sensor)
        world.tick()
    
        frame = 0
        dist_tracker = DistanceTracker()
        prev_hlc = 0
        prev_yaw = 0
        delta_yaw = 0
        running_light = False
        prev_collision = False
        infractions = {'red_light': 0, # 0.7
                       'nonjunction_time_out': 0, # 0.7
                       'junction_time_out': 0, # 0.7
                       'collision_vehicle': 0, # 0.6
                       'collision_walker': 0, # 0.5
                       'collision_other': 0, # 0.65
                       'turning': 0} # 0.7
        turning_infraction = False
        while True:
            # update spectator's location to match the vehicle's
            transform = vehicle.get_transform()
            vehicle_location = transform.location
            # spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=50), carla.Rotation(pitch=-90)))
            offset = carla.Location(x=0.0, z=2)
            spectator.set_transform(carla.Transform(transform.location + offset, transform.rotation))

            # check if agent is done
            #print( calculate_distance(vehicle_location, end_point.location))
            if has_collision:
                if not prev_collision:
                    if collision_type == carla.libcarla.Vehicle:
                        infractions['collision_vehicle'] += 1
                    elif collision_type == carla.libcarla.Walker:
                        infractions['collision_walker'] += 1
                    else:
                        infractions['collision_other'] += 1
                    print(infractions)
                    prev_collision = True
            else:
                prev_collision = False
            has_collision = False
            collision_type = None
                

            if calculate_distance(vehicle_location, end_point.location) < 1.5 or frame > params.max_frames_per_episode or turning_infraction:
                if calculate_distance(vehicle_location, end_point.location) < 1.5: 
                    end_condition = "success"
                    success_record.append(1)
                # if has_collision: 
                #     end_condition = 'collision'
                #     success_record.append(0)
                if frame > params.max_frames_per_episode: 
                    end_condition = 'max frames exceeded'
                    success_record.append(0)
                    vehicle_location =vehicle.get_transform().location
                    vehicle_waypoint = map.get_waypoint(vehicle_location)
                    if vehicle_waypoint.is_junction:
                        infractions['junction_time_out'] += 1
                    else:
                        infractions['nonjunction_time_out'] += 1

                if turning_infraction:
                    end_condition = 'made wrong turn'
                    success_record.append(0)
                    infractions['turning'] += 1
                distances.append(dist_tracker.get_total_distance())

                info.append(f'episode {i} #{episode_config[0][0]} to #{episode_config[0][1]}: {end_condition}')
                if end_condition == 'success': 
                    route_completion = 1.0
                else:
                    route_completion = dist_tracker.get_total_distance() / route_length
                
                infraction_penalty = 0.5 ** infractions['collision_walker'] * \
                                     0.6 ** infractions['collision_vehicle'] * \
                                     0.65 ** infractions['collision_other'] * \
                                     0.7 ** infractions['red_light'] * \
                                     0.7 ** (infractions['junction_time_out'] + infractions['nonjunction_time_out']) * \
                                     0.7 ** infractions['turning']
                driving_score = route_completion * infraction_penalty
                for infraction_type in ['collision_walker', 'collision_vehicle', 'collision_other', 'red_light', 'nonjunction_time_out', 'junction_time_out', 'turning']:
                    total_infractions[infraction_type] += infractions[infraction_type]
                info.append(f'route compleition: {route_completion}')
                info.append(f'infraction penalty: {infraction_penalty}')
                info.append(f'driving score: {driving_score}\n')
                driving_scores.append(driving_score)
                

                print(f'Episode {i} ending')
                vehicle.destroy()
                for sensor in sensors: sensor.destroy()

                print('\ndestroying %d vehicles' % len(vehicles_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                # stop walker controllers (list is [controller, actor, controller, actor ...])
                for k in range(0, len(all_id), 2):
                    all_actors[k].stop()
                print('\ndestroying %d walkers' % len(pedestrians_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
                # world.tick()

                break
            
            world.tick()

            # calculate speed
            velocity = vehicle.get_velocity()
            speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed = 3.6 * speed_m_s #m/s to km/h 
            
            # calculate next high-level command
            vehicle_location =vehicle.get_transform().location
            vehicle_waypoint = map.get_waypoint(vehicle_location)

            next_to_junction = False
            for j in range(1, 11):
                next_waypoint = vehicle_waypoint.next(j * 1.0)[0]
                if next_waypoint.is_junction:
                    next_to_junction = True
            if vehicle_waypoint.is_junction or next_to_junction:
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
                if 60 < np.abs(delta_yaw) < 180:
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
            if not params.ignore_traffic_light:
                # get traffic light status
                light_status = -1
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    light_status = traffic_light.get_state()
                    traffic_light_location = traffic_light.get_transform().location
                    distance_to_traffic_light = np.sqrt((vehicle_location.x - traffic_light_location.x)**2 + (vehicle_location.y - traffic_light_location.y)**2)
                    logging.debug("distance to traffic light: ", distance_to_traffic_light)
                    logging.debug('speed: ', speed_m_s)
                    if light_status == carla.libcarla.TrafficLightState.Red and distance_to_traffic_light < 6 and speed_m_s > 5:
                        if not running_light:
                            running_light = True
                            infractions['red_light'] += 1
                            print(infractions)
                            logging.debug('running light count ', infractions['red_light'])
                    else:
                        running_light = False
                frame_data['light'] = np.array([traffic_light_to_int(light_status)])
                
                

            # Apply control commands
            control = model_control(model, frame_data, ignore_traffic_light=params.ignore_traffic_light, device=device)
            vehicle.apply_control(control)

            # Pygame visualization
            image_surface = pygame.surfarray.make_surface(carla_rgb_to_array(pygame_rgb_cam.get_sensor_data()).swapaxes(0, 1))
            image_surface = pygame.transform.scale(image_surface, (WIDTH, HEIGHT))
            screen.fill((0, 0, 0))
            screen.blit(image_surface, (0, 0))
            line1 = f'Command: {int_to_road_option(hlc)}'
            line2 = f'Distance traveled: {round(dist_tracker.get_total_distance())}m'
            line3 = f'Euclidean distance to goal: {round(calculate_distance(vehicle_location, end_point.location))}m'
            textsurface1 = myfont.render(line1, False, (0, 0, 255))
            textsurface2 = myfont.render(line2, False, (0, 0, 255))
            textsurface3 = myfont.render(line3, False, (0, 0, 255))
            window_height = screen.get_height()
            y1 = window_height - textsurface2.get_height() * 3
            y2 = window_height - textsurface2.get_height() * 2
            y3 = window_height - textsurface2.get_height()
            screen.blit(textsurface1, (0, y1))
            screen.blit(textsurface2, (0, y2))
            screen.blit(textsurface3, (0, y3))
            pygame.display.flip()

            print(f'{frame} throttle: {control.throttle}, brake: {control.brake}, steer: {control.steer}')
            #print(f'{frame} speed: {speed}')

            # Next frame
            frame += 1
            dist_tracker.update(vehicle)
    

    success_record = np.array(success_record)
    distances = np.array(distances)
    success_rate, weighted_success_rate, average_fail_distance = calculate_stats(distances, success_record)
    info.append(f'success rate: {success_rate}')
    info.append(f'success rate weighted by track length: {weighted_success_rate}')
    info.append(f'average distance traveled in failed cases: {average_fail_distance}')
    info.append(f'average driving score: {np.mean(driving_scores)}')
    info.append(f'infractions: {total_infractions}')
    info.append('--------------------END-----------------------\n')
    for line in info: print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town02')
    parser.add_argument('--tm_port', type=int, default=8000)
    parser.add_argument('--episode_file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n_vehicles', type=int, default=50)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=6000)
    parser.add_argument('--ignore_traffic_light', action="store_true")

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)

    # python test_model.py --episode_file Town02_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v6.2.pth" --n_episodes 5 --ignore_traffic_light
    # python test_model.py --episode_file Town02_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v6.3.pth" --n_episodes 5 --ignore_traffic_light
    # python test_model.py --episode_file Town02_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v6.4_20epochs.pth" --n_episodes 5
    # python test_model.py --episode_file Town02_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v7.2_10 epochs.pth" --n_episodes 5