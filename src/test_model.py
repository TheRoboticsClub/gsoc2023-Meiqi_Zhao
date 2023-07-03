import argparse
import logging
import carla
from traffic_utils import spawn_vehicles, spawn_pedestrians
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils import carla_seg_to_array, carla_rgb_to_array,  road_option_to_int, int_to_road_option, read_routes
from test_utils import model_control, calculate_distance, DistanceTracker, calculate_stats
import numpy as np
import torch
from ModifiedDeepestLSTMTinyPilotNet.utils.ModifiedDeepestLSTMTinyPilotNet import DeepestLSTMTinyPilotNet, DeepestLSTMTinyPilotNetOneHot
import random
import pygame

has_collision = False
def collision_callback(data):
    global has_collision
    has_collision = True
    print('Collision detected!')

def main(params):

    # load model
    model = DeepestLSTMTinyPilotNetOneHot((288, 200, 6), 3, 4)
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

    # configure hybrid mode
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    # set up Pygame window
    pygame.init()
    WIDTH, HEIGHT = 1280, 720
    FONT_SIZE = 20
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.font.init()
    myfont = pygame.font.SysFont('Arial', FONT_SIZE)

    
    info = []
    info.append('----------------------------------------------\n')
    success_record = []
    distances = []
    for i in range(num_episodes):
        print(f"episode {i}")

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
        route = episode_config[1].copy()

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

        # Spawn vehicles
        vehicles_list = spawn_vehicles(world, client, params.n_vehicles, traffic_manager)

        # Spawn Walkers
        all_id, all_actors, pedestrians_list = spawn_pedestrians(world, client, params.n_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=0.0)
        
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
        while True:
            # update spectator's location to match the vehicle's
            transform = vehicle.get_transform()
            vehicle_location = transform.location
            # spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=50), carla.Rotation(pitch=-90)))
            offset = carla.Location(x=0.0, z=2)
            spectator.set_transform(carla.Transform(transform.location + offset, transform.rotation))

            # check if agent is done
            #print( calculate_distance(vehicle_location, end_point.location))
            if calculate_distance(vehicle_location, end_point.location) < 1.5 or has_collision or frame > params.max_frames_per_episode:
                if calculate_distance(vehicle_location, end_point.location) < 1.5: 
                    end_condition = "success"
                    success_record.append(1)
                if has_collision: 
                    end_condition = 'collision'
                    success_record.append(0)
                if frame > 4000: 
                    end_condition = 'max frames exceeded'
                    success_record.append(0)
                distances.append(dist_tracker.get_total_distance())

                info.append(f'episode {i} #{episode_config[0][0]} to #{episode_config[0][1]}: {end_condition}\n')

                print(f'Episode {i} ending')
                vehicle.destroy()
                for sensor in sensors: sensor.destroy()

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

            # calculate speed
            velocity = vehicle.get_velocity()
            speed = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)) #m/s to km/h 
            
            # calculate high-level command
            vehicle_location =vehicle.get_transform().location
            vehicle_waypoint = map.get_waypoint(vehicle_location)
            next_waypoint = vehicle_waypoint.next(10.0)[0]
            if vehicle_waypoint.is_junction or next_waypoint.is_junction:
                if prev_hlc == 0:
                    if len(route) > 0:
                        hlc = road_option_to_int(route.pop(0))
                    else:
                        hlc = random.choice([1, 2, 3])
                else:
                    hlc = prev_hlc
            else:
                hlc = 0
            prev_hlc = hlc
            
            # save frame data
            frame_data = {
                'hlc': hlc,
                'measurements': speed,
                'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
                'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data()))
            }

            # Apply control commands
            control = model_control(model, frame_data)
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

            #print(f'{frame} throttle: {control.throttle}, brake: {control.brake}, steer: {control.steer}')
            #print(f'{frame} speed: {speed}')

            # Next frame
            frame += 1
            dist_tracker.update(vehicle)
            #print(f"{frame-1} distance travelled: {dist_tracker.get_total_distance()}")
    
    # info.append(f'success rate: {success_cnt / num_episodes}\n')
    # if fail_cnt > 0:
    #     info.append(f'average distance traveled before collision (failed cases): {dist / fail_cnt}\n')
    success_record = np.array(success_record)
    distances = np.array(distances)
    success_rate, weighted_success_rate, average_fail_distance = calculate_stats(distances, success_record)
    info.append(f'success rate: {success_rate}')
    info.append(f'success rate weighted by track length: {weighted_success_rate}')
    info.append(f'average distance traveled in failed cases: {average_fail_distance}')
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
    parser.add_argument('--n_vehicles', type=int, default=80)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=4000)

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)

    # python test_model.py --episode_file Town02_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/models/v6.2.pth" --n_episodes 100