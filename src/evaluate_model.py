import argparse
import logging
import random

import carla
from utils.traffic import spawn_vehicles, spawn_pedestrians, cleanup
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils.preprocess import carla_seg_to_array, carla_rgb_to_array,  road_option_to_int, int_to_road_option, read_routes, traffic_light_to_int
from utils.test_utils import model_control, calculate_distance, DistanceTracker, calculate_delta_yaw
from utils.high_level_command import HighLevelCommandLoader
from ModifiedDeepestLSTMTinyPilotNet.utils.ModifiedDeepestLSTMTinyPilotNet import PilotNetOneHot, PilotNetOneHotNoLight
from utils.metrics import MetricsRecorder

import numpy as np
import torch
import pygame


PYGAME_WINDOW_SIZE = (1280, 720)

has_collision = False
collision_type = None
def collision_callback(data):
    global has_collision, collision_type
    has_collision = True
    collision_type = type(data.other_actor)
    logging.debug(f'Collision detected with {type(data.other_actor)}')

def load_model(model_path, device, ignore_traffic_light=False, one_hot=True, combined_control=True):
    num_labels = 2 if combined_control else 3
    if ignore_traffic_light:
        model = PilotNetOneHotNoLight((288, 200, 6), num_labels, 4)
    else:
        model = PilotNetOneHot((288, 200, 6), num_labels, 4, 4)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model

def setup_carla_world(params):
    client = carla.Client(params.ip, params.port)
    client.set_timeout(params.timeout)
    
    world = client.get_world()
    map = world.get_map()
    if map.name.split('/')[-1] != params.map:
        world = client.load_world(params.map)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    return world, client

def setup_traffic_manager(client, params):
    traffic_manager = client.get_trafficmanager(params.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    return traffic_manager

def setup_sensors(world, vehicle):
    rgb_cam = RGBCamera(world, vehicle)
    seg_cam = SegmentationCamera(world, vehicle)
    collision_sensor = setup_collision_sensor(world, vehicle)
    collision_sensor.listen(collision_callback)
    pygame_cam = RGBCamera(world, vehicle, size_x='1280', size_y='720')

    sensors = [rgb_cam.get_sensor(), seg_cam.get_sensor(), pygame_cam.get_sensor(), collision_sensor]
    return rgb_cam, seg_cam, pygame_cam, sensors

def initialize_pygame(size, font_size=20):
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.font.init()
    myfont = pygame.font.SysFont('Arial', font_size)
    return screen, myfont

def update_pygame_frame(screen, myfont, size, image, text):
    image_surface = pygame.surfarray.make_surface(image)
    image_surface = pygame.transform.scale(image_surface, size)
    screen.fill((0, 0, 0))
    screen.blit(image_surface, (0, 0))

    window_height = screen.get_height()
    for i, line in enumerate(text):
        text_surface = myfont.render(line, False, (0, 0, 255))
        y_val = window_height - text_surface.get_height() * (i+1)
        screen.blit(text_surface, (0, y_val))

    pygame.display.flip()

def sample_route(world, episode_configs):
    spawn_points = world.get_map().get_spawn_points()
    episode_config = random.choice(episode_configs)
    start_point = spawn_points[episode_config[0][0]]
    end_point = spawn_points[episode_config[0][1]]
    logging.info(f"from spawn point #{episode_config[0][0]} to #{episode_config[0][1]}")
    route_length = episode_config[1]
    route = episode_config[2].copy()
    return episode_config, start_point, end_point, route_length, route

def initialize_vehicle(world, start_point):
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.filter('model3')[0]
    blueprint.set_attribute('role_name', 'hero')
    vehicle = world.spawn_actor(blueprint, start_point)
    return vehicle

def initialize_traffic(world, client, traffic_manager):
    all_id, all_actors, pedestrians_list, vehicles_list = [], [], [], []
    if params.n_vehicles > 0:
        vehicles_list = spawn_vehicles(world, client, params.n_vehicles, traffic_manager)

    if params.n_pedestrians > 0:
        all_id, all_actors, pedestrians_list = spawn_pedestrians(world, client, params.n_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=0.0)
    
    logging.info('spawned %d vehicles and %d walkers.' % (len(vehicles_list), len(pedestrians_list)))
    return all_id, all_actors, pedestrians_list, vehicles_list

def check_collision(metrics_recorder, prev_collision):
    global has_collision, collision_type
    if has_collision:
        if not prev_collision:
            if collision_type == carla.libcarla.Vehicle:
                metrics_recorder.record('collision_vehicle')
            elif collision_type == carla.libcarla.Walker:
                metrics_recorder.record('collision_walker')
            else:
                    metrics_recorder.record('collision_other')
            prev_collision = True
    else:
        prev_collision = False
    has_collision = False
    collision_type = None
    return prev_collision

def check_end_conditions(world, vehicle, end_point, frame, params, dist_tracker, route_length, metrics_recorder, turning_infraction):
    reached_destination = calculate_distance(vehicle.get_transform().location, end_point.location) < 1.5
    exceeded_max_frames = frame > params.max_frames_per_episode

    if not (reached_destination or exceeded_max_frames or turning_infraction):
        return False

    # Determine and record the end condition
    if reached_destination:
        metrics_recorder.success_record.append(1)
        end_condition = "success"
    elif exceeded_max_frames:
        metrics_recorder.success_record.append(0)
        if world.get_map().get_waypoint(vehicle.get_transform().location).is_junction:
            metrics_recorder.record('junction_time_out')
        else:
            metrics_recorder.record('nonjunction_time_out')
        end_condition = 'max frames exceeded'
    else:
        metrics_recorder.success_record.append(0)
        metrics_recorder.record('turning')
        end_condition = 'made wrong turn'

    metrics_recorder.distances.append(dist_tracker.get_total_distance())
    metrics_recorder.route_completion = dist_tracker.get_total_distance() / route_length if end_condition != 'success' else 1.0
    metrics_recorder.info.append(f'termination: {end_condition}')

    return True

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'device: {device}')
    model = load_model(params.model, device, ignore_traffic_light=params.ignore_traffic_light, combined_control=params.combined_control)

    world, client = setup_carla_world(params)
    traffic_manager = setup_traffic_manager(client, params)
    episode_configs = read_routes(params.episode_file)
    
    screen, myfont = initialize_pygame(size=PYGAME_WINDOW_SIZE)

    metrics_recorder = MetricsRecorder()
    
    for i in range(params.n_episodes):
        logging.info(f"-----episode {i + 1}-----")
        
        global has_collision, collision_type
        has_collision = False

        episode_config, start_point, end_point, route_length, route = sample_route(world, episode_configs)

        vehicle = initialize_vehicle(world, start_point)

        all_id, all_actors, pedestrians_list, vehicles_list = initialize_traffic(world, client, traffic_manager)

        rgb_cam, seg_cam, pygame_cam, sensors = setup_sensors(world, vehicle)

        hlc_loader = HighLevelCommandLoader(vehicle, world.get_map(), route)
        
        for _ in range(10):
            world.tick()

        frame = 0
        dist_tracker = DistanceTracker()
        prev_hlc = 0
        prev_yaw = 0
        delta_yaw = 0
        running_light = False
        prev_collision = False
        metrics_recorder.start_episode()
        turning_infraction = False
       
        while True:
            transform = vehicle.get_transform()
            vehicle_location = transform.location

            prev_collision = check_collision(metrics_recorder, prev_collision)
                
            if check_end_conditions(world, vehicle, end_point, frame, params, dist_tracker, route_length, metrics_recorder, turning_infraction):
                metrics_recorder.end_episode()
                logging.info(f'episode {i + 1} ending')
                cleanup(vehicles_list, pedestrians_list, all_id, all_actors, vehicle, sensors, client)
                
                break
            
            world.tick()

            velocity = vehicle.get_velocity()
            speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed = 3.6 * speed_m_s #m/s to km/h 
            
            # read next high-level command or choose a random direction
            hlc = hlc_loader.get_next_hlc()
            if hlc != 0:
                if prev_hlc == 0:
                    prev_yaw = vehicle.get_transform().rotation.yaw
                else:
                    cur_yaw = vehicle.get_transform().rotation.yaw
                    delta_yaw += calculate_delta_yaw(prev_yaw, cur_yaw)
                    prev_yaw = cur_yaw
            
            # detect whether the vehicle made the correct turn
            if prev_hlc != 0 and hlc == 0:
                logging.info(f'turned {delta_yaw} degrees')
                if 60 < np.abs(delta_yaw) < 180:
                    if delta_yaw < 0 and prev_hlc != 1:
                        turning_infraction = True
                    elif delta_yaw > 0 and prev_hlc != 2:
                        turning_infraction = True
                elif prev_hlc != 3:
                    turning_infraction = True
                if turning_infraction:
                   logging.info('Wrong Turn!!!')
                delta_yaw = 0
            
            prev_hlc = hlc

            frame_data = {
                'hlc': hlc,
                'measurements': speed,
                'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
                'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data()))
            }

            # detect running red light
            if not params.ignore_traffic_light:
                light_status = -1
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    light_status = traffic_light.get_state()
                    traffic_light_location = traffic_light.get_transform().location
                    distance_to_traffic_light = np.sqrt((vehicle_location.x - traffic_light_location.x)**2 + (vehicle_location.y - traffic_light_location.y)**2)
                    if light_status == carla.libcarla.TrafficLightState.Red and distance_to_traffic_light < 6 and speed_m_s > 5:
                        if not running_light:
                            running_light = True
                            metrics_recorder.record('red_light')
                            logging.debug('running light count ', metrics_recorder.infractions['red_light'])
                    else:
                        running_light = False
                frame_data['light'] = np.array([traffic_light_to_int(light_status)])
            
            
            # update pygame frame
            line1 = f'Euclidean distance to goal: {round(calculate_distance(vehicle_location, end_point.location))}m'
            line2 = f'Distance traveled: {round(dist_tracker.get_total_distance())}m'
            line3 = f'Command: {int_to_road_option(hlc)}'
            text = [line1, line2, line3]
            update_pygame_frame(screen, myfont, PYGAME_WINDOW_SIZE, carla_rgb_to_array(pygame_cam.get_sensor_data()).swapaxes(0, 1), text)

            # apply vehicle control
            control = model_control(model, frame_data, ignore_traffic_light=params.ignore_traffic_light, device=device, combined_control=params.combined_control)
            vehicle.apply_control(control)

            #logging.debug(f'{frame} throttle: {control.throttle}, brake: {control.brake}, steer: {control.steer}')
            #logging.debug(f'{frame} speed: {speed}')

            frame += 1
            dist_tracker.update(vehicle)

    metrics_recorder.get_evaluation_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=int, default=200)
    parser.add_argument('--map', default='Town02')
    parser.add_argument('--tm_port', type=int, default=8000)
    parser.add_argument('--episode_file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n_vehicles', type=int, default=50)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=6000)
    parser.add_argument('--ignore_traffic_light', action="store_true")
    parser.add_argument('--combined_control', action="store_true")

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)

    # python evaluate_model.py --episode_file test_suites/Town02_All.txt --model "ModifiedDeepestLSTMTinyPilotNet/v10.0.pth" --n_episodes 5 --combined_control