import argparse
import random
import logging
import carla
import numpy as np
import h5py

from traffic_utils import spawn_vehicles, spawn_pedestrians, get_traffic_light_status, cleanup
from agent import NoisyTrafficManagerAgent
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils import (carla_seg_to_array, carla_rgb_to_array, road_option_to_int,
                   read_routes, traffic_light_to_int)

has_collision = False
def collision_callback(data):
    global has_collision
    has_collision = True

def setup_carla_world(params):
    client = carla.Client(params.ip, params.port)
    client.set_timeout(params.timeout)
    
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != params.map:
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

def main_loop(world, client, traffic_manager, params):
    episode_configs = read_routes(params.episode_file)
    episode_cnt = 0

    while episode_cnt < params.n_episodes:
        logging.info(f'episode {episode_cnt + 1}')
        handle_episode(world, client, traffic_manager, episode_configs, params, episode_cnt)
        episode_cnt += 1

def handle_episode(world, client, traffic_manager, episode_configs, params, episode_cnt):
    global has_collision
    has_collision = False

    episode_config, vehicle = setup_episode(world, traffic_manager, episode_configs, params)
    agent, spectator = setup_agent_and_spectator(vehicle, traffic_manager, episode_config, world)

    all_id, all_actors, pedestrians_list, vehicles_list = spawn_dynamic_agents(world, client, traffic_manager, params)

    rgb_cam, seg_cam, sensors = setup_sensors(world, vehicle)
    for _ in range(10):
        world.tick()

    collect_data_for_episode(vehicle, agent, spectator, world, params, episode_cnt, rgb_cam, seg_cam)

    cleanup(vehicles_list, pedestrians_list, all_id, all_actors, vehicle, sensors, client)

def setup_episode(world, traffic_manager, episode_configs, params):
    spawn_points = world.get_map().get_spawn_points()
    episode_config = random.choice(episode_configs)
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.filter('model3')[0]
    blueprint.set_attribute('role_name', 'hero')
    start_point = spawn_points[episode_config[0][0]]
    vehicle = world.spawn_actor(blueprint, start_point)

    configure_vehicle_for_traffic_manager(vehicle, traffic_manager, params)

    return episode_config, vehicle

def configure_vehicle_for_traffic_manager(vehicle, traffic_manager, params):
    vehicle.set_autopilot(True)
    if params.ignore_traffic_light:
        traffic_manager.ignore_lights_percentage(vehicle, 100)
    traffic_manager.ignore_signs_percentage(vehicle, 100)
    traffic_manager.distance_to_leading_vehicle(vehicle, 4.0)
    traffic_manager.set_desired_speed(vehicle, params.target_speed)

def setup_agent_and_spectator(vehicle, traffic_manager, episode_config, world):
    agent = NoisyTrafficManagerAgent(vehicle, traffic_manager)
    route = episode_config[2]
    end_point = world.get_map().get_spawn_points()[episode_config[0][1]]
    agent.set_route(route, end_point)

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

    return agent, spectator

def spawn_dynamic_agents(world, client, traffic_manager, params):
    all_id, all_actors, pedestrians_list, vehicles_list = [], [], [], []
    
    if params.n_vehicles > 0:
        vehicles_list = spawn_vehicles(world, client, params.n_vehicles, traffic_manager)

    if params.n_pedestrians > 0:
        all_id, all_actors, pedestrians_list = spawn_pedestrians(world, client, params.n_pedestrians)

    logging.info('spawned %d vehicles and %d walkers.' % (len(vehicles_list), len(pedestrians_list)))
    
    return all_id, all_actors, pedestrians_list, vehicles_list

def setup_sensors(world, vehicle):
    rgb_cam = RGBCamera(world, vehicle)
    seg_cam = SegmentationCamera(world, vehicle)
    collision_sensor = setup_collision_sensor(world, vehicle)
    collision_sensor.listen(collision_callback)
    sensors = [rgb_cam.get_sensor(), seg_cam.get_sensor(), collision_sensor]
    return rgb_cam, seg_cam, sensors

def collect_data_for_episode(vehicle, agent, spectator, world, params, episode_cnt, rgb_cam, seg_cam):
    episode_data = {
        'frame': [],
        'hlc': [],
        'light': [],
        'controls': [],
        'measurements': [],
        'rgb': [],
        'segmentation': []
    }

    frame = 0
    while True:
        if should_end_episode(agent, frame, params):
            break

        update_spectator(vehicle, spectator)
        process_frame(vehicle, agent, episode_data, frame, rgb_cam, seg_cam)
        world.tick()

        frame += 1

    if not has_collision and frame < params.max_frames_per_episode:
        save_episode_data(episode_data, params.dataset_path, episode_cnt)


def should_end_episode(agent, frame, params):
    done = False
    if agent.done():
        logging.info("The target has been reached, episode ending")
        done = True
    elif frame >= params.max_frames_per_episode:
        logging.info("Maximum frames reached, episode ending")
        done = True
    elif has_collision:
        logging.info("Collision detected! episode ending")
        done = True
    return done


def process_frame(vehicle, agent, episode_data, frame, rgb_cam, seg_cam):
    # apply control commands
    control, noisy_control = agent.run_step()
    if noisy_control:
        vehicle.apply_control(noisy_control)
    velocity = vehicle.get_velocity()
    speed_km_h = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)) #m/s to km/h
    
    # record data
    if (not agent.noise):
        frame_data = {
            'frame': np.array([frame]),
            'hlc': np.array([road_option_to_int(agent.get_next_action())]),
            'light': np.array([traffic_light_to_int(get_traffic_light_status(vehicle))]),
            'controls': np.array([control.throttle, control.steer, control.brake]),
            'measurements': np.array([speed_km_h]),
            'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
            'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data())),
        }
        for key, value in frame_data.items():
            episode_data[key].append(value)


def update_spectator(vehicle, spectator):
    vehicle_location = vehicle.get_transform().location
    spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=50), carla.Rotation(pitch=-90)))
    
def save_episode_data(episode_data, dataset_path, episode_cnt):
    with h5py.File(f'{dataset_path}/episode_{episode_cnt + 1}.hdf5', 'w') as file:
        for key, data_list in episode_data.items():
            data_array = np.array(data_list)
            file.create_dataset(key, data=data_array, maxshape=(None,)+data_array.shape[1:])

def main(params):
    world, client = setup_carla_world(params)
    traffic_manager = setup_traffic_manager(client, params)
    world.tick()
    main_loop(world, client, traffic_manager, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town01')
    parser.add_argument('--tm_port', type=int, default=8000)
    parser.add_argument('--timeout', type=int, default=100)
    parser.add_argument('--episode_file', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--n_vehicles', type=int, default=80)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=8000)
    parser.add_argument('--target_speed', type=int, default=40)
    parser.add_argument('--ignore_traffic_light', action="store_true")

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)

# Examples: 
# python data_collector.py --dataset_path ./data --episode_file test_suites/Town01_Junctions.txt --n_episodes 80 
# python data_collector.py --dataset_path ./data --episode_file test_suites/Town01_All.txt --n_episodes 2 --ignore_traffic_light