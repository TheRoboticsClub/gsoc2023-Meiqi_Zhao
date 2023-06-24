import argparse
import random
import logging
import carla
from traffic_utils import spawn_vehicles, spawn_pedestrians
from sensors import RGBCamera, SegmentationCamera, setup_collision_sensor
from utils import carla_seg_to_array, carla_rgb_to_array
import numpy as np
import torch
from ModifiedDeepestLSTMTinyPilotNet.utils.ModifiedDeepestLSTMTinyPilotNet import DeepestLSTMTinyPilotNet
import matplotlib.pyplot as plt

has_collision = False
def read_start_end_points(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    points = [(int(line.split()[0]), int(line.split()[1])) for line in lines]
    return points

def collision_callback(data):
    global has_collision
    has_collision = True
    print('Collision detected!')


def model_control(model, frame_data):
    global counter 
    img, speed = preprocess_data2(frame_data)

    prediction = model(img, speed).detach().numpy().flatten()
    #print(f"prediction: {prediction}")

    throttle, steer, brake = prediction
    throttle = float(throttle)
    brake = float(brake)
    if brake < 0.05: brake = 0.0
    
    steer = (float(steer) * 2.0) - 1.0

    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

def preprocess_data(data):
    rgb = torch.tensor(data['rgb'].copy(), dtype=torch.float32).permute(2, 0, 1)
    rgb /= 255.0
    
    segmentation = torch.tensor(data['segmentation'].copy(), dtype=torch.float32).permute(2, 0, 1)
    segmentation /= 255.0

    img = torch.cat((rgb, segmentation), dim=0)
    img = img.unsqueeze(0)
    
    speed = torch.tensor(data['measurements'].copy(), dtype=torch.float32)
    speed /= 90.0
    speed = speed.unsqueeze(0)

    return img, speed

def preprocess_data2(data):
    rgb = data['rgb'].copy()
    segmentation = data['segmentation'].copy()

    rgb, segmentation = filter_classes(rgb, segmentation)

    rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
    rgb /= 255.0
    
    segmentation = torch.tensor(segmentation, dtype=torch.float32).permute(2, 0, 1)
    segmentation /= 255.0

    img = torch.cat((rgb, segmentation), dim=0)
    img = img.unsqueeze(0)
    
    speed = torch.tensor(data['measurements'].copy(), dtype=torch.float32)
    speed /= 90.0
    speed = speed.unsqueeze(0)

    return img, speed


def filter_classes(rgb, seg, classes_to_keep=[4, 6, 7, 10]):
    classes = {
            0: [0, 0, 0],         # Unlabeled
            1: [70, 70, 70],      # Buildings
            2: [100, 40, 40],     # Fences
            3: [55, 90, 80],      # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 142],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0],    # TrafficSigns
            13: [70, 130, 180],   # Sky
            14: [81, 0, 81],      # Ground
            15: [150, 100, 100],  # Bridge
            16: [230, 150, 140],  # RailTrack
            17: [180, 165, 180],  # GuardRail
            18: [250, 170, 30],   # TrafficLight
            19: [110, 190, 160],  # Static
            20: [170, 120, 50],   # Dynamic
            21: [45, 60, 150],    # Water
            22: [145, 170, 100],  # Terrain
        }

    classes_to_keep_rgb = np.array([classes[class_id] for class_id in classes_to_keep])

    # Create a mask of pixels to keep
    mask = np.isin(seg, classes_to_keep_rgb).all(axis=-1)

    # Initialize filtered images as black images
    filtered_seg = np.zeros_like(seg)
    filtered_rgb = np.zeros_like(rgb)

    # Use the mask to replace the corresponding pixels in the filtered images
    filtered_seg[mask] = seg[mask]
    filtered_rgb[mask] = rgb[mask]

    return filtered_rgb, filtered_seg


# Calculate the distance from the current vehicle location to the target location.
def calculate_distance(location_1, location_2):
    dx = location_1.x - location_2.x
    dy = location_1.y - location_2.y
    return np.sqrt(dx*dx + dy*dy)

class DistanceTracker:
    def __init__(self):
        self.prev_location = None
        self.total_distance = 0.0

    def update(self, vehicle):
        location = vehicle.get_transform().location
        if self.prev_location is not None:
            self.total_distance += np.sqrt((location.x - self.prev_location.x) ** 2 +
                                             (location.y - self.prev_location.y) ** 2)
        self.prev_location = location

    def get_total_distance(self):
        return self.total_distance


def main(params):

    # load model
    model = DeepestLSTMTinyPilotNet((288, 200, 6), 3)
    model.load_state_dict(torch.load("ModifiedDeepestLSTMTinyPilotNet/v5.4_20 epochs.pth", map_location=torch.device('cpu')))
    model.eval()

    # load episodes
    episode_configs = read_start_end_points('Town02_RightTurn.txt')
    num_episodes = len(episode_configs)

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

    success_cnt = 0
    for i in range(num_episodes):
        print(f"episode {i}")

        # reset collision checker
        global has_collision
        has_collision = False
        
        # Get the specific spawn points
        spawn_points = map.get_spawn_points()
        #episode_config = random.choice(episode_configs)
        episode_config = episode_configs[i]
        start_point = spawn_points[episode_config[0]]
        end_point = spawn_points[episode_config[1]]
        print(f"from spawn point #{episode_config[0]} to #{episode_config[1]}")

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
        offset = carla.Location(x=0.8, z=2)
        spectator.set_transform(carla.Transform(transform.location + offset, transform.rotation))
        

        for i in range(10):
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
        while True:
            # Update spectator's location to match the vehicle's
            transform = vehicle.get_transform()
            vehicle_location = transform.location
            # spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=50), carla.Rotation(pitch=-90)))
            offset = carla.Location(x=0.8, z=2)
            spectator.set_transform(carla.Transform(transform.location + offset, transform.rotation))

            # Check if agent is done
            #print( calculate_distance(vehicle_location, end_point.location))
            if calculate_distance(vehicle_location, end_point.location) < 1 or has_collision or frame > 4000:
                print("Episode ending")
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

            velocity = vehicle.get_velocity()
            speed = (3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)) #m/s to km/h 

            # save frame data
            frame_data = {
                'measurements': speed,
                'rgb': np.copy(carla_rgb_to_array(rgb_cam.get_sensor_data())),
                'segmentation': np.copy(carla_seg_to_array(seg_cam.get_sensor_data()))
            }

            # Apply control commands
            control = model_control(model, frame_data)
            vehicle.apply_control(control)

            #print(f'{frame} throttle: {control.throttle}, brake: {control.brake}, steer: {control.steer}')
            #print(f'{frame} speed: {speed}')

            # Next frame
            frame += 1
            dist_tracker.update(vehicle)
            print(f"{frame-1} distance travelled: {dist_tracker.get_total_distance()}")
        
        if not has_collision:
            success_cnt += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town02')
    parser.add_argument('--tm_port', type=int, default=8002)
    # parser.add_argument('--episode_file', required=True)
    # parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--n_vehicles', type=int, default=80)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--max_frames_per_episode', type=int, default=2000)

    params = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(params)
