import torch
import numpy as np
import carla
import torch.nn.functional as F

def model_control(model, frame_data):
    global counter 
    img, speed, hlc = preprocess_data2(frame_data)

    prediction = model(img, speed, hlc).detach().numpy().flatten()
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

    hlc = torch.tensor(data['hlc'], dtype=torch.long).unsqueeze(0)

    return img, speed, hlc

def preprocess_data2(data, hlc_one_hot=True):
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

    hlc = torch.tensor(data['hlc'], dtype=torch.long)
    if hlc_one_hot:
        hlc = F.one_hot(hlc.to(torch.int64), num_classes=4)
    hlc = hlc.unsqueeze(0)

    return img, speed, hlc


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


def calculate_stats(distances, success_record):
    assert len(distances) == len(success_record), "Mismatch between lengths of distances and success_record arrays"
    
    num_episodes = len(success_record)
    
    # success rate
    success_rate = sum(success_record) / num_episodes
    
    # sccess rate weighted by track length
    weighted_success_rate = sum([dist * success for dist, success in zip(distances, success_record)]) / sum(distances)
    
    # Average distance traveled for failed episodes only
    fail_distances = [dist for dist, success in zip(distances, success_record) if success == 0]
    average_fail_distance = sum(fail_distances) / len(fail_distances) if fail_distances else 0
    
    return success_rate, weighted_success_rate, average_fail_distance