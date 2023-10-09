import torch
import numpy as np
import carla
import torch.nn.functional as F

def model_control(model, frame_data, device='cpu', filter=True, one_hot=True, ignore_traffic_light=True, combined_control=False):
    global counter 
    img, speed, hlc, light = preprocess_data(frame_data, filter=filter, one_hot=one_hot, ignore_traffic_light=ignore_traffic_light)
    img = img.to(device)
    speed = speed.to(device)
    hlc = hlc.to(device)
    light = light.to(device)
    if ignore_traffic_light:
        prediction = model(img, speed, hlc)
    else:
        prediction = model(img, speed, hlc, light)
    prediction = prediction.detach().cpu().numpy().flatten()
    #print(f"prediction: {prediction}")

    if not combined_control:
        throttle, steer, brake = prediction
        throttle = float(throttle)
        brake = float(brake)
        if brake < 0.05: brake = 0.0
    else:
        combined, steer = prediction
        combined = float(combined)
        throttle, brake = 0.0, 0.0
        if combined >= 0.5:
            throttle = (combined - 0.5) / 0.5
        else:
            brake = (0.5 - combined) / 0.5
    
    steer = (float(steer) * 2.0) - 1.0

    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

# def preprocess_data(data):
#     rgb = torch.tensor(data['rgb'].copy(), dtype=torch.float32).permute(2, 0, 1)
#     rgb /= 255.0
    
#     segmentation = torch.tensor(data['segmentation'].copy(), dtype=torch.float32).permute(2, 0, 1)
#     segmentation /= 255.0

#     img = torch.cat((rgb, segmentation), dim=0)
#     img = img.unsqueeze(0)
    
#     speed = torch.tensor(data['measurements'].copy(), dtype=torch.float32)
#     speed /= 90.0
#     speed = speed.unsqueeze(0)

#     hlc = torch.tensor(data['hlc'], dtype=torch.long).unsqueeze(0)

#     return img, speed, hlc

def preprocess_data(data, filter=True, one_hot=True, ignore_traffic_light=True):
    rgb = data['rgb'].copy()
    segmentation = data['segmentation'].copy()

    if filter:
        rgb, segmentation = filter_classes(rgb, segmentation)

    rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
    rgb /= 255.0
    
    segmentation = torch.tensor(segmentation, dtype=torch.float32).permute(2, 0, 1)
    segmentation /= 255.0

    img = torch.cat((rgb, segmentation), dim=0)
    img = img.unsqueeze(0)
    
    speed = torch.tensor(data['measurements'].copy(), dtype=torch.float32)
    speed = torch.clamp(speed / 40, 0, 1)
    speed = speed.unsqueeze(0)

    hlc = torch.tensor(data['hlc'], dtype=torch.long)
    if one_hot:
        hlc = F.one_hot(hlc.to(torch.int64), num_classes=4)
    hlc = hlc.unsqueeze(0)

    if not ignore_traffic_light:
        light = torch.tensor(data['light'], dtype=torch.long)
        if one_hot:
            light = F.one_hot(light.to(torch.int64), num_classes=4)
        light = light.unsqueeze(0)
    else:
        light = None

    return img, speed, hlc, light


def filter_classes(rgb, seg, classes_to_keep = [1, 7, 12, 13, 14, 15, 16, 17, 18, 19, 24]):
    classes = {
            0: [0, 0, 0],         # Unlabeled  
            1: [128,  64, 128],   # Road ***
            2: [244,  35, 232],   # Sidewalk
            3: [70,  70,  70],    # Building
            4: [102, 102, 156],   # Wall
            5: [190, 153, 153],   # Fence
            6: [153, 153, 153],   # Pole
            7: [250, 170,  30],   # Traffic Light ***
            8: [220, 220,   0],   # Traffic Sign
            9: [107, 142,  35],   # Vegetation
            10: [152, 251, 152],  # Terrain
            11: [70, 130, 180],   # Sky
            12: [220,  20,  60],  # Pedestrain ***
            13: [255,   0,   0],  # Rider ***
            14: [0,   0, 142],    # Car ***
            15: [0,   0,  70],    # Truck ***
            16: [0,  60, 100],    # Bus ***
            17: [0,  80, 100],    # Train ***
            18: [0,   0, 230],    # Motorcycle ***
            19: [119,  11,  32],  # Bicycle ***
            20: [110, 190, 160],  # Static
            21: [170, 120,  50],  # Dynamic
            22: [55,  90,  80],   # Other
            23: [45,  60, 150],   # Water
            24: [157, 234,  50],  # Road Line ***
            25: [81,   0,  81],   # Ground
            26: [150, 100, 100],  # Bridge
            27: [230, 150, 140],  # Rail Track
            28: [180, 165, 180]   # Guard Rail
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