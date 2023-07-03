import numpy as np
import carla


def road_option_to_int(high_level_command):
    """convert CARLA.RoadOptions to integer"""
    road_option_dict = {
        "LaneFollow": 0,
        "Left": 1,
        "Right": 2,
        "Straight": 3
    }
    return road_option_dict[high_level_command]

def int_to_road_option(high_level_command):
    """convert integer high-level command to Carla.RoadOptions"""
    road_option_dict = {
        0: "LaneFollow",
        1: "Left",
        2: "Right",
        3: "Straight"
    }
    return road_option_dict[high_level_command]

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def carla_rgb_to_array(image):
    """Convert a CARLA raw image to an RGB numpy array"""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def carla_seg_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    #https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
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

    array = to_bgra_array(image)[:, :, 2]
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result.astype(np.uint8)


def read_routes(filename):
    """Read routes/episodes from txt file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    routes = [((int(line.split()[0]), int(line.split()[1])), line.split()[2:]) for line in lines]
    return routes

