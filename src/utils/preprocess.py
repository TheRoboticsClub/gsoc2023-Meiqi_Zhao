import numpy as np
import carla

def traffic_light_to_int(light_status):
    light_dict = {
        -1: 0,
        carla.libcarla.TrafficLightState.Red: 1,
        carla.libcarla.TrafficLightState.Green: 2,
        carla.libcarla.TrafficLightState.Yellow: 3
    }
    return light_dict[light_status]


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
    #https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/CityScapesPalette.h
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

    array = to_bgra_array(image)[:, :, 2]
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result.astype(np.uint8)


def read_routes(filename):
    """Read routes/episodes from txt file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    routes = [((int(line.split()[0]), int(line.split()[1])), int(line.split()[2]), line.split()[3:]) for line in lines]
    return routes

