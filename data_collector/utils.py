import numpy as np
import carla

"""
Pickling of "carla.libcarla.VehicleControl" instances is not enabled,
need to convert from VehicleControl to dict
"""
def vehicle_control_to_dict(vehicle_control):
    return {
        'throttle': vehicle_control.throttle,
        'steer': vehicle_control.steer,
        'brake': vehicle_control.brake,
    }


def carla_seg_to_array(image):
    img_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    img_array = np.reshape(img_array, (image.height, image.width, 4))
    img_array = img_array[:, :, 2]  # get the class id channel
    return img_array

def carla_rgb_to_array(image):
    img_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    img_array = np.reshape(img_array, (image.height, image.width, 4))
    img_array = img_array[:, :, :3]
    img_array = img_array[:, :, ::-1]
    
    return img_array
