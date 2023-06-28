import os
import carla
from carla import ColorConverter as cc

# store data
def save_data(data, sensor_name):
    path = f'./data/{sensor_name}'
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = str(f'{data.frame:05d}.png')
    if sensor_name == 'segmentation':
        data.save_to_disk(f'{path}/{timestamp}', cc.CityScapesPalette)
    else:
        data.save_to_disk(f'{path}/{timestamp}')


class RGBCamera:
    def __init__(self, world, vehicle):
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_location = carla.Location(x=1.5, z=2.4)
        cam_rotation = carla.Rotation(pitch=-15)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        cam_bp.set_attribute('image_size_x', '288')
        cam_bp.set_attribute('image_size_y', '200')
        cam_bp.set_attribute('fov', '90')

        self._sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        self._data = None
        self._sensor.listen(lambda data: self._callback(data))

    def _callback(self, data):
        self._data = data
    
    def get_sensor_data(self):
        return self._data

    def get_sensor(self):
        return self._sensor

class SegmentationCamera:
    def __init__(self, world, vehicle):
        cam_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        cam_location = carla.Location(x=1.5, z=2.4)
        cam_rotation = carla.Rotation(pitch=-15)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        cam_bp.set_attribute('image_size_x', '288')
        cam_bp.set_attribute('image_size_y', '200')
        cam_bp.set_attribute('fov', '90')

        self._sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        self._data = None
        
        self._sensor.listen(lambda data: self._callback(data))

    def _callback(self, data):
        self._data = data
    
    def get_sensor_data(self):
        return self._data

    def get_sensor(self):
        return self._sensor


def setup_collision_sensor(world, vehicle):
    bp = world.get_blueprint_library().find('sensor.other.collision')
    transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    return sensor