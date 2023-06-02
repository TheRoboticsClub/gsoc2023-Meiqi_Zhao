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


# set up rgb camera
def setup_rgb_camera(world, vehicle):
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_location = carla.Location(x=1.5, z=2.4)
    cam_rotation = carla.Rotation(pitch=-15)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    cam_bp.set_attribute('image_size_x', '288')
    cam_bp.set_attribute('image_size_y', '200')
    cam_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    camera.listen(lambda data: save_data(data, 'rgb'))
    return camera


# set up semantic segmentation camera
def setup_semseg_camera(world, vehicle):
    cam_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    cam_location = carla.Location(x=1.5, z=2.4)
    cam_rotation = carla.Rotation(pitch=-15)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    cam_bp.set_attribute('image_size_x', '288')
    cam_bp.set_attribute('image_size_y', '200')
    cam_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    camera.listen(lambda data: save_data(data, 'segmentation'))
    return camera
