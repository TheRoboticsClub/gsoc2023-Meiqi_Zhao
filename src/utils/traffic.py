import carla
import logging
import random

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def get_pedestrain_spawn_points(world, n):
    spawn_points = []
    for i in range(n):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    return spawn_points


def get_vehicle_spawn_points(world, n_vehicles):
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    if n_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif n_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, n_vehicles, number_of_spawn_points)
        n_vehicles = number_of_spawn_points
    return spawn_points

def spawn_vehicles(world, client, n_vehicles, traffic_manager):
    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    # blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car'] # cars only
    blueprints = sorted(blueprints, key=lambda bp: bp.id)
    spawn_points = get_vehicle_spawn_points(world, n_vehicles)

    vehicles_list = []
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= n_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    return vehicles_list

def spawn_pedestrians(world, client, n_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=1.0):
    walkers_list = []
    all_id = []

    # 1. get spawn points and blueprints
    spawn_points = get_pedestrain_spawn_points(world, n_pedestrians)
    blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')

    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        #if walker_bp.has_attribute('is_invincible'):
        walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
    
    return all_id, all_actors, walkers_list

def get_traffic_light_status(vehicle):
    light_status = -1
    if vehicle.is_at_traffic_light():
        traffic_light = vehicle.get_traffic_light()
        light_status = traffic_light.get_state()
    return light_status

def cleanup(vehicles_list, pedestrians_list, all_id, all_actors, vehicle, sensors, client):
    # destroy ego vehicle
    vehicle.destroy()

    # destroy sensors
    for sensor in sensors: sensor.destroy()
    
    # destroy vehicles
    logging.info('destroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    # destroy walkers
    logging.info('destroying %d walkers' % len(pedestrians_list))
    for i in range(0, len(all_id), 2):
        all_actors[i].stop()
    client.apply_batch([carla.command.DestroyActor(x) for x in all_id])