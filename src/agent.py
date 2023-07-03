import random
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent
import numpy as np
import carla

class NoisyBehaviorAgent(BehaviorAgent):
    """
    This agent adds noise to the output controls of the CARLA behavior agent for data collection.
    """

    def __init__(self, vehicle, behavior='normal'):
        super().__init__(vehicle, behavior=behavior)
        self.noise_count = 0
        self.noise_duration = random.randint(50, 150)
        self.noise = False
        self.multiplier = 1.0
    

    def run_step(self, debug=False):
        """
        Execute one step of navigation with noise added to the controls.

        :param debug: boolean for debugging
        :return control: carla.VehicleControl
        """

        true_control = super().run_step(debug)

        if self.noise:
            noisy_control = self._add_noise_to_control(true_control)
            self.noise_count += 1
        else:
            noisy_control = true_control
            self.noise_count += 1
        
        if self.noise_count > self.noise_duration:
            self.noise = not self.noise
            self.noise_duration = random.randint(50, 100) if self.noise else random.randint(50, 150)
            self.noise_count = 0
            self.multiplier = random.uniform(-0.3, 0.3)
            if self.noise: print('------------------------------------------')

        return true_control, noisy_control

    def _add_noise_to_control(self, control):
        """
        This method adds a temporally correlated noise to steering.

        :param control: carla.VehicleControl
        :return control: carla.VehicleControl
        """
        control_copy = carla.VehicleControl()

        # copy all properties
        control_copy.throttle = control.throttle
        control_copy.steer = control.steer
        control_copy.brake = control.brake
        control_copy.hand_brake = control.hand_brake
        control_copy.reverse = control.reverse
        control_copy.manual_gear_shift = control.manual_gear_shift
        control_copy.gear = control.gear

        control_copy.steer += self.multiplier * np.sin(self.noise_count * np.pi / self.noise_duration)

        # clip value to valid range
        control_copy.steer = max(-1., min(1., control_copy.steer))

        return control_copy


class NoisyTrafficManagerAgent:
    """
    This agent adds noise to the output controls of the CARLA Traffic Manager for data collection.
    """

    def __init__(self, vehicle, traffic_manager):
        self.vehicle = vehicle
        self.traffic_manager = traffic_manager
        self.noise_count = 0
        self.noise_duration = random.randint(50, 150)
        self.noise = False
        self.multiplier = 1.0
        self.path = None
        self.route = None
        self.destination = None

    def run_step(self, debug=False):
        """
        Execute one step of navigation with noise added to the controls.

        :param debug: boolean for debugging
        :return control: carla.VehicleControl
        """
        true_control =self.vehicle.get_control()
        noisy_control = None

        if self.noise:
            noisy_control = self._add_noise_to_control(true_control)
            self.noise_count += 1
        else:
            self.noise_count += 1
        
        if self.noise_count > self.noise_duration:
            self.noise = not self.noise
            self.noise_duration = random.randint(10, 20) if self.noise else random.randint(50, 150)
            self.noise_count = 0
            self.multiplier = random.uniform(0.005, 0.015) * np.random.choice([1.0, -1.0])
        
        return true_control, noisy_control

    def _add_noise_to_control(self, control):
        """
        This method adds a temporally correlated noise to steering.

        :param control: carla.VehicleControl
        :return control: carla.VehicleControl
        """
        control_copy = carla.VehicleControl()

        # copy all properties
        control_copy.throttle = control.throttle
        control_copy.steer = control.steer
        control_copy.brake = control.brake
        control_copy.hand_brake = control.hand_brake
        control_copy.reverse = control.reverse
        control_copy.manual_gear_shift = control.manual_gear_shift
        control_copy.gear = control.gear

        control_copy.steer += self.multiplier * np.sin(self.noise_count * np.pi / self.noise_duration)

        # clip value to valid range
        control_copy.steer = max(-1., min(1., control_copy.steer))

        return control_copy
    

    def set_path(self, path):
        self.path = path
        self.traffic_manager.set_path(self.vehicle, path)

    def set_route(self, route, destination):
        self.route = route
        self.traffic_manager.set_route(self.vehicle, route)
        self.destination = destination

    def get_next_action(self):
        return self.traffic_manager.get_next_action(self.vehicle)[0]

    def done(self):
        vehicle_location = self.vehicle.get_location()
        final_location = self.path[-1] if self.path else self.destination.location if self.route else None

        if final_location is None:
            return False  # The vehicle has no path or route

        distance = vehicle_location.distance(final_location)
        return distance < 1.0
