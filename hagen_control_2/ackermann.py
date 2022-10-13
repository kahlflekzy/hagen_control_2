#! /usr/bin/python3
import numpy as np

from build.hagen_control_2.hagen_control_2.controllers import Controller


class ControlInputs:
    Linear = 0
    Angular = 1
    LinearAngular = 2
    LinearWheels = 3
    AngularWheels = 4


class BicycleModelNode(Controller):
    """
    Bicycle Models are steered from the front and driven from the rear. They could also be both powered and driven from
    the front.
    """

    def __init__(self, axle_length=1.25, transmitter_length=1.25, radius=0.3, sample_rate=0.033):
        super().__init__("ackermann_model_node", axle_length=axle_length, transmitter_length=transmitter_length,
                         radius=radius, sample_rate=sample_rate)

    def get_velocities_and_duration(self):
        """
       TODO: Continue, get v, w based on the control_inputs.

        :return:
        """
        input_, control_type, duration = self.data.pop(0)
        if control_type == ControlInputs.Angular:
            v, w, _ = self.steering_angular_input(*input_)
        elif control_type == ControlInputs.LinearAngular:
            v, w, _ = self.steering_angular_input(*input_)
        else:
            v, w, duration = 0.0, 0.0, 0
        return v, w, duration

    def input_angular_velocity_and_icr_radius(self, w, radius):
        """"""
        alpha_r = np.arctan(self.transmitter_length/(radius - (self.axle_length/2)))
        alpha_l = np.arctan(self.transmitter_length/(radius + (self.axle_length/2)))