#! /usr/bin/python3
import numpy as np

from build.hagen_control_2.hagen_control_2.controllers import Controller
from geometry_msgs.msg import Vector3
from rclpy import Node


class ControlInputs:
    FrontWheelAngular = 0
    FrontWheelLinear = 1
    LinearAngular = 2
    RearWheelAngular = 3
    RearWheelLinear = 4


class BicycleModelNode(Controller):
    """
    Bicycle Models are steered from the front and driven from the rear. They could also be both powered and driven from
    the front.
    """

    def __init__(self, transmitter_length=1.25, radius=0.3, sample_rate=0.033):
        super().__init__("bicycle_model_node", axle_length=0, transmitter_length=transmitter_length, radius=radius,
                         sample_rate=sample_rate)

    def steering_angular_input(self, alpha, ws):
        """
        Given alpha, ws, return the robot linear, angular velocity and ICR radius

        :param alpha:
        :param ws:
        :return:
        """
        icr_radius = self.transmitter_length / np.tan(alpha)
        vs = ws * self.wheel_radius
        w = vs * np.sin(alpha) / self.transmitter_length
        v = vs * np.cos(alpha)
        return v, w, icr_radius

    def steering_linear_input(self, alpha, vs):
        """
        Giving the tangential velocity of the steering wheel,
        calculate and return the robot linear and angular velocity.

        :param alpha:
        :param vs:
        :return:
        """
        w = vs * np.sin(alpha) / self.transmitter_length
        v = vs * np.cos(alpha)
        return v, w

    def rear_linear_input(self, alpha, vs):
        """
        Given alpha and the tangential velocity of steering wheel,
        calculate the robot linear, angular velocity.

        :param alpha:
        :param vs:
        :return:
        """
        v = vs * np.cos(alpha)
        w = vs * np.tan(alpha) / self.transmitter_length
        return v, w

    def rear_angular_input(self, alpha, ws):
        """
        Given alpha and the angular velocity of steering wheel,
        calculate the robot linear, angular velocity.

        :param alpha:
        :param ws:
        :return:
        """
        vs = ws * self.wheel_radius
        v = vs * np.cos(alpha)
        w = vs * np.tan(alpha) / self.transmitter_length
        return v, w

    def get_velocities_and_duration(self):
        """
        Return linear (v) and angular (w) velocity and duration.
        Inputs are always [(alpha, velocity)] velocity can be ws or vs.
        The calculation depends on the control type.

        :return:
        """
        input_, control_type, duration = self.data.pop(0)
        if control_type == ControlInputs.FrontWheelAngular:
            v, w, _ = self.steering_angular_input(*input_)
        elif control_type == ControlInputs.FrontWheelLinear:
            v, w = self.steering_linear_input(*input_)
        elif control_type == ControlInputs.LinearAngular:
            v, w = input_[0], input_[1]
        elif control_type == ControlInputs.RearWheelAngular:
            v, w = self.rear_linear_input(*input_)
        elif control_type == ControlInputs.RearWheelLinear:
            v, w = self.rear_angular_input(*input_)
        else:
            v, w, duration = 0.0, 0.0, 0
        return v, w, duration

