"""
Given
v - linear velocity
w - angular velocity
v, w
vr, vl
wr, wl
Move the robot base on duration


"""
import numpy as np

import nav_msgs.msg
import rclpy
from geometry_msgs.msg import Vector3, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


# noinspection PyClassHasNoInit
class ControlInputs:
    Linear = 0
    Angular = 1
    LinearAngular = 2
    LinearWheels = 3
    AngularWheels = 4


class Controller(Node):
    def __init__(self, node_name: str, axle_length=1.25, transmitter_length=1.25, radius=0.3, sample_rate=0.033):
        """"""
        self.axle_length = axle_length
        self.transmitter_length = transmitter_length
        self.wheel_radius = radius
        self.transform_matrix = np.array([[self.wheel_radius * 0.5, self.wheel_radius * 0.5],
                                          [-self.wheel_radius / self.transmitter_length,
                                           self.wheel_radius / self.transmitter_length]])
        self.robot_pose = None
        self.initial_pose = None
        self.data = None
        self.odom_sub = None
        self.duration = 0
        self.current_time = 0
        self.control_timer = None
        self.save_odom_data = False
        self.sample_rate = sample_rate
        self.v = 0.0
        self.w = 0.0
        self.estimated_pose = Vector3()
        self.pose_estimates = []
        self.odometry_list = []
        self.busy = False
        self.twist_publisher = None
        self.pose_publisher = None
        self.ref_pose = [0] * 3
        self.kw = 0
        self.kv = 0
        self.d_tol = 0
        self.epsilon = 0.001
        self.control_to_ref = False
        self.file_prefix = ""
        super().__init__(node_name)
        self.logger = self.get_logger()
        self.create_subscribers()
        self.create_publishers()

    def create_subscribers(self):
        """
        Subscribe to odometry for now

        :return:
        """
        self.odom_sub = self.create_subscription(Odometry, "/hagen/odom", self.set_robot_pose, 20)

    def create_publishers(self):
        """"""
        self.twist_publisher = self.create_publisher(Twist, '/hagen/cmd_vel', 5)
        self.pose_publisher = self.create_publisher(Vector3, '/hagen2/pose', 5)

    def set_robot_pose(self, msg: nav_msgs.msg.Odometry):
        """
        Get and set the published robot odometry data.

        :return:
        """
        _, _, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        if self.initial_pose is None:
            self.initial_pose = self.robot_pose

    @staticmethod
    def euler_from_quaternion(quaternion):
        """
        Converts quaternion ([x, y, z, w]) to euler roll, pitch, yaw
        Should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sin_r_cos_p = 2 * (w * x + y * z)
        cos_r_cos_p = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sin_r_cos_p, cos_r_cos_p)

        sin_p = 2 * (w * y - z * x)
        pitch = np.arcsin(sin_p)

        sin_y_cos_p = 2 * (w * z + x * y)
        cos_y_cos_p = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(sin_y_cos_p, cos_y_cos_p)

        return roll, pitch, yaw

    def send_velocity(self, v=0.0, w=0.0):
        """
        Published linear and angular velocity

        :param v: linear velocity
        :param w: angular velocity
        :return:
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.twist_publisher.publish(msg)

    @staticmethod
    def wrap_to_pi(theta: float):
        """"""
        x, max_ = theta + np.pi, 2 * np.pi
        return -np.pi + ((max_ + (x % max_)) % max_)

    @staticmethod
    def wrap_to_pi_2(theta: float):
        """"""
        return DifferentialDrive.wrap_min_max(theta, -np.pi, np.pi)

    @staticmethod
    def wrap_max(x, max_):
        return (max_ + (x % max_)) % max_

    @staticmethod
    def wrap_min_max(x, min_, max_):
        return min_ + DifferentialDrive.wrap_max(x - min_, max_ - min_)

    def control_commands(self, data: list):
        """
        Given a data list of control input which is a list of tuples.
        [([Input], ControlType, duration)]
        Input can be a 1 or more values. 1 if the second value can be inferred.
        Input takes the one of the forms below
        [v]
        [w]
        [v, w]
        [vr, vl]
        [wr, wl]

        :param data:
        :return:
        """
        if not self.busy:
            self.data = data
            x, y, z = self.initial_pose
            self.estimated_pose = Vector3(x=x, y=y, z=z)
            self.start_next_command()
            self.busy = True
            return True
        return False

    def estimate_pose(self):
        """
        Store estimated (calculated pose) and reported odometry.

        :return:
        """
        alpha = self.estimated_pose.z + (0.5 * self.w * self.sample_rate)
        self.estimated_pose.x = self.estimated_pose.x + (self.v * self.sample_rate * np.cos(alpha))
        self.estimated_pose.y = self.estimated_pose.y + (self.v * self.sample_rate * np.sin(alpha))
        # self.estimated_pose.z = self.wrap_to_pi(self.estimated_pose.z + (self.w * self.sample_rate))
        self.estimated_pose.z = self.estimated_pose.z + (self.w * self.sample_rate)
        self.pose_estimates.append(np.array([self.estimated_pose.x, self.estimated_pose.y, self.estimated_pose.z]))
        self.odometry_list.append(self.robot_pose)
        # self.pose_publisher.publish(self.estimated_pose)
        x, y, z = self.robot_pose
        msg = Vector3(x=x, y=y, z=z)
        self.pose_publisher.publish(msg)

    def save_odom_to_file(self):
        """
        Save estimated pose and odometry data to file for plotting.

        :return:
        """
        if self.save_odom_data:
            np_array = np.array(self.odometry_list)
            self.get_logger().info(f"Odom Data Size: {np_array.shape}.")
            np.save(f"{self.file_prefix}odom", np_array)
            np_array = np.array(self.pose_estimates)
            self.get_logger().info(f"Pose Estimate Data Size: {np_array.shape}.")
            np.save(f"{self.file_prefix}pose_estimates", np_array)

    def start_next_command(self):
        """
        Start next command from list of control commands.

        :return:
        """
        self.logger.info("New Command.")
        self.v, self.w, duration = self.get_velocities_and_duration()
        self.duration = int((1 / self.sample_rate) * duration)
        self.send_velocity(self.v, self.w)
        self.control_timer = self.create_timer(self.sample_rate, self.control_callback)

    def get_velocities_and_duration(self):
        """Override in subclass"""
        return 0.0, 0.0, 0

    def control_callback(self):
        """Override in subclass"""
        pass

    def external_kinematics(self, theta):
        """

        :param theta:
        :return: vx, vy, w
        """
        transform = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])
        return transform @ np.array([self.v, self.w])


class DifferentialDrive(Controller):
    def __init__(self, node_name: str = "differential_drive"):
        """"""
        self.references = []
        self.point_index = 0
        self.kr = 0
        super().__init__(node_name)

    def split_linear_inputs(self, vr, vl):
        """
        Given linear velocities of wheel, calculates and returns the robot linear and angular velocities.

        :param vr: right wheel linear velocity
        :param vl: left wheel linear velocity
        :return:
        """
        v = (vl + vr) / 2
        icr_radius = -(0.5 * self.axle_length) * ((vl + vr) / (vl - vr))
        w = v / icr_radius
        return v, w

    def split_angular_inputs(self, wr, wl):
        """
        Given angular velocities of wheel, calculates and returns the robot linear and angular velocities.

        :param wr: right wheel angular velocity
        :param wl: left wheel angular velocity
        :return:
        """
        return self.transform_matrix @ np.array([wl, wr])

    def get_velocities_and_duration(self):
        """
        Return linear and angular velocities based on control input type and duration for this command.

        :return:
        """
        input_, control_type, duration = self.data.pop(0)
        if control_type == ControlInputs.Linear:
            v, w = input_[0], 0.0
        elif control_type == ControlInputs.Angular:
            v, w = 0.0, input_[0]
        elif control_type == ControlInputs.LinearAngular:
            v, w = input_[0], input_[1]
        elif control_type == ControlInputs.LinearWheels:
            v, w = self.split_linear_inputs(*input_)
        elif control_type == ControlInputs.AngularWheels:
            v, w = self.split_angular_inputs(*input_)
        else:
            v, w, duration = 0.0, 0.0, 0
        return v, w, duration

    def control_callback(self):
        if self.current_time == self.duration:
            self.send_velocity()
            self.current_time = 0
            self.logger.info("Completed Command. Resetting.")
            if len(self.data) == 0:
                self.control_timer.cancel()
                self.busy = False
                if self.save_odom_data:
                    self.save_odom_to_file()
            else:
                self.start_next_command()
        else:
            self.current_time += 1
            self.estimate_pose()
            self.send_velocity(self.v, self.w)

    def orientation_control(self, theta_ref, vr=0.8, k=0.3):
        """

        :param theta_ref: reference orientation in radians
        :param vr: tangential velocity
        :param k: proportional term for angular velocity.
        :return:
        """
        if not self.busy:
            self.ref_pose[2] = theta_ref
            self.kw = k
            self.file_prefix = "orientation_control_"
            self.control_timer = self.create_timer(self.sample_rate, lambda: self.orientation_callback(vr))
            self.busy = True
            return True
        return False

    def orientation_callback(self, vr):
        _, _, theta = self.robot_pose
        error = self.wrap_to_pi(self.ref_pose[2] - theta)
        self.estimate_pose()
        if error <= self.epsilon:
            self.send_velocity()
            self.control_timer.cancel()
            self.busy = False
            self.save_odom_to_file()
        else:
            alpha = self.kw * error
            # differential
            # w = alpha
            # ackermann
            self.w = (vr * np.tan(alpha)) / self.transmitter_length
            self.send_velocity(w=self.w)

    def position_control(self, ref_pos: list, kv=0.25):
        """
        Control robot to a position given by ref_pos. This usually would not work.
        Forward motion can not drive robot to reference pose without correct orientation.
        So we assert that either x or y is 0.

        :param ref_pos: [x, y]
        :param kv: proportional term
        :return:
        """
        assert len(ref_pos) == 2
        assert ref_pos[0] == 0 or ref_pos[-1] == 0
        if not self.busy:
            self.ref_pose = ref_pos
            self.kv = kv
            self.epsilon = 0.05
            self.file_prefix = "position_control_"
            self.control_timer = self.create_timer(self.sample_rate, self.position_callback)
            self.busy = True
            return True
        return False

    def position_callback(self):
        x, y, theta = self.robot_pose
        xr, yr = self.ref_pose
        d = np.sqrt(((xr - x) ** 2) + ((yr - y) ** 2))
        self.estimate_pose()
        if d <= self.epsilon:
            self.send_velocity()
            self.control_timer.cancel()
            self.busy = False
            self.save_odom_to_file()
        else:
            self.v = self.kv * d
            self.send_velocity(v=self.v)

    def reference_pose_control(self, ref_pose: list, kv=0.3, kw=0.3):
        """
        Given a ref pose, control robot to target.
        TODO: Maybe reference orientation could be known and passed to the ref_pose argument?

        :param ref_pose: [x, y]
        :param kv: proportional term for linear velocity
        :param kw: proportional term for angular velocity
        :return:
        """
        assert len(ref_pose) == 2
        if not self.busy:
            self.ref_pose = ref_pose
            self.kv = kv
            self.kw = kw
            self.epsilon = 0.06
            self.file_prefix = "reference_pose_"
            self.control_timer = self.create_timer(self.sample_rate, self.reference_pose_callback)
            self.busy = True
            return True
        return False

    def reference_pose_callback(self):
        x, y, theta = self.robot_pose
        xr, yr = self.ref_pose
        d = np.sqrt(((xr - x) ** 2) + ((yr - y) ** 2))
        self.estimate_pose()
        if d <= self.epsilon:
            self.control_timer.cancel()
            self.send_velocity()
            self.busy = False
            self.save_odom_to_file()
        else:
            theta_ref = np.arctan((yr - y) / (xr - x))
            e = self.wrap_to_pi(theta_ref - theta)
            # w = self.kw * e
            # v = self.kv * d
            # to handle when orientation error sharply changes
            self.w = self.kw * np.arctan(np.tan(e))
            self.v = self.kv * d * np.sign(np.cos(e))
            self.send_velocity(self.v, self.w)

    def ref_pose_intermediate_point(self, ref_pose: list, kv=0.35, kw=0.45, r=2.0, d_tol=0.8):
        """ref_pose = [x, y, theta]"""
        if len(ref_pose) == 2:
            ref_pose.append(np.arctan2(ref_pose[1], ref_pose[0]))
        assert len(ref_pose) == 3
        if not self.busy:
            self.ref_pose = ref_pose
            self.kv = kv
            self.kw = kw
            self.d_tol = d_tol
            self.file_prefix = "ref_pose_inter_point_"
            self.epsilon = 0.15
            # intermediate point
            xr, yr, theta_ref = self.ref_pose
            x_i = xr - (r * np.cos(theta_ref))
            y_i = yr - (r * np.sin(theta_ref))
            self.control_timer = self.create_timer(self.sample_rate, lambda: self.intermediate_point_callback(x_i, y_i))
            self.busy = True
            return True
        return False

    def intermediate_point_callback(self, x_i: float, y_i: float):
        """
        Control to intermediate point or to reference point.
        Try to first start control to intermediate point before switching to reference point.
        If control starts to reference point it doesn't switch back to intermediate point control.

        :param y_i:
        :param x_i:
        :return:
        """
        # reference pose
        xr, yr, theta_ref = self.ref_pose
        # current pose
        x, y, theta = self.robot_pose
        # x, y, theta = self.pose_estimates[-1]
        # distance to reference
        d = np.sqrt(((xr - x) ** 2) + ((yr - y) ** 2))
        # distance to intermediate point
        d_i = np.sqrt(((x_i - x) ** 2) + ((y_i - y) ** 2))
        if d <= self.epsilon:
            self.send_velocity()
            self.control_timer.cancel()
            self.busy = False
            self.save_odom_to_file()
        else:
            self.estimate_pose()
            if self.control_to_ref:
                # control to ref pose
                e = theta_ref - theta
                e = self.wrap_to_pi(e)
                w = self.kw * np.arctan(np.tan(e))
                # w = self.kw * e
                v = self.kv * d * np.sign(np.cos(e))
                print(d)
            else:
                # control to intermediate point
                self.control_to_ref = d_i < self.d_tol
                theta_ref = np.arctan2((y_i - y), (x_i - x))
                e = self.wrap_to_pi(theta_ref - theta)
                w = self.kw * np.arctan(np.tan(e))
                v = self.kv * d_i * np.sign(np.cos(e))
                self.logger.info(f"Approaching Intermediate point: {d_i}")
            self.v = v
            self.w = w
            self.send_velocity(v, w)
        # print(d)

    def ref_pose_intermediate_direction(self, ref_pose: list, kv=0.35, kw=0.47, r=1.3, d_tol=0.3):
        """ref_pose = [x, y, theta]"""
        assert len(ref_pose) == 3
        if not self.busy:
            self.ref_pose = ref_pose
            self.kv = kv
            self.kw = kw
            self.epsilon = d_tol
            self.file_prefix = "ref_pose_inter_dir_"
            self.control_timer = self.create_timer(self.sample_rate,
                                                   lambda: self.intermediate_direction_callback(r))
            self.busy = True
            return True
        return False

    def intermediate_direction_callback(self, r: float):
        """
        Control to reference point via intermediate direction.

        :param r: distance from intermediate point to reference point
        :return:
        """
        # reference pose
        xr, yr, theta_ref = self.ref_pose
        # current pose
        x, y, theta = self.robot_pose
        # distance to reference
        d = np.sqrt(((xr - x) ** 2) + ((yr - y) ** 2))
        if d < self.epsilon:
            # stop
            self.send_velocity()
            self.control_timer.cancel()
            self.busy = False
            self.save_odom_to_file()
            # self.logger.info("Readjusting orientation")
            # self.orientation_control(theta_ref, 1, self.kw)
        else:
            # control
            # intermediate direction
            self.estimate_pose()
            theta_r = np.arctan2(yr - y, xr - x)
            alpha = self.wrap_to_pi(theta_r - theta_ref)
            if alpha > 0:
                beta = np.arctan(r/d)
            else:
                beta = -np.arctan(r/d)
            e = theta_r - theta + (alpha if np.abs(alpha) < np.abs(beta) else beta)
            e = self.wrap_to_pi(e)
            w = self.kw * e
            v = self.kv * d
            self.logger.info(f"d: {d:.4f}")
            self.estimate_pose()
            self.send_velocity(v, w)

    def ref_path_control(self, references: list, kr=0.95, kv=0.45, kw=0.55):
        """
        Given references,
        for any line segment
            v = T_i_1 - T_i
            the line equation is given as
            L = ti + t(ti - ti1)
            v1 = ti (t = 0)
            v2 = -ti2 (t = 1)
            vn = v1 x v2 (cross-product)
        :param kr:
        :param kv:
        :param kw:
        :param references:
        :return:
        """
        if not self.busy:
            self.references = [np.array(a) for a in references]
            self.point_index = 0
            self.kr = kr
            self.kv = kv
            self.kw = kw
            self.file_prefix = "ref_path_control_"
            self.epsilon = 0.35
            self.control_timer = self.create_timer(self.sample_rate, self.control_ref_path_callback)
            self.busy = True
            return True
        return False

    def control_ref_path_callback(self):
        """"""
        x, y, theta = self.robot_pose
        xr, yr = self.references[-1]
        d_ = np.sqrt(((xr - x) ** 2) + ((yr - y) ** 2))
        self.logger.info(f"Distance to ref point: d {d_:.4f}")
        if d_ <= self.epsilon:
            self.send_velocity()
            self.control_timer.cancel()
            self.busy = False
        else:
            references = self.references
            ti = references[self.point_index]
            ti1 = references[self.point_index+1]
            v = ti1 - ti
            q = np.array([x, y])
            r = (q - ti)
            u = (v@r)/(v@v)
            if u > 1 and self.point_index + 2 < len(self.references):
                # follow next line
                self.point_index += 1
                ti = references[self.point_index]
                ti1 = references[self.point_index+1]
                v = ti1 - ti
                r = (q - ti)
            # follow current line v
            vn = np.array([v[1], -v[0]])
            vn_transpose = vn
            d = (vn_transpose@r)/(vn_transpose@vn)
            theta_line = np.arctan2(v[1], v[0])
            theta_rot = np.arctan(self.kr*d)
            theta_ref = theta_line + theta_rot
            _, _, theta = self.robot_pose
            e = theta_ref - theta
            # e = self.wrap_to_pi(e)
            self.w = self.kw * e
            self.v = self.kv * np.cos(e)
            self.logger.info(f"point: {self.point_index}; [{ti} - {ti1}]")
            self.send_velocity(v=self.v, w=self.w)
            self.estimate_pose()

    def ppack_a(self, a: list, b: list, kv=0.45, kw=0.41):
        """"""
        if not self.busy:
            self.kv = kv
            self.kw = kw
            self.file_prefix = "ref_ppark_a_control_"
            self.epsilon = 0.4
            self.d_tol = 0.1
            centre = (np.array(b) - np.array(a)) / 2
            self.control_timer = self.create_timer(self.sample_rate, lambda: self.ppack_a_callback(centre, b))
            self.busy = True
            return True
        return False

    def ppack_a_callback(self, centre, ref):
        """"""
        x, y, theta = self.robot_pose
        xr, yr = ref
        d = np.sqrt(((xr - x) ** 2) + ((yr - y) ** 2))
        if d <= self.epsilon:
            # stop
            self.control_timer.cancel()
            self.send_velocity()
            self.busy = False
            self.save_odom_to_file()
        else:
            self.estimate_pose()
            if self.control_to_ref:
                # control to ref pose (b)
                e = -theta
                e = self.wrap_to_pi(e)
                w = self.kw * e
                v = self.kv * d
                self.logger.info(f"Moving to target: {d:.4f}, e: {e:.4f}")
            else:
                # control to intermediate point
                xi, yi = centre
                d_i = np.sqrt(((xi - x) ** 2) + ((yi - y) ** 2))
                self.control_to_ref = d_i <= self.d_tol
                theta_ref = np.pi/2
                e = self.wrap_to_pi(theta_ref - theta)
                w = self.kw * e  # np.arctan(np.tan(e))
                v = self.kv * d_i  # * np.sign(np.cos(e))
                self.logger.info(f"Approaching Intermediate point: {d_i:.4f}")
            self.v = v
            self.w = w
            self.send_velocity(v, w)

    def ppack_b(self):
        """"""


def main():
    rclpy.init()
    node = DifferentialDrive()

    while rclpy.ok():
        try:
            if node.initial_pose is not None:
                node.get_logger().info(f"Initial Robot Pose: {node.initial_pose}")
                break
        except Exception as e:
            print(f"Something went wrong in the ROS Loop: {e}")
        rclpy.spin_once(node)

    # orientation control
    # node.save_odom_data = True
    # node.orientation_control(np.pi/2)

    # position control
    # node.save_odom_data = True
    # node.position_control([10, 0])

    # reference control
    # node.save_odom_data = True
    # node.reference_pose_control([10, 5])

    # reference via point
    # node.save_odom_data = True
    # node.ref_pose_intermediate_point([3, 6], r=1.3)

    # reference via direction
    # node.save_odom_data = True
    # node.ref_pose_intermediate_direction([3, 6, 0])

    # reference via paths
    # node.save_odom_data = True
    import time
    count = 0
    max_ = 5
    while count <= max_:
        node.pose_publisher.publish(node.estimated_pose)
        time.sleep(1)
        count += 1
    if node.pose_publisher.get_subscription_count():
        node.ref_path_control([(3, 0), (6, 4), (3, 4), (3, 1), (0, 3)])
    # node.ref_path_control([(3, 0), (6, 4), (3, 4), (3, 1), (0, 3)])

    # ppack
    # node.save_odom_data = True
    # node.ppack_a([0, 0], [4, 4])

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
