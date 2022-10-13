import numpy as np

import nav_msgs.msg
import rclpy
from geometry_msgs.msg import Vector3, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class Controller(Node):
    """"""
    def __init__(self, node_name: str = "controller", sample_rate=0.033):
        """"""
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
        self.ref_pose = [0] * 3
        self.kw = 0
        self.kv = 0
        self.epsilon = 0.001
        self.file_prefix = ""
        self.references = []
        self.point_index = 0
        self.kr = 0
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
        x, max_ = theta + np.pi, 2*np.pi
        return -np.pi + ((max_ + (x % max_)) % max_)

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
        self.estimated_pose.z = self.wrap_to_pi(self.estimated_pose.z + (self.w * self.sample_rate))
        # self.estimated_pose_publisher.publish(self.estimated_pose)
        self.pose_estimates.append(np.array([self.estimated_pose.x, self.estimated_pose.y, self.estimated_pose.z]))
        self.odometry_list.append(self.robot_pose)

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

    def ref_path_control(self, references: list, kr=0.8, kv=0.4, kw=0.3):
        """
        Alias reference_path_follower_diff_drive.

        :param kr: A constant that determines the degree to which robot tries to move in a perpendicular line to quickly get to line segment.
        :param kv: proportional constant for linear velocity
        :param kw: proportional constant for angular velocity
        :param references: reference points.
        :return:
        """
        if not self.busy:
            self.references = [np.array(a) for a in references]
            self.point_index = 0
            self.kr = kr
            self.kv = kv
            self.kw = kw
            self.file_prefix = "ref_path_control_"
            self.epsilon = 0.6
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
            self.control_timer.cancel()
            self.send_velocity()
            self.busy = False
            self.save_odom_to_file()
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
            e = self.wrap_to_pi(e)
            w = self.kw * e
            v = self.kv * np.cos(e)
            self.logger.info(f"point: {self.point_index}; [{ti} - {ti1}]")
            self.estimate_pose()
            self.send_velocity(v, w)


def main():
    rclpy.init()
    node = Controller()

    while rclpy.ok():
        try:
            if node.initial_pose is not None:
                node.get_logger().info(f"Initial Robot Pose: {node.initial_pose}")
                break
        except Exception as e:
            print(f"Something went wrong in the ROS Loop: {e}")
        rclpy.spin_once(node)

    # reference via paths
    node.save_odom_data = True
    node.ref_path_control([(3, 0), (6, 4), (3, 4), (3, 1), (0, 3)])

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
