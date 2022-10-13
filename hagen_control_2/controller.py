import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
import numpy as np


class ControlInputs:
    Linear = 0
    Angular = 1
    LinearAngular = 2
    AngularWheels = 3


class MinimalPublisher(Node):
    def __init__(self, delta_t):
        super().__init__('minimal_publisher')
        self.msg = None
        self.publisher_ = self.create_publisher(Twist, '/hagen/cmd_vel', 30)
        self.vel_sub = self.create_subscription(Twist, '/hagen/cmd_vel', self.listener_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/hagen/odom", self.set_pose, 20)
        self.publisher_2 = self.create_publisher(Vector3, '/hagen/pose', 30)
        self.i = 0
        self.set_q_init = None
        self.r = 0.04  # Wheel radius
        self.L = 0.08  # Axle length
        self.D = 0.07  # Distance between the front wheel and rear axle
        self.Ts = delta_t  # Sampling time
        self.t = np.arange(0, 10, self.Ts)  # Simulation time
        self.on_task = False
        self.timer = None
        self.epoch = 0
        self.duration = 0
        # You can comment the items in the list leaving below one line at a time
        # To see the path for the individual velocities
        # Leaving all will make the robot execute all the paths 5 seconds each, sequentially.
        self.velocities = [
            ([0.5], ControlInputs.Linear),
            ([1.0, 2.0], ControlInputs.LinearAngular),
            ([2.0], ControlInputs.Angular),
            ([20.0, 18.0], ControlInputs.AngularWheels)
           ]
        # set this to True to store numpy data for plotting graphs
        self.save_data = False
        self.odom = []
        self.pose_estimates = []

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def listener_callback(self, msg):
        pass

    def wrap_to_pi(self, x):
        x = np.array([x])
        xwrap = np.remainder(x, 2 * np.pi)
        mask = np.abs(xwrap) > np.pi
        xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
        return xwrap[0]

    def set_pose(self, msg):
        _, _, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.set_q_init = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

    def send_vel(self, v, w):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.publisher_.publish(msg)

    def control(self, duration=5):
        self.get_logger().info("In controller")
        self.duration = int((1 / self.Ts) * duration)
        if not self.on_task:
            x, y, z = self.set_q_init
            self.msg = Vector3(x=x, y=y, z=z)
            self.get_logger().info(f"Initial pose: {self.msg}")
            self.timer = self.create_timer(self.Ts, self.call_back)
            self.new_task()

    def call_back(self):
        if self.epoch == self.duration:
            self.epoch = 0
            self.on_task = False
            self.send_vel(0.0, 0.0)
            self.get_logger().info("Finished task. Resetting.")
            self.get_logger().info(f"Odom Size: {len(self.odom)}.")
            if len(self.velocities) == 0:
                self.timer.cancel()
                if self.save_data:
                    np_array = np.array(self.odom)
                    self.get_logger().info(f"Odom Data Size: {np_array.shape}.")
                    np.save("odom", np_array)
                    np_array = np.array(self.pose_estimates)
                    self.get_logger().info(f"Pose Estimate Data Size: {np_array.shape}.")
                    np.save("pose_estimates", np_array)
            else:
                self.new_task()
        else:
            alpha = self.msg.z + (0.5*self.w*self.Ts)
            self.msg.x = self.msg.x + (self.v * self.Ts * np.cos(alpha))
            self.msg.y = self.msg.y + (self.v * self.Ts * np.sin(alpha))
            # todo set below to use wrap around
            self.msg.z = self.msg.z + (self.w * self.Ts)
            self.publisher_2.publish(self.msg)
            self.pose_estimates.append(np.array([self.msg.x, self.msg.y, self.msg.z]))
            self.odom.append(self.set_q_init)
            self.epoch += 1
            self.send_vel(self.v, self.w)
            # self.get_logger().info(f"Moving: {self.epoch}/{self.duration}")

    def new_task(self):
        self.get_logger().info("New Task")
        self.on_task = True
        v, w = self.get_velocities()
        self.v, self.w = v, w
        self.send_vel(v, w)

    def get_velocities(self):
        """
        Return linear and angular velocities based on control input type.

        :return:
        """
        input_ = self.velocities.pop(0)
        if input_[1] == ControlInputs.Linear:
            return input_[0][0], 0.0
        elif input_[1] == ControlInputs.LinearAngular:
            return input_[0][0], input_[0][1]
        elif input_[1] == ControlInputs.Angular:
            return 0.0, input_[0][0]
        elif input_[1] == ControlInputs.AngularWheels:
            transition = np.array([[self.r*0.5, self.r*0.5],
                                   [-self.r/self.L, self.r/self.L]])
            wl, wr = input_[0][0], input_[0][1]  # left and right wheel angular velocities
            control_input = np.array([wl, wr])
            v, w = transition@control_input  # robot tangential, angular velocity
            self.get_logger().info(f"v: [{v}]. w: [{w}]")
            return v, w
        else:
            return 0.0, 0.0

    def control_via_intermediate(self, r_d=1.3, ref_pose=np.array([3, 6, 0]), kp=5, kw=10):
        """"""
        current_pose = self.set_q_init
        x_ref = ref_pose[0]
        y_ref = ref_pose[1]
        theta_ref = ref_pose[2]
        x, y, theta = current_pose
        D = np.sqrt(((x_ref - x) ** 2) - ((y_ref - y) ** 2))
        self.get_logger().info(f"Error: {D}")
        beta = np.arctan(r_d / D)
        phi_r = np.arctan2(y_ref - y, x_ref - x)
        alpha = self.wrap_to_pi(phi_r - theta)
        if alpha < 0:
            beta = -beta
        v = kp * D
        e = theta_ref - theta + (alpha if np.abs(alpha) < np.abs(beta) else beta)
        w = kw * e
        self.send_vel(v, w)
        self.epoch += 1
        if self.epoch == self.duration:
            self.send_vel(0.0, 0.0)
            self.timer.cancel()

    def action(self, duration=10):
        self.duration = int((1 / self.Ts) * duration)
        self.epoch = 0
        self.timer = self.create_timer(self.Ts, self.control_via_intermediate)

    def control_to_ref__via_inter(self, r, ref_pose):
        """
        TODO:

        :param r:
        :param ref_pose:
        :return:
        """
        x, y, theta = self.set_q_init
        xr, yr, theta_r = ref_pose
        x_t = xr - r * np.cos(theta_r)
        y_t = yr - r * np.sin(theta_r)

        c_dist = np.sqrt(((x - x_t)**2) - ((y - y_t)**2))
        if c_dist < self.dmin:
            # todo: define self.dmin
            pass

SPIN_QUEUE = []
PERIOD = 0.01


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher(delta_t=0.033)
    SPIN_QUEUE.append(minimal_publisher)
    while rclpy.ok():
        try:
            if minimal_publisher.set_q_init is not None:
                print(f"Init value is set: {minimal_publisher.set_q_init}")
                break
            # for node in SPIN_QUEUE:
            #     rclpy.spin_once(node, timeout_sec=(PERIOD / len(SPIN_QUEUE)))
        except Exception as e:
            print(f"something went wrong in the ROS Loop: {e}")
        rclpy.spin_once(minimal_publisher)

    # minimal_publisher.action(10)
    minimal_publisher.control()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
