import numpy as np
import sympy as sp

import nav_msgs.msg
import rclpy
from copy import deepcopy
from geometry_msgs.msg import Vector3, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sympy import lambdify

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math


class Controller(Node):
    """"""

    def __init__(self, node_name: str = "controller", sample_rate=0.033, ekf=None):
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
        self.pose_publisher = None
        self.ref_pose = [0] * 3
        self.kw = 0
        self.kv = 0
        self.d_tol = 0
        self.epsilon = 0.001
        self.file_prefix = ""
        self.control_to_ref = False
        self.references = []
        self.point_index = 0
        self.kr = 0
        self.last_index = 0
        self.ekf: EKF = ekf
        super().__init__(node_name)
        self.logger = self.get_logger()
        self.create_subscribers()
        self.create_publishers()
        self.ekf_state = None
        self.ekfs = []

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

    def estimate_pose(self):
        """
        Store estimated (calculated pose) and reported odometry.

        :return:
        """
        alpha = self.estimated_pose.z + (0.5 * self.w * self.sample_rate)
        self.estimated_pose.x = self.estimated_pose.x + (self.v * self.sample_rate * np.cos(alpha))
        self.estimated_pose.y = self.estimated_pose.y + (self.v * self.sample_rate * np.sin(alpha))
        self.estimated_pose.z = self.wrap_to_pi(self.estimated_pose.z + (self.w * self.sample_rate))
        self.pose_estimates.append(np.array([self.estimated_pose.x, self.estimated_pose.y, self.estimated_pose.z]))
        self.odometry_list.append(self.robot_pose)
        # self.pose_publisher.publish(self.estimated_pose)
        x, y, z = self.robot_pose
        msg = Vector3(x=x, y=y, z=z)
        self.pose_publisher.publish(msg)
        self.ekfs.append(self.ekf.x)

    def save_odom_to_file(self):
        """
        Save estimated pose and odometry data to file for plotting.

        :return:
        """
        if self.save_odom_data:
            np_array = np.array(self.odometry_list)
            self.get_logger().info(f"Odom Data Size: {np_array.shape}.")
            # np.save(f"{self.file_prefix}odom", np_array)
            np_array1 = np.array(self.pose_estimates)
            self.get_logger().info(f"Pose Estimate Data Size: {np_array1.shape}.")
            # np.save(f"{self.file_prefix}pose_estimates", np_array1)

            x = np_array[:, 0]
            y = np_array[:, 1]
            z = np_array[:, 2]

            # plot1(x, y, z, "Odom Pose")
            plot2(x, y, "Odom Traversed Trajectory")

            x = np_array1[:, 0]
            y = np_array1[:, 1]
            z = np_array1[:, 2]

            # plot1(x, y, z, "Estimated Pose")
            plot2(x, y, "Traversed Trajectory")

            np_arr = np.array(self.ekfs)
            x = np_arr[:, 0]
            y = np_arr[:, 1]
            z = np_arr[:, 2]

            plot1(x, y, z, "Estimated Pose")
            plot2(x, y, "EKF Traversed Trajectory")

    def ppack_a(self, a: list, b: list, kv=0.45, kw=0.41):
        """"""
        if not self.busy:
            self.kv = kv
            self.kw = kw
            self.file_prefix = "ref_ppark_a_control_"
            self.epsilon = 0.12
            self.d_tol = 0.2
            centre = (np.array(b) - np.array(a)) / 2
            self.control_timer = self.create_timer(self.sample_rate, lambda: self.ppack_a_callback(centre, b))
            self.busy = True
            self.ekf_state = self.robot_pose.flatten()
            return True
        return False

    def ppack_a_callback(self, centre, ref):
        """"""
        # x, y, theta = self.ekf_state
        x, y, theta = self.estimated_pose.x, self.estimated_pose.y, self.estimated_pose.z
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
                theta_ref = -np.pi / 2
                e = self.wrap_to_pi(theta_ref - theta)
                # e = theta_ref - theta
                w = self.kw * e
                v = self.kv * d_i
                self.logger.info(f"Approaching Intermediate point: {d_i:.4f}, theta: {theta:.4f}, theta_ref: {theta_ref:.4f}")
            self.v = v
            self.w = w
            self.send_velocity(v, w)
            self.ekf_state = self.ekf.localize(np.array([self.v, self.w]))
            self.ekf_state = self.ekf.x.flatten()


class EKF:
    """
    Extended Kalman filter
    """

    def __init__(self, dt, std_vel, std_steer, range_std=0.0, bearing_std=0.0, dim_x=3, dim_z=2):
        self.predicted_state = None
        self.landmarks = None
        self.dt = dt
        self.std_vel = std_vel
        self.std_steer = std_steer
        self.range_std, self.bearing_std = range_std, bearing_std

        self._x = self._y = self._theta = self._vt = self._wt = self._dt = 0
        self.state = self.control = None
        self.f = self.f_j = self.v_j = None

        self.get_linearized_motion_model()
        self.subs = {self._x: 0, self._y: 0, self._vt: 0, self._wt: 0, self._dt: dt, self._theta: 0}

        self.x = np.zeros((dim_x, 1))  # initial state
        # prediction/uncertainty covariance
        self.P = np.diag([.1, .1, .1])
        self.R = np.diag([range_std ** 2, bearing_std ** 2])  # state uncertainty
        self.Q = np.eye(dim_x)  # process uncertainty
        self.y = np.zeros((dim_z, 1))  # residual
        self.z = np.array([None] * dim_z)
        self.K = np.zeros(self.x.shape)  # kalman gain
        self.y = np.zeros((dim_z, 1))
        self._I = np.eye(dim_x)
        # measurement model, and it's jacobian used in the sympy version
        self.h = self.hp = None
        self.sym_motion_model()

    def set_prior(self, state: np.ndarray):
        """"""
        assert state.shape == self.x.shape
        self.x = state
        self.predicted_state = self.x.copy()

    def set_landmarks(self, landmarks):
        """Could be instead passed in constructor"""
        self.landmarks = landmarks

    def get_linearized_motion_model(self):
        x, y, theta, vt, wt, dt = sp.symbols('x, y, theta, v_t, omega_t, delta_t')
        self._x, self._y, self._theta, self._vt, self._wt, self._dt = x, y, theta, vt, wt, dt
        f = sp.Matrix([[x - ((vt / wt) * sp.sin(theta)) + ((vt / wt) * sp.sin(theta + (wt * dt)))],
                       [y + ((vt / wt) * sp.cos(theta)) - ((vt / wt) * sp.cos(theta + (wt * dt)))],
                       [theta + (wt * dt)]])
        self.state = sp.Matrix([x, y, theta])
        self.control = sp.Matrix([vt, wt])
        # linearized with respect to state
        f_j = f.jacobian(self.state)
        # linearized with respect to control
        v_j = f.jacobian(self.control)
        self.f = lambdify((x, y, theta, vt, wt, dt), f, "numpy")
        self.f_j = lambdify((x, y, theta, vt, wt, dt), f_j, "numpy")
        self.v_j = lambdify((x, y, theta, vt, wt, dt), v_j, "numpy")

    def sym_motion_model(self):
        """
        Creates a symbolic representation of the measurement model, and it's jacobian

        :return:
        """
        x, y, theta, mx, my = sp.symbols('x, y, theta, m_x, m_y')
        h = sp.Matrix([[sp.sqrt(((mx - x) ** 2) + ((my - y) ** 2))],
                       [sp.atan2(my - y, mx - x) - theta],
                       ])
        hp = h.jacobian(self.state)
        self.h = lambdify((x, y, theta, mx, my), h, "numpy")
        self.hp = lambdify((x, y, theta, mx, my), hp, "numpy")

    @staticmethod
    def get_linearized_measurement_model(x, landmark_pos):
        """
        get the measurement model and it's linearized version

        :param x: estimated state, x, y, theta
        :param landmark_pos: x & y
        :return:
        """
        px = landmark_pos[0]
        py = landmark_pos[1]
        x = x.flatten()
        x, y, theta = x
        hyp = ((px - x) ** 2) + ((py - y) ** 2)
        dist = np.sqrt(hyp)
        hx = np.array([[dist],
                       [np.arctan2(py - y, px - x) - theta]], dtype=float)
        h = np.array([[-(px - x) / dist, -(py - y) / dist, 0],
                      [(py - y) / hyp, -(px - x) / hyp, -1]], dtype=float)
        return hx, h

    def forward(self, x, u):
        """
        Get predicted next state based on previous state x, control u and time step dt

        :param x: state, x, y, theta
        :param u: control input, v, w
        :return:
        """

        xp = np.array(self.f(x[0], x[1], x[2], u[0], u[1], self.dt)).astype(float)
        xp = xp.reshape((3, 1))
        return xp

    def predict(self, u):
        self.x = self.forward(self.x.copy(), u)
        x = self.x.flatten()
        f_matrix = np.array(self.f_j(x[0], x[1], x[2], u[0], u[1], self.dt)).astype(float)
        v_matrix = np.array(self.v_j(x[0], x[1], x[2], u[0], u[1], self.dt)).astype(float)
        # fixme: maybe we should not use any covariance in the control space?
        #  can set to zero if not needed
        # covariance in the control space
        m_matrix = np.array([[self.std_vel ** 2, 0], [0, self.std_steer ** 2]])
        # KF covariance matrix with the prior knowledge
        self.P = (f_matrix @ self.P @ f_matrix.T) + (v_matrix @ m_matrix @ v_matrix.T) + self.Q

    def ekf_update(self, z, landmark: np.ndarray):
        """

        :param z:
        :param landmark: 1 x 2, flat
        :return:
        """
        # T1: Get linearized sensor measurements
        hx, h = self.get_linearized_measurement_model(self.x, landmark)
        # start v2
        # _x = self.x.flatten()
        # x, y, theta = _x
        # mx, my = _landmarks
        # h = np.array(self.hp(x, y, theta, mx, my)).astype(float)
        # hx = np.array(self.h(x, y, theta, mx, my)).astype(float)
        # end v2
        # pht = np.dot(self.P, h.T)
        # print(self.P.shape, h.shape)
        pht = self.P @ h.T
        # T2: Define the kalman gain
        self.K = pht @ np.linalg.inv(h @ pht + self.R)
        # T3: calculate the residual of sensor reading
        self.y = self.residual(z, hx)
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # P = (I-KH)P is the optimal gain
        # i_kh = self._I - np.dot(self.K, h)
        i_kh = self._I - self.K @ h
        # T4: Define the KF covariance matrix with the posterior knowledge
        self.P = i_kh @ self.P
        self.z = deepcopy(z)

    @staticmethod
    def residual(a, b):
        """ compute residual (a-b) between measurements containing
        [range, bearing]. Bearing is normalized to [-pi, pi)"""
        y = a.flatten() - b.flatten()
        y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[1] > np.pi:  # to [-pi, pi)
            y[1] -= 2 * np.pi
        y = y.reshape((2, 1))
        return y

    def get_measurement(self, landmark, pose):
        """
        Also known as observation (z)

        :param landmark:
        :param pose:
        :return:
        """
        x, y, z_ = pose.flatten()
        _range = np.sqrt(((landmark[0] - x) ** 2) + ((landmark[1] - y) ** 2))
        bearing = np.arctan2(landmark[1] - y, landmark[0] - x) - z_
        z = np.array([[_range + (np.random.randn() * self.range_std)],
                      [bearing + (np.random.randn() * self.bearing_std)]])
        return z

    def localize(self, control, step=None):
        """"""
        self.predicted_state = self.forward(self.predicted_state, control)
        self.predict(control)
        p_state = self.predicted_state.copy()
        for landmark in self.landmarks:
            z = self.get_measurement(landmark, p_state)
            self.ekf_update(z, landmark)
        # return self.x.flatten()
        return self.predicted_state.flatten()

    @staticmethod
    def covariance_ellipse(p, deviations=1):
        u, s, _ = np.linalg.svd(p)
        orientation = math.atan2(u[1, 0], u[0, 0])
        width = deviations * math.sqrt(s[0])
        height = deviations * math.sqrt(s[1])
        if height > width:
            raise ValueError('width must be greater than height')
        return orientation, width, height

    def plot_covariance_ellipse(self, mean, cov, std=None, facecolor='b', edgecolor='g', alpha=0.7, ls='solid'):
        ellipse = self.covariance_ellipse(cov)
        ax = plt.gca()
        angle = np.degrees(ellipse[0])
        width = ellipse[1] * 2.
        height = ellipse[2] * 2.
        # noinspection PyTypeChecker
        e = Ellipse(xy=mean, width=std * width, height=std * height, angle=angle, facecolor=facecolor,
                    edgecolor=edgecolor, alpha=alpha, lw=2, ls=ls)
        ax.add_patch(e)
        x, y = mean
        plt.scatter(x, y, marker='+', color=edgecolor)
        a = ellipse[0]
        h, w = height / 4, width / 4
        plt.plot([x, x + h * np.cos(a + np.pi / 2)], [y, y + h * np.sin(a + np.pi / 2)])
        plt.plot([x, x + w * np.cos(a)], [y, y + w * np.sin(a)])


def main():
    rclpy.init()
    dt = 0.02
    ekf = EKF(dt=dt, std_vel=0.3, std_steer=np.radians(1), range_std=0.3, bearing_std=0.1)
    node = Controller(sample_rate=dt, ekf=ekf)

    while rclpy.ok():
        try:
            if node.initial_pose is not None:
                node.get_logger().info(f"Initial Robot Pose: {node.initial_pose}")
                break
        except Exception as e:
            print(f"Something went wrong in the ROS Loop: {e}")
        rclpy.spin_once(node)

    import time
    count = 0
    max_ = 0
    while count <= max_:
        # used to ensure plot juggler is latched
        node.pose_publisher.publish(node.estimated_pose)
        time.sleep(1)
        count += 1
    # ppack
    node.save_odom_data = True
    node.ekf.set_prior(node.robot_pose.reshape(3, 1))
    landmarks = [(5, 30), (5, -30), (-5, 0)]
    node.ekf.set_landmarks(landmarks)
    node.ppack_a([0, 0], [10, -10], kv=0.33, kw=0.32)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def check():
    """"""
    dt = 0.1
    landmarks = np.array([[50, 100], [40, 90], [150, 150], [-150, 200]])
    ekf = EKF(dt=dt, std_vel=5.1, std_steer=np.radians(1), range_std=0.3, bearing_std=0.1)
    ekf.set_prior(np.array([[2, 6, .3]]).T)
    ekf.set_landmarks(landmarks)
    iteration_num = 20000
    u = np.array([1.1, 0.01])
    pos = ekf.x.copy()
    plt.figure()
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=60)
    track = []
    state = []
    for i in range(iteration_num):
        pos = ekf.forward(pos, u)
        track.append(pos)
        ellipse_step = 2000
        if i % 10 == 0:
            ekf.predict(u)
            if i % ellipse_step == 0:
                ekf.plot_covariance_ellipse((ekf.x[0, 0], ekf.x[1, 0]), ekf.P[0:2, 0:2], std=6, facecolor='k',
                                            alpha=0.3)
            for landmark in landmarks:
                z = ekf.get_measurement(landmark, pos)
                ekf.ekf_update(z, landmark)
            if i % ellipse_step == 0:
                ekf.plot_covariance_ellipse((ekf.x[0, 0], ekf.x[1, 0]), ekf.P[0:2, 0:2], std=6, facecolor='g',
                                            alpha=0.8)
        state.append(ekf.x.flatten())
    track = np.array(track)

    plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
    plt.axis('equal')
    plt.title("EKF Robot localization")
    plt.show()

    state = np.array(state)
    plt.plot(state[:, 0], state[:, 1], color='r', lw=2)
    plt.axis('equal')
    plt.title("EKF Robot localization 2")
    plt.show()


def analytical():
    import time
    dt = 0.04
    landmarks = [(5, 30), (5, -30), (-5, 0)]

    start = np.array([0, 0])
    end = np.array([30, -30])
    prior = np.array([0, 0, 0])
    centre = (end - start) / 2

    kv = 0.45
    kw = 0.43
    epsilon = 0.2
    d_tol = 0.1

    states = []
    states2 = []
    ekf_state = prior.flatten()

    ekf = EKF(dt=dt, std_vel=0.5, std_steer=np.radians(1), range_std=0.1, bearing_std=0.1)
    ekf.set_prior(prior.reshape(3, 1))
    ekf.set_landmarks(landmarks)
    control_to_ref = False

    count = 0
    while count < 710:
        count += 1
        states.append(ekf_state)
        states2.append(ekf.x.flatten())
        x, y, theta = ekf_state
        xr, yr = end
        d = np.sqrt(((xr - x) ** 2) + ((yr - y) ** 2))
        if d <= epsilon:
            break
        else:
            if control_to_ref:
                e = -theta
                e = Controller.wrap_to_pi(e)
                w = kw * e
                v = kv * d
                print(f"Moving to target: {d:.4f}, e: {e:.4f}")
            else:
                xi, yi = centre
                d_i = np.sqrt(((xi - x) ** 2) + ((yi - y) ** 2))
                control_to_ref = d_i <= d_tol
                theta_ref = -np.pi/2
                e = Controller.wrap_to_pi(theta_ref - theta)
                w = kw * e
                v = kv * d_i
                print(f"Approaching intermediate point: {d_i:.4f}")
        ekf_state = ekf.localize(np.array([v, w]))
        # time.sleep(0.05)
    print(f"finished: {len(states)}")

    np_array = np.array(states)
    x = np_array[:, 0]
    y = np_array[:, 1]
    z = np_array[:, 2]

    # plot1(x, y, z, "Estimated Pose")
    plot2(x, y, "Traversed Trajectory", landmarks=np.array(landmarks))

    np_array = np.array(states2)
    x = np_array[:, 0]
    y = np_array[:, 1]
    z = np_array[:, 2]

    # plot1(x, y, z, "Estimated Pose")
    plot2(x, y, "Traversed Trajectory", landmarks=np.array(landmarks))


def plot1(x, y, z, title, landmarks=None):
    _, ax = plt.subplots()
    plt.plot(x, 'r', linewidth=2.0)
    plt.plot(y, 'g', linewidth=2.0)
    plt.plot(z, 'b', linewidth=2.0)

    plt.grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    plt.grid(True)
    plt.ylabel(r'$x$(m), $y$(m), $\theta$(rad)')
    plt.xlabel(r'Time $t$ (s)')
    ax.set_title(title)
    plt.legend(['x', 'y', 'theta'])
    plt.show()


def plot2(x, y, title, landmarks=None):
    _, ax = plt.subplots()
    plt.plot(x, y, 'r', linewidth=2.0)
    if landmarks is not None:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=60)
    plt.grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    plt.grid(True)
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    # main()
    # check()
    analytical()
