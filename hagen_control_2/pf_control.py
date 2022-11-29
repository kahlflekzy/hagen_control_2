import numpy as np
import scipy.stats
import sympy as sp

import nav_msgs.msg
import rclpy
from geometry_msgs.msg import Vector3, Twist
from nav_msgs.msg import Odometry
from numpy.random import default_rng
from rclpy.node import Node
from sympy import lambdify

import matplotlib.pyplot as plt


class Controller(Node):
    """"""

    def __init__(self, node_name: str = "controller", sample_rate=0.033, filter=None):
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
        self.filter: PF = filter
        super().__init__(node_name)
        self.logger = self.get_logger()
        self.create_subscribers()
        self.create_publishers()
        self.pf_state = None
        self.states = []

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
        self.states.append(self.ekf.x)

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

            np_arr = np.array(self.states)
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
        # x, y, theta = self.pf_state
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
                self.logger.info(
                    f"Approaching Intermediate point: {d_i:.4f}, theta: {theta:.4f}, theta_ref: {theta_ref:.4f}")
            self.v = v
            self.w = w
            self.send_velocity(v, w)
            self.pf_state = self.filter.localize(np.array([self.v, self.w]))
            self.pf_state = self.filter.x.flatten()


class PF:
    def __init__(self, dt, std_vel, std_steer, dim_x=3, dim_z=2, dim_u=2, range_std=None, bearing_std=None):
        self.dt = dt
        self.std_vel = std_vel
        self.std_steer = std_steer
        self.get_linearized_motion_model()
        self.subs = {self._x: 0, self._y: 0, self._vt: 0, self._wt: 0, self._dt: dt, self._theta: 0}
        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)  # uncertainty covariance
        self.R = np.eye(dim_z)  # state uncertainty
        self.Q = np.eye(dim_x)  # process uncertainty
        self.y = np.zeros((dim_z, 1))  # residual
        self.z = np.array([None] * dim_z)
        self.K = np.zeros(self.x.shape)  # kalman gain
        self.y = np.zeros((dim_z, 1))
        self._I = np.eye(dim_x)

        # my additions
        self._x = self._y = self._theta = self._vt = self._wt = self._dt = 0
        self.state = self.control = None
        self.f = self.F_j = self.V_j = None

        self.get_linearized_motion_model()

        self.rng = default_rng()

        self.range_std, self.bearing_std = range_std, bearing_std
        self.predicted_state = self.landmarks = self.particles = self.weights = None

    def get_linearized_motion_model(self):
        x, y, theta, vt, wt, dt = sp.symbols('x, y, theta, v_t, omega_t, delta_t')
        f = sp.Matrix([[x - ((vt / wt) * sp.sin(theta)) + ((vt / wt) * sp.sin(theta + (wt * dt)))],
                       [y + ((vt / wt) * sp.cos(theta)) - ((vt / wt) * sp.cos(theta + (wt * dt)))],
                       [theta + (wt * dt)]
                       ])
        self._x, self._y, self._theta, self._vt, self._wt, self._dt = x, y, theta, vt, wt, dt
        self.state = sp.Matrix([x, y, theta])
        self.control = sp.Matrix([vt, wt])
        f_j = f.jacobian(self.state)
        v_j = f.jacobian(self.control)
        self.f = lambdify((x, y, theta, vt, wt, dt), f, "numpy")
        self.F_j = lambdify((x, y, theta, vt, wt, dt), f_j, "numpy")
        self.V_j = lambdify((x, y, theta, vt, wt, dt), v_j, "numpy")

    def set_prior(self, state: np.ndarray, n=100, std=None):
        """"""
        assert state.shape == self.x.shape
        self.x = state
        self.predicted_state = self.x.copy()
        if std:
            self.particles = self.create_gaussian_particles(mean=state, std=std, n=n)
        else:
            self.particles = self.create_uniform_particles((0, 40), (0, 40), (0, 6.28), n)
        self.weights = np.zeros(n)

    def set_landmarks(self, landmarks):
        """Could be instead passed in constructor"""
        self.landmarks = landmarks

    def forward(self, x, u, _dt):
        """

        :param x: state, x, y, theta
        :param u: control input, v, w
        :param _dt: time step
        :return:
        """
        x_plus = np.array(self.f(x[0], x[1], x[2], u[0], u[1], _dt)).astype(float)
        x_plus = x_plus.reshape((3, 1))
        return x_plus

    @staticmethod
    def get_linearized_measurement_model(x, landmark_pos):
        px = landmark_pos[0]
        py = landmark_pos[1]
        hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
        dist = np.sqrt(hyp)
        hx = np.array([[dist],
                       [np.atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]], dtype=float)
        h = np.array([[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
                      [(py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1]], dtype=float)
        return hx, h

    def create_gaussian_particles(self, mean, std, n):
        """
        Generate particles having mean and std.
        Normalize the theta part of the particles.
        Updated to use the new style Random Number Generator.

        :param mean: 3,
        :param std: 3,
        :param n:
        :return:
        """
        rng = self.rng
        particles = np.empty((n, 3))
        particles[:, 0] = mean[0] + (rng.standard_normal(n) * std[0])
        particles[:, 1] = mean[1] + (rng.standard_normal(n) * std[1])
        particles[:, 2] = mean[2] + (rng.standard_normal(n) * std[2])
        particles[:, 2] %= 2 * np.pi
        return particles

    def create_uniform_particles(self, x_range, y_range, hdg_range, n):
        """
        Generate uniformly distributed particles.

        :param x_range: 2, (low, high)
        :param y_range: 2, (low, high)
        :param hdg_range: 2, (low, high)
        :param n: sample size
        :return:
        """
        rng = self.rng
        particles = np.empty((n, 3))
        particles[:, 0] = rng.uniform(x_range[0], x_range[1], size=n)
        particles[:, 1] = rng.uniform(y_range[0], y_range[1], size=n)
        particles[:, 2] = rng.uniform(hdg_range[0], hdg_range[1], size=n)
        particles[:, 2] %= 2 * np.pi
        return particles

    @staticmethod
    def residual(a, b):
        """ compute residual (a-b) between measurements containing
        [range, bearing]. Bearing is normalized to [-pi, pi)"""
        y = a - b
        y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[1] > np.pi:  # to [-pi, pi)
            y[1] -= 2 * np.pi
        return y

    def _predict(self, particles, u, std, dt=1.):
        """
        Geesara's version. More efficient

        :param particles:
        :param u:
        :param std:
        :param dt:
        :return:
        """
        r = u[0] / u[1]
        theta = particles[:, 2]
        rotation = particles[:, 2] + u[1] * dt
        n = len(particles)

        particles[:, 0] = particles[:, 0] + -r * np.sin(theta) + r * np.sin(rotation)
        particles[:, 1] = particles[:, 1] + r * np.cos(theta) - r * np.cos(rotation)
        particles[:, 2] = particles[:, 2] + u[1] * dt + (self.rng.standard_normal(n) * std[0])
        particles[:, 2] %= 2 * np.pi

        return particles

    def predict(self, particles, u, std, dt=1.):
        """

        :param particles: n x dim_x
        :param u: dim_u x 1
        :param std: dim_z x 1 (some noise added to the orientation)
        :param dt: sampling time
        :return:
        """
        n = len(particles)
        # Update the particle next state based on the motion model defined in self. x_forward()
        n_particles = np.empty(particles.shape)
        for i in range(n):
            x = particles[i, :]
            p = self.forward(x, u, dt)
            n_particles[i] = p.reshape((1, 3))

        particles = n_particles
        particles[:, 2] += (self.rng.standard_normal(n) * std[0])
        particles[:, 2] %= 2 * np.pi

        return particles

    def update1(self, particles, weights, x, r, landmarks):
        weights.fill(1.)
        # distance from robot to each landmark
        nl = len(landmarks)
        z = (np.linalg.norm(landmarks - x, axis=1) + (self.rng.standard_normal(nl) * r))
        for i, landmark in enumerate(landmarks):
            # Calculate measurement residual, i.e., |particles - landmark|
            distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
            # Calculate the weighting parameters with respect to each sensor measurement
            weights *= scipy.stats.norm(distance, r).pdf(z[i])

        weights += 1.e-300  # avoid round-off to zero
        # Normalize the weights
        weights /= sum(weights)
        return weights

    def update(self, particles, weights, state, r, landmarks):
        """
        Generate weights natively by getting measurements, calculating innovations and
        using the normal distribution to calculate the weights.

        :param particles:
        :param weights:
        :param state:
        :param r:
        :param landmarks:
        :return:
        """
        x, y = state.flatten()[:2]
        weights.fill(1.)
        # calculate distance between robot and landmarks to generate some measurement
        z = np.array(
            [(np.sqrt((yi - y) ** 2) + ((xi - x) ** 2)) + (self.rng.standard_normal() * r) for xi, yi in landmarks])
        # calculate distance between particles and landmarks
        innovations = []
        for particle in particles:
            x, y = particle.flatten()[:2]
            # calculate z_ from this particle to all landmarks
            z_ = np.array([np.sqrt((yi - y) ** 2) + ((xi - x) ** 2)
                           for xi, yi in landmarks])
            innovations.append(z - z_)
        # innovations = np.array(innovations)
        d = 1 / np.sqrt(2 * np.pi * r)
        r_inv = (1 / r)
        # calculate weights
        weights = np.array([(d * np.exp(-0.5 * ((inv * r_inv) @ inv))) for inv in innovations])
        # readjust to avoid round-off to zero
        weights += 1E-300
        # normalize
        weights /= sum(weights)
        return weights

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

    @staticmethod
    def importance_sampling(particles, weights, indexes):
        # Retrieve selected particle
        particles[:] = particles[indexes]
        # Retrieve selected weights
        weights[:] = weights[indexes]
        weights.fill(1.0 / len(weights))
        return weights

    @staticmethod
    def estimate(particles, weights):
        # todo 1 look into this `particles[:, 0:2]`
        # todo 2 try to incorporate bearing into measurements
        pos = particles #[:, 0:2]
        # Estimate mean value of the particles
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=weights, axis=0)
        return mean, var

    @staticmethod
    def neff(weights):
        return 1. / np.sum(np.square(weights))

    @staticmethod
    def resample_particles(weights):
        N = len(weights)
        positions = (np.random.random() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        # Calculate the cumulative sum of weight distribution
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def run_localization(self, N, landmarks, iteration_num=18, sensor_std_err=.1, initial_x=None):
        plt.figure()
        # create particles and weights
        if initial_x is not None:
            particles = self.create_gaussian_particles(mean=initial_x, std=(5, 5, np.pi / 4), n=N)
        else:
            particles = self.create_uniform_particles((0, 50), (0, 50), (0, 6.28), N)
        weights = np.zeros(N)

        xs = []
        self.x = np.array([[2, 6, .3]]).T  # x, y, steer angle
        sim_pos = self.x.copy()
        u = np.array([1.1, .01])
        plt.scatter(particles[:, 0], particles[:, 1], alpha=0.1, color='b')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=699)
        track = []
        for i in range(iteration_num):
            sim_pos = self.forward(sim_pos, u, self.dt)  # simulate robot
            track.append(sim_pos)
            particles = self.predict(particles, u=u, std=(.02, .05), dt=self.dt)
            # incorporate measurements
            weights = self.update(particles, weights, sim_pos.flatten()[0:2], r=sensor_std_err, landmarks=landmarks)
            if self.neff(weights) < N / 2:
                indexes = self.resample_particles(weights)
                weights = self.importance_sampling(particles, weights, indexes)

            mu, var = self.estimate(particles, weights)
            xs.append(mu)
            p1 = plt.scatter(sim_pos[0], sim_pos[1], marker='+', color='k', s=180, lw=3)
            p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
            # p3 = plt.scatter(particles[:, 0], particles[:, 1], alpha=0.1)

        xs = np.array(xs)
        plt.legend([p1, p2], ['Real', 'PF'], loc=4, numpoints=1)
        plt.show()

    def localize(self, control):
        """"""
        self.predicted_state = self.forward(self.predicted_state, control, self.dt)
        self.particles = self.predict(particles=self.particles, u=control, std=(.02, .05), dt=self.dt)
        self.weights = self.update(self.particles, self.weights, self.predicted_state.flatten()[:2], r=self.range_std,
                                   landmarks=self.landmarks)
        if self.neff(self.weights) < len(self.particles)/2:
            indexes = self.resample_particles(self.weights)
            self.weights = self.importance_sampling(self.particles, self.weights, indexes)

        self.x, var = self.estimate(self.particles, self.weights)
        return self.x.flatten()


def main():
    rclpy.init()
    dt = 0.04
    pf = PF(dt=dt, std_vel=2.0, std_steer=np.radians(1), range_std=0.1, bearing_std=0.1)
    node = Controller(sample_rate=dt, filter=pf)

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


def analytical():
    dt = 0.06
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

    pf = PF(dt=dt, std_vel=2.0, std_steer=np.radians(1), range_std=0.1, bearing_std=0.1)
    pf.set_prior(prior.reshape(3, 1), std=(5, 5, np.pi/4))
    pf.set_landmarks(landmarks)
    control_to_ref = False

    count = 0
    while count < 710:
        count += 1
        states.append(ekf_state)
        states2.append(pf.x.flatten())
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
                theta_ref = -np.pi / 2
                e = Controller.wrap_to_pi(theta_ref - theta)
                w = kw * e
                v = kv * d_i
                print(f"Approaching intermediate point: {d_i:.4f}")
        ekf_state = pf.localize(np.array([v, w]))
        ekf_state = pf.predicted_state.flatten()
    print(f"finished: {len(states)}")

    np_array = np.array(states)
    x = np_array[:, 0]
    y = np_array[:, 1]
    z = np_array[:, 2]

    plot1(x, y, z, "Estimated Pose")
    plot2(x, y, "Estimated Trajectory", landmarks=np.array(landmarks))

    np_array = np.array(states2)
    x = np_array[:, 0]
    y = np_array[:, 1]
    z = np_array[:, 2]

    plot1(x, y, z, "Actual Pose")
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
