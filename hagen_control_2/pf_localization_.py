import scipy.stats
import sympy
from sympy import lambdify

from numpy import array, sqrt
import numpy as np
from numpy.random import randn, default_rng, uniform
from math import cos, sin, atan2
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.patches import Ellipse
import math

sympy.init_printing(use_latex='mathjax')


class PFLocalization:
    def __init__(self, dt, std_vel, std_steer, dim_x=3, dim_z=2, dim_u=2):
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

    def get_linearized_motion_model(self):
        x, y, theta, vt, wt, dt = sympy.symbols('x, y, theta, v_t, omega_t, delta_t')
        f = sympy.Matrix([[x - ((vt / wt) * sympy.sin(theta)) + ((vt / wt) * sympy.sin(theta + (wt * dt)))],
                          [y + ((vt / wt) * sympy.cos(theta)) - ((vt / wt) * sympy.cos(theta + (wt * dt)))],
                          [theta + (wt * dt)]
                          ])
        self._x, self._y, self._theta, self._vt, self._wt, self._dt = x, y, theta, vt, wt, dt
        self.state = sympy.Matrix([x, y, theta])
        self.control = sympy.Matrix([vt, wt])
        f_j = f.jacobian(self.state)
        v_j = f.jacobian(self.control)
        self.f = lambdify((x, y, theta, vt, wt, dt), f, "numpy")
        self.F_j = lambdify((x, y, theta, vt, wt, dt), f_j, "numpy")
        self.V_j = lambdify((x, y, theta, vt, wt, dt), v_j, "numpy")

    def x_forward(self, x, u, _dt):
        """

        :param x: state, x, y, theta
        :param u: control input, v, w
        :param _dt: time step
        :return:
        """
        x_plus = array(self.f(x[0], x[1], x[2], u[0], u[1], _dt)).astype(float)
        x_plus = x_plus.reshape((3, 1))
        return x_plus

    @staticmethod
    def get_linearized_measurement_model(x, landmark_pos):
        px = landmark_pos[0]
        py = landmark_pos[1]
        hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
        dist = sqrt(hyp)
        hx = array([[dist],
                    [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]], dtype=float)
        h = array([[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
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

    def residual(self, a, b):
        """ compute residual (a-b) between measurements containing
        [range, bearing]. Bearing is normalized to [-pi, pi)"""
        y = a - b
        y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[1] > np.pi:  # to [-pi, pi)
            y[1] -= 2 * np.pi
        return y

    def predict1(self, u):
        self.x = self.x_forward(self.x, u, self.dt)
        self.subs[self._x] = self.x[0, 0]
        self.subs[self._y] = self.x[1, 0]
        self.subs[self._theta] = self.x[2, 0]
        self.subs[self._vt] = u[0]
        self.subs[self._wt] = u[1]

        F = array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance in the control space
        M = array([[self.std_vel ** 2, 0], [0, self.std_steer ** 2]])

        self.P = F @ self.P @ F.T + V @ M @ V.T + self.Q

    @staticmethod
    def _predict(particles, u, std, dt=1.):
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
        particles[:, 2] = particles[:, 2] + u[1] * dt + (randn(n) * std[0])
        particles[:, 2] %= 2 * np.pi

        return particles

    def predict(self, particles, u, std, dt=1.):
        n = len(particles)
        # Update the particle next state based on the motion model defined in self. x_forward()
        n_particles = np.empty(particles.shape)
        for i in range(n):
            x = particles[i, :]
            p = self.x_forward(x, u, dt)
            n_particles[i] = p.reshape((1, 3))

        particles = n_particles
        particles[:, 2] += (self.rng.standard_normal(n) * std[0])
        particles[:, 2] %= 2 * np.pi

        return particles

    def update0(self, particles, weights, x, r, landmarks):
        weights.fill(1.)
        # distance from robot to each landmark
        NL = len(landmarks)
        z = (np.linalg.norm(landmarks - x, axis=1) + (randn(NL) * r))
        for i, landmark in enumerate(landmarks):
            # TODO calculate measurement residual, i.e., |particles - landmark|
            distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
            # TODO calculate the weighting parameters with respect to each sensor measurement
            # , i.e., scipy.stats.norm(distance, R).pdf(z[i])
            weights *= scipy.stats.norm(distance, r).pdf(z[i])

        weights += 1.e-300  # avoid round-off to zero
        # TODO normalize the weights
        weights /= sum(weights)
        return weights

    def update(self, particles, weights, state, r, landmarks):
        """"""
        x, y = state.flatten()[:2]
        # weights = np.empty(len(particles))
        weights.fill(1.)
        m = len(landmarks)
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

    def z_landmark(self, lmark, sim_pos, std_rng, std_brg):
        x, y = sim_pos[0, 0], sim_pos[1, 0]
        d = np.sqrt((lmark[0] - x) ** 2 + (lmark[1] - y) ** 2)
        a = atan2(lmark[1] - y, lmark[0] - x) - sim_pos[2, 0]
        z = np.array([[d + randn() * std_rng],
                      [a + randn() * std_brg]])
        return z

    def covariance_ellipse(self, P, deviations=1):
        U, s, _ = np.linalg.svd(P)
        orientation = math.atan2(U[1, 0], U[0, 0])
        width = deviations * math.sqrt(s[0])
        height = deviations * math.sqrt(s[1])
        if height > width:
            raise ValueError('width must be greater than height')
        return (orientation, width, height)

    def plot_covariance_ellipse(self, mean, cov, std=None, facecolor='b', edgecolor='g', alpha=0.7, ls='solid'):
        ellipse = self.covariance_ellipse(cov)
        ax = plt.gca()
        angle = np.degrees(ellipse[0])
        width = ellipse[1] * 2.
        height = ellipse[2] * 2.
        e = Ellipse(xy=mean, width=std * width, height=std * height, angle=angle, facecolor=facecolor,
                    edgecolor=edgecolor, alpha=alpha, lw=2, ls=ls)
        ax.add_patch(e)
        x, y = mean
        plt.scatter(x, y, marker='+', color=edgecolor)
        a = ellipse[0]
        h, w = height / 4, width / 4
        plt.plot([x, x + h * cos(a + np.pi / 2)], [y, y + h * sin(a + np.pi / 2)])
        plt.plot([x, x + w * cos(a)], [y, y + w * sin(a)])

    def importance_sampling(self, particles, weights, indexes):
        # TODO retrieve selected particle
        particles[:] = particles[indexes]
        # TODO retrieve selected weights
        weights[:] = weights[indexes]
        weights.fill(1.0 / len(weights))
        return weights

    def estimate(self, particles, weights):
        pos = particles[:, 0:2]
        # TODO estimate mean value of the particles
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=weights, axis=0)
        return mean, var

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def resample_particles(self, weights):
        N = len(weights)
        positions = (np.random.random() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        # TODO calculate the cumulative sum of weight distribution
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
        self.x = array([[2, 6, .3]]).T  # x, y, steer angle
        sim_pos = self.x.copy()
        u = array([1.1, .01])
        plt.scatter(particles[:, 0], particles[:, 1], alpha=0.1, color='b')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=699)
        track = []
        for i in range(iteration_num):
            sim_pos = self.x_forward(sim_pos, u, self.dt)  # simulate robot
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


class PFLocalization2:
    def __init__(self, dt, std_vel, std_steer, dim_x=3, dim_z=2, dim_u=2):
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

    def get_linearized_motion_model(self):
        x, y, theta, vt, wt, dt = sympy.symbols('x, y, theta, v_t, omega_t, delta_t')
        f = sympy.Matrix([[x - ((vt / wt) * sympy.sin(theta)) + ((vt / wt) * sympy.sin(theta + (wt * dt)))],
                          [y + ((vt / wt) * sympy.cos(theta)) - ((vt / wt) * sympy.cos(theta + (wt * dt)))],
                          [theta + (wt * dt)]
                          ])
        self._x, self._y, self._theta, self._vt, self._wt, self._dt = x, y, theta, vt, wt, dt
        self.state = sympy.Matrix([x, y, theta])
        self.control = sympy.Matrix([vt, wt])
        f_j = f.jacobian(self.state)
        v_j = f.jacobian(self.control)
        self.f = lambdify((x, y, theta, vt, wt, dt), f, "numpy")
        self.F_j = lambdify((x, y, theta, vt, wt, dt), f_j, "numpy")
        self.V_j = lambdify((x, y, theta, vt, wt, dt), v_j, "numpy")

    def x_forward(self, x, u, _dt):
        """

        :param x: state, x, y, theta
        :param u: control input, v, w
        :param _dt: time step
        :return:
        """
        x_plus = array(self.f(x[0], x[1], x[2], u[0], u[1], _dt)).astype(float)
        x_plus = x_plus.reshape((3, 1))
        return x_plus

    @staticmethod
    def get_linearized_measurement_model(x, landmark_pos):
        px = landmark_pos[0]
        py = landmark_pos[1]
        hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
        dist = sqrt(hyp)
        hx = array([[dist],
                    [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]], dtype=float)
        h = array([[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
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
        y = a.flatten() - b.flatten()
        y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[1] > np.pi:  # to [-pi, pi)
            y[1] -= 2 * np.pi
        y = y.reshape((2, 1))
        return y

    def predict1(self, u):
        self.x = self.x_forward(self.x, u, self.dt)
        self.subs[self._x] = self.x[0, 0]
        self.subs[self._y] = self.x[1, 0]
        self.subs[self._theta] = self.x[2, 0]
        self.subs[self._vt] = u[0]
        self.subs[self._wt] = u[1]

        F = array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance in the control space
        M = array([[self.std_vel ** 2, 0], [0, self.std_steer ** 2]])

        self.P = F @ self.P @ F.T + V @ M @ V.T + self.Q

    def predict(self, particles, u, std, dt=1.):
        n = len(particles)
        # TODO update the particle next state based on the motion model defined in self. x_forward()
        n_particles = np.empty(particles.shape)
        for i in range(n):
            x = particles[i, :]
            p = self.x_forward(x, u, dt)
            n_particles[i] = p.reshape((1, 3))

        particles = n_particles
        particles[:, 2] += (self.rng.standard_normal(n) * std[0])
        particles[:, 2] %= 2 * np.pi

        return particles

    @staticmethod
    def _predict(particles, u, std, dt=1.):
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
        particles[:, 2] = particles[:, 2] + u[1] * dt + (randn(n) * std[0])
        particles[:, 2] %= 2 * np.pi

        return particles

    def update2(self, particles, weights, x, R, landmarks):
        weights.fill(1.)
        # distance from robot to each landmark
        m = len(landmarks)
        z = (np.linalg.norm(landmarks - x, axis=1) + (self.rng.standard_normal(m) * R))
        for i, landmark in enumerate(landmarks):
            # TODO calculate innovation or measurement residual, i.e., |particles - landmark|
            distance = np.linalg.norm(particles[:, 0:2] - landmark)
            # TODO calculate the weighting parameters with respect to each sensor measurement
            # , i.e., scipy.stats.norm(distance, R).pdf(z[i])
            weights *= scipy.stats.norm(distance, R).pdf(z[i])

        weights += 1.e-300  # avoid round-off to zero
        # TODO normalize the weights 
        weights /= sum(weights)
        return weights

    def update(self, particles, state, r, landmarks):
        """"""
        x, y = state.flatten()[:2]
        weights = np.empty(len(particles))
        weights.fill(1.)
        m = len(landmarks)
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

    def z_landmark(self, land_mark, sim_pos, std_rng, std_brg):
        x, y, theta = sim_pos.flatten()
        d = np.sqrt((land_mark[0] - x) ** 2 + (land_mark[1] - y) ** 2)
        a = atan2(land_mark[1] - y, land_mark[0] - x) - theta
        z = np.array([[d + self.rng.standard_normal() * std_rng],
                      [a + self.rng.standard_normal() * std_brg]])
        return z

    def covariance_ellipse(self, P, deviations=1):
        U, s, _ = np.linalg.svd(P)
        orientation = math.atan2(U[1, 0], U[0, 0])
        width = deviations * math.sqrt(s[0])
        height = deviations * math.sqrt(s[1])
        if height > width:
            raise ValueError('width must be greater than height')
        return (orientation, width, height)

    def plot_covariance_ellipse(self, mean, cov, std=None, facecolor='b', edgecolor='g', alpha=0.7, ls='solid'):
        ellipse = self.covariance_ellipse(cov)
        ax = plt.gca()
        angle = np.degrees(ellipse[0])
        width = ellipse[1] * 2.
        height = ellipse[2] * 2.
        e = Ellipse(xy=mean, width=std * width, height=std * height, angle=angle, facecolor=facecolor,
                    edgecolor=edgecolor, alpha=alpha, lw=2, ls=ls)
        ax.add_patch(e)
        x, y = mean
        plt.scatter(x, y, marker='+', color=edgecolor)
        a = ellipse[0]
        h, w = height / 4, width / 4
        plt.plot([x, x + h * cos(a + np.pi / 2)], [y, y + h * sin(a + np.pi / 2)])
        plt.plot([x, x + w * cos(a)], [y, y + w * sin(a)])

    def importance_sampling(self, particles, weights: np.ndarray, indexes):
        """
        Select particles and weights based on chosen indexes
        Then normalize selected weights.

        :param particles:
        :param weights:
        :param indexes:
        :return:
        """
        # TODO retrieve selected particle
        particles[:] = particles[indexes]
        # TODO retrieve selected weights
        weights[:] = weights[indexes]
        weights.fill(1.0 / len(weights))
        return weights

    def estimate(self, particles, weights):
        pos = particles[:, 0:3]
        # TODO estimate mean value of the particles 
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=weights, axis=0)
        return mean, var

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def resample_particles(self, weights):
        n = len(weights)
        positions = (np.random.random() + np.arange(n)) / n

        indexes = np.zeros(n, 'i')
        # TODO calculate the cumulative sum of weight distribution 
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < n:
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
        weights = np.empty(N)

        xs = []
        self.x = array([[2, 6, .3]]).T  # x, y, steer angle
        sim_pos = self.x.copy()
        u = array([1.1, .01])
        plt.scatter(particles[:, 0], particles[:, 1], alpha=0.1, color='b')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=699)
        track = []
        for i in range(iteration_num):
            # print(i)
            sim_pos = self.x_forward(sim_pos, u, self.dt)  # simulate robot
            track.append(sim_pos)
            particles = self.predict(particles, u=u, std=(.02, .05), dt=self.dt)

            # incorporate measurements
            weights = self.update(particles, weights, sim_pos.flatten()[0:2], R=sensor_std_err, landmarks=landmarks)

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


dt = 0.1
_landmarks = array([[50, 100], [40, 90], [150, 150], [-150, 200]])
pfl = PFLocalization(dt, std_vel=2.0, std_steer=np.radians(1))
pfl.run_localization(100, _landmarks, initial_x=(1, 1, np.pi / 4), iteration_num=500)
# pfl.run_localization(500, _landmarks, iteration_num=200)
