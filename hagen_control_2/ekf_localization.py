import math
from copy import deepcopy
from math import cos, sin, atan2
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.patches import Ellipse
from numpy import array, sqrt
from numpy.random import randn
from sympy import lambdify

sympy.init_printing(use_latex='mathjax')


class EKFLocalization:
    def __init__(self, dt, std_vel, std_steer, dim_x=3, dim_z=2):
        self.dt = dt
        self.std_vel = std_vel
        self.std_steer = std_steer

        self._x = self._y = self._theta = self._vt = self._wt = self._dt = 0
        self.state = self.control = None
        self.f = self.F_j = self.V_j = None

        self.get_linearized_motion_model()
        self.subs = {self._x: 0, self._y: 0, self._vt: 0, self._wt: 0, self._dt: dt, self._theta: 0}

        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)  # uncertainty covariance
        self.R = np.eye(dim_z)  # state uncertainty
        self.Q = np.eye(dim_x)  # process uncertainty
        self.y = np.zeros((dim_z, 1))  # residual
        self.z = np.array([None] * dim_z)
        self.K = np.zeros(self.x.shape)  # kalman gain
        self._I = np.eye(dim_x)

    def get_linearized_motion_model(self):
        x, y, theta, vt, wt, dt = sympy.symbols('x, y, theta, v_t, omega_t, delta_t')
        # define the kinematic model
        f = sympy.Matrix([[x - ((vt / wt) * sympy.sin(theta)) + ((vt / wt) * sympy.sin(theta + (wt * dt)))],
                          [y + ((vt / wt) * sympy.cos(theta)) - ((vt / wt) * sympy.cos(theta + (wt * dt)))],
                          [theta + (wt * dt)]
                          ])
        self._x, self._y, self._theta, self._vt, self._wt, self._dt = x, y, theta, vt, wt, dt
        self.state = sympy.Matrix([x, y, theta])
        self.control = sympy.Matrix([vt, wt])
        # Task calculate the jacobian with respect to self.state
        f_j = f.jacobian(self.state)
        # Task calculate the jacobian with respect to self.control
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
        self.subs[self._x] = x[0]
        self.subs[self._y] = x[1]
        self.subs[self._theta] = x[2]
        self.subs[self._vt] = u[0]
        self.subs[self._wt] = u[1]
        x_plus = array(self.f(x[0], x[1], x[2], u[0], u[1], self.dt)).astype(float)
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

    def predict(self, u):
        self.x = self.x_forward(self.x, u, self.dt)
        x = self.x.flatten()
        self.subs[self._x] = self.x[0, 0]
        self.subs[self._y] = self.x[1, 0]
        self.subs[self._theta] = self.x[2, 0]
        self.subs[self._vt] = u[0]
        self.subs[self._wt] = u[1]
        f_matrix = array(self.F_j(x[0], x[1], x[2], u[0], u[1], self.dt)).astype(float)
        v_matrix = array(self.V_j(x[0], x[1], x[2], u[0], u[1], self.dt)).astype(float)
        # covariance in the control space
        m_matrix = array([[self.std_vel ** 2, 0], [0, self.std_steer ** 2]])
        # KF covariance matrix with the prior knowledge
        self.P = (f_matrix @ self.P @ f_matrix.T) + (v_matrix @ m_matrix @ v_matrix.T) + self.Q

    def ekf_update(self, z, _landmarks):
        # T1: Get linearized sensor measurements
        hx, h = self.get_linearized_measurement_model(self.x, _landmarks)
        pht = np.dot(self.P, h.T)
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
    def z_landmark(land_mark, sim_pos, std_rng, std_brg):
        x, y = sim_pos[0, 0], sim_pos[1, 0]
        d = np.sqrt((land_mark[0] - x) ** 2 + (land_mark[1] - y) ** 2)
        a = atan2(land_mark[1] - y, land_mark[0] - x) - sim_pos[2, 0]
        z = np.array([[d + randn() * std_rng],
                      [a + randn() * std_brg]])
        return z

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
        plt.plot([x, x + h * cos(a + np.pi / 2)], [y, y + h * sin(a + np.pi / 2)])
        plt.plot([x, x + w * cos(a)], [y, y + w * sin(a)])

    def run_localization(self, land_marks, std_range, std_bearing, step=10, ellipse_step=2000, ylim=None,
                         iteration_num=5):
        self.x = array([[2, 6, .3]]).T  # x, y, steer angle
        self.P = np.diag([.1, .1, .1])
        self.R = np.diag([std_range ** 2, std_bearing ** 2])
        sim_pos = self.x.copy()
        u = array([1.1, .01])
        plt.figure()
        plt.scatter(land_marks[:, 0], land_marks[:, 1], marker='s', s=60)
        track = []
        for i in range(iteration_num):
            sim_pos = self.x_forward(sim_pos, u, self.dt)  # simulate robot
            track.append(sim_pos)
            if i % step == 0:
                self.predict(u=u)
                if i % ellipse_step == 0:
                    self.plot_covariance_ellipse((self.x[0, 0], self.x[1, 0]), self.P[0:2, 0:2], std=6, facecolor='k',
                                                 alpha=0.3)

                for landmark in land_marks:
                    z = self.z_landmark(landmark, sim_pos, std_range, std_bearing)
                    self.ekf_update(z, landmark)
                if i % ellipse_step == 0:
                    self.plot_covariance_ellipse((self.x[0, 0], self.x[1, 0]), self.P[0:2, 0:2], std=6, facecolor='g',
                                                 alpha=0.8)
        track = np.array(track)
        plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
        plt.axis('equal')
        plt.title("EKF Robot localization")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.show()
        return ekf


class EKF2(EKFLocalization):
    def __init__(self, dt, std_vel, std_steer, dim_z=8):
        super().__init__(dt, std_vel, std_steer, dim_z=dim_z)
        self.dim_z = dim_z
        self.sensor_type = 2
        self.dim_x = 3

    def ekf_update(self, z, _landmarks):
        # T1: Get linearized sensor measurements
        hx_total = np.empty((self.dim_z, 1))
        h_total = np.empty((self.dim_z, self.dim_x))
        for ind, landmark in enumerate(landmarks):
            hx, h = self.get_linearized_measurement_model(self.x, landmark)
            hx_total[ind * self.sensor_type:(ind + 1) * self.sensor_type, 0:self.dim_x] = hx
            h_total[ind * self.sensor_type:(ind + 1) * self.sensor_type, :] = h
        h = h_total
        hx = hx_total

        pht = np.dot(self.P, h.T)
        # T2: Define the kalman gain
        self.K = pht @ np.linalg.inv(h @ pht + self.R)
        # T3: calculate the residual of sensor reading
        y = np.empty((self.dim_z, 1))
        for i in range(0, self.dim_z, self.sensor_type):
            _z = z[i:i + self.sensor_type]
            _hx = hx[i:i + self.sensor_type]
            y[i:i + self.sensor_type] = self.residual(_z, _hx)
        self.y = y
        self.x = self.x + np.dot(self.K, self.y)

        i_kh = self._I - self.K @ h
        # T4: Define the KF covariance matrix with the posterior knowledge
        self.P = i_kh @ self.P
        self.z = deepcopy(z)

    def run_localization(self, land_marks, std_range, std_bearing, step=10, ellipse_step=2000, ylim=None,
                         iteration_num=5):
        self.x = array([[2, 6, .3]]).T  # x, y, steer angle
        self.P = np.diag([.1, .1, .1])
        cov_range = std_range ** 2
        cov_bearing = std_bearing ** 2
        for i in range(0, self.dim_z, 2):
            self.R[i:i] = cov_range
            self.R[i+1:i+1] = cov_bearing
        sim_pos = self.x.copy()
        u = array([1.1, .01])
        plt.figure()
        plt.scatter(land_marks[:, 0], land_marks[:, 1], marker='s', s=60)
        track = []
        for i in range(iteration_num):
            sim_pos = self.x_forward(sim_pos, u, self.dt)  # simulate robot
            track.append(sim_pos)
            if i % step == 0:
                self.predict(u=u)
                if i % ellipse_step == 0:
                    self.plot_covariance_ellipse((self.x[0, 0], self.x[1, 0]), self.P[0:2, 0:2], std=6, facecolor='k',
                                                 alpha=0.3)

                z_total = np.empty((self.dim_z, 1))
                for _id, landmark in enumerate(land_marks):
                    z = self.z_landmark(landmark, sim_pos, std_range, std_bearing)
                    z_total[_id * self.sensor_type:(_id + 1) * self.sensor_type] = z
                z = z_total
                self.ekf_update(z, land_marks)
                if i % ellipse_step == 0:
                    self.plot_covariance_ellipse((self.x[0, 0], self.x[1, 0]), self.P[0:2, 0:2], std=6, facecolor='g',
                                                 alpha=0.8)
        track = np.array(track)
        plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
        plt.axis('equal')
        plt.title("EKF Robot localization")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.show()
        return ekf


_dt = 0.1
landmarks = array([[50, 100], [40, 90], [150, 150], [-150, 200]])
# ekf = EKFLocalization(_dt, std_vel=5.1, std_steer=np.radians(1))
ekf = EKF2(_dt, std_vel=5.1, std_steer=np.radians(1))
ekf.run_localization(landmarks, std_range=0.3, std_bearing=0.1, ellipse_step=2000, iteration_num=20000)
