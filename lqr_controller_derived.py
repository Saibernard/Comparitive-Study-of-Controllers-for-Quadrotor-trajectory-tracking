import numpy as np
import math
from math import atan2, asin
from scipy.linalg import solve_continuous_are
from scipy.spatial.transform import Rotation
from flightsim.crazyflie_params import quad_params


def quaternion_to_euler(quaternion):
    """
                Convert a quaternion to roll, pitch, and yaw (Euler angles) representation.

                Args:
                    quaternion: A numpy array representing the quaternion in the order (w, x, y, z).

                Returns:
                    (roll, pitch, yaw): Euler angles in radians.
                """
    # Extract quaternion components
    x, y, z, w = quaternion

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.pi / 2 if sinp > 0 else -np.pi / 2
    else:
        pitch = np.arcsin(sinp)

    # sinp = np.sqrt(1.0 + 2.0 * (w*y - x*z))
    # cosp = np.sqrt(1.0 - 2.0*(x*x + y*y))
    # pitch = 2*np.arctan2(sinp,cosp) - np.pi/2

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class lqr_controller(object):
    def __init__(self, quad_params):

        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.m = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']  # Nm/(rad/s)**2
        self.k_tx = 1
        self.k_ty = 1
        self.k_tz = 1

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        l = self.arm_length
        gam = self.k_drag/self.k_thrust

        # System matrix A as a 12x12 matrix
        # self.A = np.array([
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, -self.g, 0, 0, 0],
        #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, self.g, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # ])
# works
        # self.A = np.array([
        #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, self.g*self.m, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.A = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, self.g*self.m, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -self.g*self.m, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.B = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0], #should be -1, -1, -1, -1
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, l/self.Ixx, 0, -l/self.Ixx],
            [-l/self.Iyy, 0, l/self.Iyy, 0],
            [gam/self.Izz, -gam/self.Izz, gam/self.Izz, -gam/self.Izz]])

        # self.B = np.zeros((12, 4))
        #
        # self.B[5][0] = 1 / self.m
        # self.B[7][1] = 1 / self.Ixx
        # self.B[9][2] = 1 / self.Iyy
        # self.B[11][3] = 1 / self.Izz

        self.Q = np.eye(12)

        self.Q[0][0] = 5/2 #x pos
        self.Q[1][1] = 5/2 #y pos
        self.Q[2][2] = 100 #z pos
        # self.Q[3][3] = 20 #x pos
        # self.Q[4][4] = 20 #x pos
        # self.Q[5][5] = 20 #x pos
        self.Q[6][6] = 125
        self.Q[7][7] = 125
        self.Q[8][8] = 10
        self.Q[9][9] = 5
        self.Q[10][10] = 5
        self.Q[11][11] = 100

        self.R = np.eye(4)

        self.R[0][0] = 200
        self.R[1][1] = 200
        self.R[2][2] = 200
        self.R[3][3] = 200

        # print("A", self.A)

        # print("B", self.B)

        # print("Q", self.Q)

        # print("R", self.R)

    def compute_lqr_gain(self):
        """
                    Compute the LQR gain matrix.

                    A, B: State-space model matrices
                    Q, R: Cost matrices
                    """
        p = solve_continuous_are(self.A, self.B, self.Q, self.R)
        # k = np.linalg.inv(self.R) @ self.B.T @ p

        # k = np.linalg.inv(self.R + self.B.T @ p @ self.B) @ self.B.T @ p @ self.A
        k = np.linalg.inv(self.R) @ self.B.T @ p

        return k


    def update(self, time, state, flat_outputs):
        k_mat = self.compute_lqr_gain()
        pos_std = 0.005  # Adjust these values based on your simulation requirements
        vel_std = 0.05
        orientation_std = 0.01
        angular_velocity_std = 0.01

        # Add noise to state variables
        pos_noise = np.random.normal(0, pos_std, 1)
        # vel_noise = np.random.normal(0, vel_std, 3)
        # orientation_noise = np.random.normal(0, orientation_std, 4)  # For quaternion
        # angular_velocity_noise = np.random.normal(0, angular_velocity_std, 3)


        # Update state with noise
        # state['x'] += pos_noise
        # state['v'] += vel_noise
        # state['q'] += orientation_noise
        # state['w'] += angular_velocity_noise

        pos = state['x']
        state['x'] += pos_noise
        q = state['q']

        pos_des = flat_outputs['x']
        x_dot_des = flat_outputs['x_dot']
        x_ddot_des = flat_outputs['x_ddot']

        quat = state['q']

        v = state['v']
        w = state['w']

        v_des = flat_outputs['x_dot']

        roll, pitch, yaw = quaternion_to_euler(quat)

        R = Rotation.from_quat(state['q']).as_matrix()

        roll = np.arcsin(R[2, 1])
        pitch = np.arctan2(-R[2, 0], R[2, 2])
        yaw = np.arctan2(-R[1, 0], R[1, 1])

        print("roll", roll)
        print("pitch", pitch)
        print("yaw", yaw)

        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        # R = Rotation.from_quat(q).as_matrix()
        # yaw = atan2(R[1, 0], R[0, 0])
        # pitch = atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        # roll = atan2(R[2, 1], R[2, 2])

        # https://iopscience.iop.org/article/10.1088/1757-899X/260/1/012026/pdf
        # roll_des = atan2( , self.g - (x_ddot_des[2] + ()*x_dot_des[2]))


        # Calculating cosines and sines of the current yaw angle
        c_psi = np.cos(yaw)
        s_psi = np.sin(yaw)

        yaw_des = flat_outputs['yaw']

        # Calculating the desired pitch angle theta_d
        pitch_des = 0 #np.arctan2(
        #     -c_psi * (x_ddot_des[0] + self.k_tx / self.m * x_dot_des[0]) - s_psi * (
        #             x_ddot_des[1] + self.k_ty / self.m * x_dot_des[1]),
        #     self.g - (x_ddot_des[2] + self.k_tz / self.m * x_dot_des[2])
        # )

        print("pitch desired", pitch_des)

        # Calculating the desired roll angle phi_d
        roll_des = 0 #np.arctan2(
        #     (-s_psi * (x_ddot_des[0] + self.k_tx / self.m * x_ddot_des[0]) + c_psi * (x_ddot_des[1] + self.k_ty / self.m * x_ddot_des[1])) * np.cos(pitch_des),
        #     self.g - (x_ddot_des[2] + self.k_tz / self.m * x_ddot_des[2])
        # )

        print("roll desired", roll_des)

        #
        # yaw = atan2(2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz);
        # pitch = asin(-2.0 * (qx * qz - qw * qy));
        # roll = atan2(2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz);

        #
        # print("pos", pos)
        # print("roll", roll)
        # print("pitch", pitch)
        # print("yaw", yaw)

        x = -np.array([pos_des[0] - pos[0], pos_des[1]-pos[1], pos_des[2]-pos[2],
                      v_des[0] - v[0], v_des[1] - v[1], v_des[2] - v[2],
                      roll_des-roll, pitch_des-pitch, yaw_des-yaw,
                      -w[0], -w[1], -w[2]])

        # x = -np.array([pos_des[0] - pos[0], v_des[0] - v[0], pos_des[1]-pos[1],
        #               v_des[1] - v[1], pos_des[2]-pos[2], v_des[2] - v[2],
        #               (roll_des - roll), -w[0],  (pitch_des-pitch),
        #               -w[1], (yaw_des-yaw), -w[2]])

        # x = np.array([pos_des[0] - pos[0], pos_des[1] - pos[1], pos_des[2] - pos[2],
        #                roll_des - roll, pitch_des - pitch,  yaw_des - yaw,
        #               v_des[0] - v[0], v_des[1] - v[1],  v_des[2] - v[2],
        #               -w[0], -w[1],  -w[2]])

        print("pitch error", pitch_des - pitch)
        print("roll eroor", roll_des - roll)
        print("yaw error", yaw_des-yaw)
        print("yaw rate", w[2])

        print(x)
        print(k_mat)


        # x = np.array([pos[0], pos[1], pos[2],
        #               roll, pitch, yaw,
        #               v[0], v[1], v[2],
        #               w[0], w[1], w[2]])

        # print("x", np.shape(x))
        # print("K", np.shape(k))
        moment_thrust = - k_mat @ x
        print("thrust", moment_thrust)
        u_ctrl = np.array([1788,1788,1788,1788])

        u1 = moment_thrust[0]
        u2 = moment_thrust[1]
        u3 = moment_thrust[2]
        u4 = moment_thrust[3]

        b = self.k_thrust
        d = self.k_drag

        # print(u1, u2, u3, u4, b, d)
        # print((u1 / (4*b)))
        # print((u2 / (2 *b)))
        # print((u4/(4*d)))
        # print((u1 / (4*b)) - (u2 / (2 *b)) - (u4/(4*d)))

        l = self.arm_length

        # u_ctrl[0] = np.sqrt(np.abs((u1/(4*b)) + (u3/(2*b)) + (u4/(4*d)))) * np.sign((u1/(4*b)) + (u3/(2*b)) + (u4/(4*d))) + np.sqrt(self.m *self.g/(4*b))
        # u_ctrl[1] = np.sqrt(np.abs((u1/(4*b)) - (u2/(2*b)) - (u4/(4*d)))) * np.sign((u1/(4*b)) - (u2/(2*b)) - (u4/(4*d))) + np.sqrt(self.m *self.g/(4*b))
        # u_ctrl[2] = np.sqrt(np.abs((u1/(4 * b)) - (u3/(2*b)) + (u4/(4*d)))) * np.sign((u1/(4*b)) - (u3/(2*b)) + (u4/(4*d))) + np.sqrt(self.m *self.g/(4*b))
        # u_ctrl[3] = np.sqrt(np.abs((u1/(4 * b)) + (u2/(2*b)) - (u4/(4*d)))) * np.sign((u1/(4*b)) + (u2/(2*b)) - (u4/(4*d))) + np.sqrt(self.m *self.g/(4*b))

        # u_ctrl[0] = (np.sqrt(np.abs((-(l*d*u1) + (2*d*u3) + (b*l*u4))/(4*b*l*d))) * np.sign((-(l*d*u1) + (2*d*u3) + (b*l*u4))/(4*b*l*d)) + np.sqrt(self.m *self.g/(4*b)))
        # u_ctrl[1] = (np.sqrt(np.abs((-(l*d*u1) - (2*d*u2) - (b*l*u4))/(4*b*l*d))) * np.sign((-(l*d*u1) - (2*d*u2) - (b*l*u4))/(4*b*l*d)) + np.sqrt(self.m *self.g/(4*b)))
        # u_ctrl[2] = (np.sqrt(np.abs((-(l*d*u1) - (2*d*u3) + (b*l*u4))/(4*b*l*d))) * np.sign((-(l*d*u1) - (2*d*u3) + (b*l*u4))/(4*b*l*d)) + np.sqrt(self.m *self.g/(4*b)))
        # u_ctrl[3] = (np.sqrt(np.abs((-(l*d*u1) + (2*d*u2) - (b*l*u4))/(4*b*l*d))) * np.sign((-(l*d*u1) + (2*d*u2) - (b*l*u4))/(4*b*l*d)) + np.sqrt(self.m *self.g/(4*b)))

        # big = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
        #                 [-l * b, 0, l * b, 0],
        #                 [0, -l*b, 0, l*b],
        #                 [-d, d, -d, d]])
        # u = np.linalg.inv(big) @ moment_thrust
        # for i in range(4):
        #     u_ctrl[i] = np.sqrt(np.abs(u[i])) * np.sign(u[i]) #+ np.sqrt(self.m * self.g/(4*b))

        # u_ctrl = np.array([1788,1788,1788,1788])
        # u_ctrl[2] = 2000
        # # u_ctrl[1] = 178
        # u_ctrl = np.array([2496.6398276,  384.48180258, 405.72876128, 2823.19287719])

        cmd_motor_speeds = (moment_thrust + (self.m*self.g/4)) / (self.k_thrust)
        u_ctrl = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        u_dict = {'cmd_motor_speeds':u_ctrl,
                  'cmd_thrust':np.zeros(4),
                  'cmd_moment':np.zeros(4),
                  'cmd_q':np.zeros(4)
                  }

        print("control speed", u_ctrl)
        print("state_des", flat_outputs)


        return u_dict

if __name__ == "__main__":

    lqr = lqr_controller(quad_params)
    initial_state = {
        "x": [0.0, 0.0, 0.0],
        "v": [0.0, 0.0, 0.0],
        "q": [0.0, -0.2, 0.0, 1.0],
        "w": [0.0, 0.0, 0.0]
    }
    desired_state = {
        "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x_dot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x_ddot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "yaw": 0.0,
        "yaw_dot": 0.0,

    }
    u = lqr.update(1, initial_state, desired_state)

    print("u", u)
    print(quad_params)

    print("S", solve_continuous_are(lqr.A, lqr.B, lqr.Q, lqr.R))
    s = solve_continuous_are(lqr.A, lqr.B, lqr.Q, lqr.R)
    k = np.linalg.inv(lqr.R + lqr.B.T @ s @ lqr.B) @ lqr.B.T @ s @ lqr.A
    # k = np.linalg.inv(lqr.R + lqr.B.T @ s @ lqr.B) @ lqr.B.T @ s @ lqr.A

    print("k", np.linalg.inv(lqr.R + lqr.B.T @ s @ lqr.B))
