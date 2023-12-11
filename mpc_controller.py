import numpy as np
import cvxpy as cp
from flightsim.crazyflie_params import quad_params
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




# def quadrotor_rk4(x, u, Ts):
#     """Discrete-time dynamics: Integration with RK4 method."""
#     f = quadrotor_dynamics
#     k1 = Ts*f(x, u)
#     k2 = Ts*f(x + k1/2, u)
#     k3 = Ts*f(x + k2/2, u)
#     k4 = Ts*f(x + k3, u)
#     x_next = x + 1/6*(k1 + 2*k2 + 2*k3 + k4)
#     return x_next



class MPCController:
    def __init__(self, quad_params, horizon=10):
        # Quadrotor physical parameters.
        self.m = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        # self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        # self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']  # Nm/(rad/s)**2
        self.k_tx = 1
        self.k_ty = 1
        self.k_tz = 1
        self.T = 0.1

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        l = self.arm_length
        gam = self.k_drag / self.k_thrust

        # System matrix A as a 12x12 matrix


        self.A = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, self.g * self.m, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -self.g * self.m, 0, 0, 0, 0, 0],
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
            [0, 0, 0, 0],  # should be -1, -1, -1, -1
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, l / self.Ixx, 0, -l / self.Ixx],
            [-l / self.Iyy, 0, l / self.Iyy, 0],
            [gam / self.Izz, -gam / self.Izz, gam / self.Izz, -gam / self.Izz]])

        # self.B = np.zeros((12, 4))
        #
        # self.B[5][0] = 1 / self.m
        # self.B[7][1] = 1 / self.Ixx
        # self.B[9][2] = 1 / self.Iyy
        # self.B[11][3] = 1 / self.Izz

        # self.Q = np.eye(12)
        #
        # self.Q[0][0] = 5 / 2  # x pos
        # self.Q[1][1] = 5 / 2  # y pos
        # self.Q[2][2] = 100  # z pos
        # # self.Q[3][3] = 500 #x pos
        # # self.Q[4][4] = 500 #x pos
        # # self.Q[5][5] = 500 #x pos
        # self.Q[6][6] = 125
        # self.Q[7][7] = 125
        # self.Q[8][8] = 10
        # self.Q[9][9] = 5
        # self.Q[10][10] = 5
        # self.Q[11][11] = 100
        self.Q = np.eye(12)
        self.Q[0][0] = 100  # x position
        self.Q[1][1] = 100  # y position
        self.Q[2][2] = 500  # z position (often more critical for a quadrotor)
        self.Q[3][3] = 1  # x velocity
        self.Q[4][4] = 1# y velocity
        self.Q[5][5] = 1  # z velocity
        self.Q[6][6] = 10  # roll
        self.Q[7][7] = 10  # pitch
        self.Q[8][8] = 10  # yaw
        self.Q[9][9] = 1  # roll rate
        self.Q[10][10] = 1  # pitch rate
        self.Q[11][11] = 1  # yaw rate

        self.R = np.eye(4)

        self.R[0][0] = 200
        self.R[1][1] = 200
        self.R[2][2] = 200
        self.R[3][3] = 200
        self.horizon = horizon

        # Control input constraints
        self.umin = 0
        self.umax = 5.5

        self.A_d = self.A *self.T + np.identity(12)
        self.B_d = self.B * self.T

    # def quadrotor_dynamics(self,time,state, x, u, flat_outputs):
    #     """Computes the continuous-time dynamics for a quadrotor ẋ=f(x,u).
    #
    #     State is x = [r, v, p, omega], where:
    #     - r ∈R^3 is the position in world frame N
    #     - v ∈R^3 is the linear velocity in world frame N
    #     - p ∈R^3 is the attitude from B->N (MRP)
    #     - omega ∈R^3 is the angular velocity in body frame B
    #     Inputs:
    #       - x(np.ndarray): The system state   [12x1]
    #       - u(np.ndarray): The control inputs [4x1]
    #     Returns:
    #       - x_d(np.ndarray): The time derivative of the state [12x1]
    #     """
    #     # Quadrotor parameters
    #     mass = self.m
    #     L = self.arm_length
    #     kf = self.k_thrust
    #     km = self.k_drag
    #
    #     pos = state['x']
    #     q = state['q']
    #
    #     pos_des = flat_outputs['x']
    #     x_dot_des = flat_outputs['x_dot']
    #     x_ddot_des = flat_outputs['x_ddot']
    #
    #     quat = state['q']
    #
    #     v = state['v']
    #     w = state['w']
    #
    #     v_des = flat_outputs['x_dot']
    #
    #     roll, pitch, yaw = quaternion_to_euler(quat)
    #
    #     R = Rotation.from_quat(state['q']).as_matrix()
    #
    #     roll = np.arcsin(R[2, 1])
    #     pitch = np.arctan2(-R[2, 0], R[2, 2])
    #     yaw = np.arctan2(-R[1, 0], R[1, 1])
    #
    #
    #     qx = q[0]
    #     qy = q[1]
    #     qz = q[2]
    #     qw = q[3]
    #
    #     # Calculating cosines and sines of the current yaw angle
    #     c_psi = np.cos(yaw)
    #     s_psi = np.sin(yaw)
    #
    #     yaw_des = flat_outputs['yaw']
    #
    #     # Calculating the desired pitch angle theta_d
    #     pitch_des = 0
    #
    #     # Calculating the desired roll angle phi_d
    #     roll_des = 0
    #
    #     x0 = np.array([
    #         pos[0], pos[1], pos[2],  # Position
    #         v[0], v[1], v[2],  # Velocity
    #         roll, pitch, yaw,  # Orientation (Euler angles)
    #         w[0], w[1], w[2]  # Angular velocity
    #     ])
    #
    #     x_ref = np.array([
    #         pos_des[0], pos_des[1], pos_des[2],  # Desired position
    #         v_des[0], v_des[1], v_des[2],  # Desired velocity
    #         roll_des, pitch_des, yaw_des,  # Desired orientation (Euler angles)
    #         0, 0, 0  # Desired angular velocity (zero for hover/level flight)
    #     ])
    #
    #     return x_ref
    #
    # def quadrotor_rk4(self,x, u, time, state, flat_outputs, Ts):
    #     """Discrete-time dynamics: Integration with RK4 method."""
    #     f = self.quadrotor_dynamics
    #     k1 = Ts * f(self,time,state,x, u, flat_outputs)
    #     k2 = Ts * f(self,time,state,x+k1/2, u, flat_outputs)
    #     k3 = Ts * f(self,time,state,x + k2/2, u, flat_outputs)
    #     k4 = Ts * f(self,time,state,x+k3, u, flat_outputs)
    #     x_next = x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #     return x_next

    def compute_lqr_gain(self):
        """
                    Compute the LQR gain matrix.

                    A, B: State-space model matrices
                    Q, R: Cost matrices
                    """
        p = solve_continuous_are(self.A, self.B, self.Q, self.R)
        # k = np.linalg.inv(self.R) @ self.B.T @ p

        # k = np.linalg.inv(self.R + self.B.T @ p @ self.B) @ self.B.T @ p @ self.A
        # k = np.linalg.inv(self.R) @ self.B.T @ p
        print("P", p)

        return p

    def discretized_dynamics(self, T):
        A_d = self.A *self.T + np.identity(12)
        B_d = self.B * self.T

        return A_d, B_d

    def compute_control(self, x0, x_ref):
        x = cp.Variable((12, self.horizon + 1))
        u = cp.Variable((4, self.horizon))

        cost = 0
        constraints = [x[:, 0] == x0]
        # constraints = []
        # print("X0", constraints)

        for t in range(self.horizon):
            cost += cp.quad_form(x[:, t] - x_ref, self.Q) + cp.quad_form(u[:, t], self.R)
            # print("cost",cost)
            constraints += [x[:, t + 1] == self.A_d @ x[:, t] + self.B_d @ u[:, t]]
            constraints += [self.umin <= u[:, t], u[:, t] <= self.umax]

        cost += cp.quad_form(x[:, self.horizon] - x_ref, self.Q)  # Terminal cost
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        if prob.status not in ["infeasible", "unbounded"]:
            # Only return the control if a solution is found
            print("U value", u.value)
            return u[:, 0].value
        else:
            print("Solver Status:", prob.status)
            return np.zeros(4)  # or some safe default value

    def update(self, time, state, flat_outputs):
        pos = state['x']
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

        # print("roll", roll)
        # print("pitch", pitch)
        # print("yaw", yaw)

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
        pitch_des = 0  # np.arctan2(
        #     -c_psi * (x_ddot_des[0] + self.k_tx / self.m * x_dot_des[0]) - s_psi * (
        #             x_ddot_des[1] + self.k_ty / self.m * x_dot_des[1]),
        #     self.g - (x_ddot_des[2] + self.k_tz / self.m * x_dot_des[2])
        # )

        # print("pitch desired", pitch_des)

        # Calculating the desired roll angle phi_d
        roll_des = 0  # np.arctan2(
        #     (-s_psi * (x_ddot_des[0] + self.k_tx / self.m * x_ddot_des[0]) + c_psi * (x_ddot_des[1] + self.k_ty / self.m * x_ddot_des[1])) * np.cos(pitch_des),
        #     self.g - (x_ddot_des[2] + self.k_tz / self.m * x_ddot_des[2])
        # )

        # print("roll desired", roll_des)

        #
        # yaw = atan2(2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz);
        # pitch = asin(-2.0 * (qx * qz - qw * qy));
        # roll = atan2(2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz);

        #
        # print("pos", pos)
        # print("roll", roll)
        # print("pitch", pitch)
        # print("yaw", yaw)

        x0 = np.array([
            pos[0], pos[1], pos[2],  # Position
            v[0], v[1], v[2],  # Velocity
            roll, pitch, yaw,  # Orientation (Euler angles)
            w[0], w[1], w[2]  # Angular velocity
        ])

        x_ref = np.array([
            pos_des[0], pos_des[1], pos_des[2],  # Desired position
            v_des[0], v_des[1], v_des[2],  # Desired velocity
            roll_des, pitch_des, yaw_des,  # Desired orientation (Euler angles)
            0, 0, 0  # Desired angular velocity (zero for hover/level flight)
        ])

        x = x0 - x_ref

        u_ctrl = self.compute_control(x0, x_ref)

        print("uctrl", u_ctrl)

        # Calculate control outputs similar to LQR
        l = self.arm_length
        b = self.k_thrust
        d = self.k_drag

        cmd_motor_speeds = (u_ctrl + (self.m * self.g / 4))/ b
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        print("motor speeds", cmd_motor_speeds)

        u_dict = {
            'cmd_motor_speeds': cmd_motor_speeds,
            'cmd_thrust': np.zeros(4),
            'cmd_moment': np.zeros(4),
            'cmd_q': np.zeros(4)
        }

        return u_dict

# Example usage
if __name__ == "__main__":
    mpc = MPCController(quad_params)

    initial_state = {
        "x": [0.0, 0.0, 0.0],
        "v": [0.0, 0.0, 0.0],
        "q": [0.0, -0.5, 0.0, 1.0],
        "w": [0.0, 0.0, 0.0]
    }
    desired_state = {
        "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x_dot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x_ddot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "yaw": 0.0,
        "yaw_dot": 0.0,

    }

    control_output = mpc.update(0, initial_state, desired_state)
    # print("Control Output:", control_output)
