#!/usr/bin/env python3
import numpy as np
from numpy import linalg
import math

import cvxpy

from matplotlib import pyplot as plt

class MPC:
    def __init__(self, x=0, y=0, yaw=0, v=0, delta=0,
                 max_steering_angle=0.61, lf = 1.105 , lr = 1.738, L=2.89, Q=np.eye(4), Qf=np.eye(4),
                 Rd=np.eye(2), len_horizon=6, a_max=1, a_min=-1.5,
                 R = np.eye(2),R_= np.eye(2),Hp = 6,Hc = 4,ts=0.1,td=0.1, a_rate_min=-3,
                 a_rate_max=1.5, steer_rate_max=0.3, v_min=-1, v_max=80, dist = 1.5, time_step = 0.1):

        # States
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

        # Steering angle
        self.delta = delta
        self.max_steering_angle = max_steering_angle

        # Wheel base
        self.lf = lf
        self.lr = lr
        self.L  = lr + lf

        # Control gain
        self.Q = np.array([[  2.5,  0,  0,  0],
                            [  0,  2.5,  0,  0],
                            [  0,  0,  1.1,  0],
                            [  0,  0,  0,  5.5]])
        self.R = R
        self.Rd = Rd
        self.R = R
        self.R_= R_
        self.Qf = np.array([[  3.5,  0,  0,  0],
                    [  0,  3.5,  0,  0],
                    [  0,  0,  1.5,  0],
                    [  0,  0,  0,  3.5]])

        #forward time step
        self.time_step = time_step

        self.len_horizon = len_horizon
        self.Hp = Hp
        self.dt=0.1

        self.v_min = v_min
        self.v_max = v_max
        self.a_max = a_max
        self.a_min = a_min
        self.a_rate_max = a_rate_max
        self.steer_rate = steer_rate_max
        self.dist = dist

        self.prev_idx = 0
        self.send_prev = 0
        self.prev_accelerations = np.array([0.0] * self.Hp)
        self.prev_deltas = np.array([0.0] * self.Hp)
        self.prev_index = 0
    
    def update_state(self, x, y, v, yaw):
        self.x = x
        self.y = y
        self.v = v
        self.yaw = yaw

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def update_speed(self, v):
        self.v = v

    def update_yaw(self, yaw):
        self.yaw = yaw

    @staticmethod
    def bound_angles(theta):
        if theta > np.pi:
            theta = theta - 2*np.pi
        elif theta < -np.pi:
            theta = theta + 2*np.pi

        return theta



    def get_linearized_dynamics(self, yaw, delta, v, dt=0.01):
        # A = np.eye(4)
        # A[0, 2] = np.cos(yaw) * dt
        # A[0, 3] = -v * np.sin(yaw) * dt
        # A[1, 2] = np.sin(yaw) * dt
        # A[1, 3] = v * np.cos(yaw) * dt
        # A[3, 2] = np.tan(delta) * dt / self.L

        # B = np.zeros((4, 2))
        # B[2, 0] = dt
        # B[3, 1] = v * dt / (self.L * np.cos(delta)**2)

        # C = np.zeros((4, 1))
        # C[0, 0] = v * np.sin(yaw) * yaw * dt
        # C[1, 0] = - v * np.cos(yaw) * yaw * dt
        # C[3, 0] = - v * delta * dt / (self.L * np.cos(delta)**2)

        tandelta = math.tan(delta)
        angel = yaw + math.atanh((self.lr*tandelta)/self.L)
        deno1 = np.tan(delta)**2 + 1
        deno2 = (self.lr**2*tandelta**2)/self.L**2 + 1
        deno3 = self.L * np.sqrt(deno2)
        # print(f"angles {yaw}, {angel}, {tandelta}")

        A = np.array([[ 0, 0, np.cos(angel), -v*np.sin(angel)],
                    [ 0, 0, np.sin(angel),  v*np.cos(angel)],
                    [ 0, 0, 0, 0],
                    [ 0, 0, tandelta/deno3, 0]]) * dt
        A = A + np.eye(4)

        B = np.array([[ 0, -(self.lr*v*np.sin(angel)*(deno1))/(self.L*(deno2))],
                    [ 0, (self.lr*v*np.cos(angel)*(deno1))/(self.L*(deno2))],
                    [ 1, 0],
                    [ 0, (v*(deno1))/deno3 - (self.lr**2*v*tandelta**2*(deno1))/(deno3**3)]])
        B *= dt

        C = np.zeros((4, 1))
        C[0, 0] = yaw*v*np.sin(angel) + (delta*self.lr*v*np.sin(angel)*deno1)/(self.L*(deno2))
        C[1, 0] = -yaw*v*np.cos(angel) - (delta*self.lr*v*np.cos(angel)*deno1)/(self.L*(deno2))
        C[2, 0] = 0
        C[3, 0] = -delta*(v*deno1)/(deno3) - (self.lr**2*v*tandelta**2*deno1)/(deno3**3)
        C *= dt

        return A, B, C


    def linear_mpc(self, z_ref, z_initial):
            
            z = cvxpy.Variable((4, self.Hp + 1))
            u = cvxpy.Variable((2, self.Hp))

            cost = 0
            constraints = [z[:, 0] == z_initial.flatten()]

            if z_ref.shape[1] < self.Hp:
                return None, None, None
            for i in range(self.Hp - 1):
                ## Cost
                if i != 0:
                    cost += cvxpy.quad_form(z_ref[:, i] - z[:, i], self.Q)
                    cost += cvxpy.quad_form(u[:, i] - u[:, i-1], self.R)
                else:
                    u_prev = [self.prev_accelerations[0], self.prev_deltas[0]]
                    cost += cvxpy.quad_form(u[:, i] - u_prev, self.R)

                cost += cvxpy.quad_form(u[:, i], self.R)
                # print(f'INPUTS: min delta: {self.prev_deltas[0]:.2f}, yaw: {z_ref[3, i]:.2f}, velocity: {z_ref[2, i]:.2f}')
                # print(f'yaw: {z_ref[3, i]:.2f}')
                # print(f'velocity: {z_ref[2, i]:.2f}')
                # Constraints
                # A, B = self.Kinematic_model(z_ref[3, i], self.prev_deltas[np.min([ i + 1, len(self.prev_deltas) - 1])], self.dt)

                # constraints += [z[:, i+1] == A @ z[:, i] + B @ u[:, i] ]

                A, B, C = self.get_linearized_dynamics(z_ref[3, i], self.prev_deltas[np.min([ i + 1, len(self.prev_deltas) - 1])],
                                                z_ref[2, i], self.dt)

                constraints += [z[:, i+1] == A @ z[:, i] + B @ u[:, i] + C.flatten()]



                # Velocity limits
                constraints += [z[2, i] <= self.v_max]
                constraints += [z[2, i] >= self.v_min]

                # Input limits
                constraints += [self.a_min <= u[0, i]]
                constraints += [u[0, i] <= self.a_max]
                constraints += [u[1, i] <= self.max_steering_angle]
                constraints += [u[1, i] >= -self.max_steering_angle]
                # # Rate of change of input limit
                if i != 0:
                    constraints += [u[0, i] - u[0, i-1]<= self.a_rate_max]
                    constraints += [u[0, i] - u[0, i-1] >= -self.a_rate_max]
                    constraints += [u[1, i] - u[1, i-1] <= self.steer_rate]
                    constraints += [u[1, i] - u[1, i-1]  >= -self.steer_rate]
                    # constraints += [(z[0, i + 1] - z_ref[0, i])*np.sin(z_ref[3,i]) <= self.dist]
                    # constraints += [(z[0, i + 1] - z_ref[0, i])*np.sin(z_ref[3,i]) >= -self.dist]
                    # constraints += [(z[1, i + 1] - z_ref[1, i])*np.cos(z_ref[3,i]) <= self.dist]
                    # constraints += [(z[1, i + 1] - z_ref[1, i])*np.cos(z_ref[3,i]) >= -self.dist]

            # # Terminal cost
            cost += cvxpy.quad_form(z_ref[:, -1] - \
                                    z[:, -1], self.Qf)

            # Quadratic Program
            qp = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
            qp.solve(solver=cvxpy.ECOS, verbose=False)

            # print(qp.status)
            qp_status = str(qp.status)
            if qp_status == "infeasible":
                print(qp_status)

            if qp.status == cvxpy.OPTIMAL or qp.status == cvxpy.OPTIMAL_INACCURATE:
                x = np.array(z.value[0, :]).flatten()
                y = np.array(z.value[1, :]).flatten()
                v = np.array(z.value[2, :]).flatten()
                yaw = np.array(z.value[3, :]).flatten()
                a = np.array(u.value[0, :]).flatten()
                delta = np.array(u.value[1, :]).flatten()
                # print(f'delta: {delta[0]:.2f}, a: {a[0]:.2f}, yaw: {yaw[0]:.2f}, v: {v[0]:.2f}, y: {y[0]:.2f}, x: {x[0]:.2f}')
            else:
                # x, y, v, yaw, a, delta = None, None, None, None, None, None
                a, delta = None, None
            # neg = [-i for i in delta]

            print(f'delta: {delta}')   
            # print(f'neg: {neg}')
            # 
            return a, delta, qp_status

 

    def get_inputs(self, x, y, yaw, v, waypoints):
        self.update_position(x, y)
        self.update_yaw(yaw)
        self.update_speed(v)
        stat = None
        x0 = np.array([[x], [y], [v], [yaw]])

        accelerations, deltas, stat = self.linear_mpc(waypoints, x0)

        # print (accelerations)
        if accelerations is None:
            self.prev_accelerations = self.prev_accelerations
            self.prev_deltas = self.prev_deltas
        else:
            self.prev_accelerations = accelerations
            self.prev_deltas = deltas
        # print(self.prev_accelerations)
        print(self.prev_deltas[0])
        

        return self.prev_accelerations[0], self.prev_deltas[0], stat