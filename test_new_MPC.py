#!usr/bin/env python3
# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 28 july 2022


import matplotlib.pyplot as plt
import numpy as np
import cvxpy
from math import *
from Utils.CubicSpline.cubic_spline_planner import *



class MPC:
    """Model predictive controller using non linear vehicle kinematic model"""
    def __init__(self,x=[0],y=[0],yaw=[0],v=[0],
                 steering_angle=0,acceleration=0, v_max=80,v_min =-1,
                 max_steering_angle = 0.61, a_max = 1, a_min = -1.5,a_rate_max=1.5, a_rate_min=-3, steer_rate=0.3,
                 lr=1.738,lf=1.105,R = np.eye(2),R_= np.eye(2),Hp = 6,Hc = 4,ts=0.1,td=0.1):
        #init states
        self.x  = x
        self.y  = y
        self.yaw= yaw
        self.v  = v
        self.x0 = np.array([[-18.842267990112305], [-224.22581481933594], [13.2285009239462026], [1.5481880865109414]])

        #init input variables or MV
        # self.delta_f = steering_angle
        # self.a       = acceleration
        self.Q = np.array([[  2.5,  0,  0,  0],
                            [  0,  2.5,  0,  0],
                            [  0,  0,  1.1,  0],
                            [  0,  0,  0,  5.5]])

        # Terminal Cost
        self.Qf = np.array([[  3.5,  0,  0,  0],
                    [  0,  3.5,  0,  0],
                    [  0,  0,  1.5,  0],
                    [  0,  0,  0,  3.5]])
        #init vehicle parameters
        self.lr = lr
        self.lf = lf
        self.L  = lr + lf

        #init weight matrices
        self.Q = Q
        self.R = R
        self.R_= R_

        #constraints
        self.v_max = v_max
        self.v_min = v_min
        self.max_steering_angle = max_steering_angle
        self.a_max = a_max
        self.a_min = a_min
        self.steer_rate = steer_rate
        self.a_rate_max = a_rate_max
        self.a_rate_min = a_rate_min

        #horizon and time
        self.Hp = Hp
        self.Hc = Hc
        self.ts = ts
        self.td = td
        self.dt = 0.1

        #trajectory
        self.z_ref = self.get_ref_traj()
        self.prev_idx = 0
        self.send_prev = 0
        self.prev_accelerations = [1.99999931, 1.94880953, 1.4488118,  0.94881729, 0. , 0.       ]
        self.prev_deltas = [-0.24681479, -0.40674397, -0.24680603, -0.12345067,  0., 0.        ]
        self.prev_index = 0
        self.dist = 3.5
        #storing states
        self.states = None
        self.inputs = None

    def get_ref_traj(self):
        x = np.array([-17, -19.92, -21, -25, -20, -17])
        y =np.array([-223, -220.06, -200.059, -170.059, -160, -150])
        ds = 1  # [m] distance of each interpolated points
        sp = CubicSpline2D(x, y)
        s = np.arange(0, sp.s[-1], ds)
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))
        nw_wp = []
        for i in range(len(rx)):
            nw_wp.append([rx[i], ry[i], 10, ryaw[i]])
        z_ref = np.array(nw_wp).T

        return z_ref


    def Kinematic_model(self,yaw,steering_angle=0,dt=0.1):
        """Prediction module for vehicle states"""
        beta = atan(self.lr*np.tan(steering_angle)/self.L)
        # beta = steering_angle
        A = np.array([[1,0,np.cos(yaw+beta)*dt,0],
                        [0,1,np.sin(yaw+beta)*dt,0],
                        [0,0,1,0],
                        [0,0,0,1]])
            
        B = np.array([[0,0],
                        [0,0],
                        [dt,0],
                        [0,np.sin(beta)*dt/(self.lr)]])
        
        return A, B
    

    def get_linearized_dynamics(self, yaw, delta, v, dt=0.01):

        # A = np.eye(4)
        # A[0, 0] = 0
        # A[0, 1] = 0
        # A[0, 2] = np.cos(yaw + math.atanh((self.lr*math.tan(delta))/self.L)) * dt
        # A[0, 3] = -v * np.sin(yaw + math.atanh((self.lr*math.tan(delta))/self.L)) * dt
        
        # A[1, 0] = 0
        # A[1, 1] = 0
        # A[1, 2] = np.sin(yaw + math.atanh((self.lr*math.tan(delta))/self.L)) * dt
        # A[1, 3] = v * np.cos(yaw + math.atanh((self.lr*math.tan(delta))/self.L)) * dt


        # A[2, 0] = 0
        # A[2, 1] = 0
        # A[2, 2] = 0
        # A[2, 3] = 0

        # A[3, 0] = 0
        # A[3, 1] = 0
        # A[3, 2] = math.tan(delta)/(self.L*np.sqrt((self.lr**2*math.tan(delta)**2)/self.L**2 + 1)) * dt
        # A[3, 3] = 0


        # B = np.zeros((4, 2))
        # B[0, 1] =  -(self.lr*v*np.sin(yaw + math.atan((self.lr*math.tan(delta))/self.L))*(math.tan(delta)**2 + 1))/(self.L*((self.lr**2*math.tan(delta)**2)/self.L**2 + 1)) * dt
        # B[1, 1] = (self.lr*v*np.cos(yaw + math.atan((self.lr*math.tan(delta))/self.L))*(math.tan(delta)**2 + 1))/(self.L*((self.lr**2*math.tan(delta)**2)/self.L**2 + 1))* dt
        # B[2, 0] = 1* dt
        # B[3, 1] = (v*(math.tan(delta)**2 + 1))/(self.L*((self.lr**2*math.tan(delta)**2)/self.L**2 + 1)**(1/2)) - (self.lr**2*v*math.tan(delta)**2*(math.tan(delta)**2 + 1))/(self.L**3*((self.lr**2*math.tan(delta)**2)/self.L**2 + 1)**(3/2))* dt

  
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
        # # print(f"angles {yaw}, {angel}, {tandelta}")

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

        

    def MPC1(self, z_ref, z_initial, Hp=6, ts=0.1, L=2.843, a_min=-1.5, a_max=1.0, 
        delta_f_max=0.9, a_rate_max=1.5, steer_rate=0.175):
            
            z = cvxpy.Variable((4, self.Hp + 1))
            u = cvxpy.Variable((2, self.Hp))

            cost = 0
            constraints = [z[:, 0] == z_initial.flatten()]

            if self.z_ref.shape[1] < self.Hp:
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

                A, B, C = self.get_linearized_dynamics(z_ref[3, i], self.prev_deltas[0],
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
            # print(f'delta: {delta}')
            
            return a, delta, qp_status


    def solve(self, z_intital, z_ref):
            # """function to call mpc solver"""
            # z_intital = np.array([x,y,yaw,v])
            a,delta, st = self.MPC1(z_ref,z_intital)
            if a is None:
                self.prev_accelerations = self.prev_accelerations[1:]
                self.prev_deltas = self.prev_deltas[1:]
            else:
                self.prev_accelerations = a
                self.prev_deltas = delta
            # print(self.prev_deltas)
            return self.prev_accelerations[0], self.prev_deltas[0]

    def run_step(self):
        a, delta = self.solve(self.x0, self.z_ref)
        self.states = [self.x0]
        self.inputs = []
        for i in range(0,self.z_ref.shape[1]-self.Hp):
            self.z_ref = np.delete(self.z_ref,0,axis=1)
            # print(self.z_ref)
            A,B = self.Kinematic_model(self.x0[3][0],delta,self.ts)
            z = A @ self.x0 + B @ np.array([[a],[delta]])
            print(f'outputs: {z}')
            self.x0=z
            self.states.append(self.x0)
            # print(self.x0)
            a,delta = self.solve(self.x0, self.z_ref)
            # print(delta)
            self.inputs.append([delta, a])
        
        return self.states ,self.inputs



if __name__ == "__main__":


    Q = np.array([[  3.5,  0,  0,  0],
                  [  0,  3.5,  0,  0],
                  [  0,  0,  25,  0],
                  [  0,  0,    0,  80]])

    
    controller = MPC()

    states,inputs = controller.run_step()


    states1 = np.array(states).T
    z_ref = controller.get_ref_traj()
    # plt.subplots(1)
    # plt.plot(states1[0][0], states1[0][1], "xb", label="actual")
    # plt.plot(z_ref[0], z_ref[1], "--", label="reference")

    # plt.grid(True)
    # plt.axis("equal")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.show()


    xxxx = states1[0][1]
    step = 1
    inp = np.array(inputs).T


    fig , axes = plt.subplots(8,1,figsize=(14,85))
    axes[0].plot(list(range(0,xxxx.shape[0],step)), states1[0][2])
    axes[0].set_title("Velocity tracking")
    axes[0].set(ylim=(0,20))
    axes[0].plot(list(range(0,xxxx.shape[0],step)), z_ref[2][:xxxx.shape[0]],"--")
    axes[0].legend(["actual","reference"])

    axes[1].plot(list(range(0,inp[1].shape[0],step)), inp[1])
    axes[1].set_title("acceleration")
    axes[1].set(ylim=(-2,1.5))
    

    axes[2].plot(list(range(0,inp[0].shape[0],step)), inp[0])
    axes[2].set_title("steering angle delta")
    axes[2].set(ylim=(-0.8,0.8))

    axes[3].plot(list(range(0,xxxx.shape[0],step)), states1[0][3])
    axes[3].set_title("Heading angle")
    axes[3].set(ylim=(-pi,pi))

    axes[4].plot(list(range(0,xxxx.shape[0],step)), states1[0][0])
    axes[4].set_title("x vs t")

    axes[5].plot(list(range(0,xxxx.shape[0],step)), states1[0][1])
    axes[5].set_title("y vs t")

    axes[6].plot(states1[0][0] , states1[0][1])
    axes[6].set_title("x vs y")
    axes[6].plot(z_ref[0],z_ref[1],"--")
    axes[6].legend(["actual","reference"])



    position_error=[]

    for i in list(range(0,xxxx.shape[0],step)):
        position_error.append(sqrt((z_ref[0][i]-states1[0][0][1])**2 + (z_ref[1][i]-states1[0][1][i])**2))
    position_error=np.array(position_error)

    axes[7].plot(list(range(0,xxxx.shape[0],step)), position_error[list(range(0,xxxx.shape[0],step))])
    axes[7].set_title("position error")
    print(np.mean(position_error))

    plt.show()
