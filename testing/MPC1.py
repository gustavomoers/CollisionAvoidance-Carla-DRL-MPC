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



class MPC:
    """Model predictive controller using non linear vehicle kinematic model"""
    def __init__(self,x=[0],y=[0],yaw=[0],v=[0],
                 steering_angle=0,acceleration=0, v_max=80,v_min =-1,
                 max_steering_angle = 0.9, a_max = 1, a_min = -1.5,a_rate_max=1.5, a_rate_min=-3, steer_rate=0.175,
                 lr=1.738,lf=1.105,Q = np.eye(4),R = np.eye(2),R_= np.eye(2),Hp = 8,Hc = 8,ts=0.1,td=0.2):
        #init states
        self.x  = x
        self.y  = y
        self.yaw= yaw
        self.v  = v

        #init input variables or MV
        # self.delta_f = steering_angle
        # self.a       = acceleration

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
        self.delta_f_max = max_steering_angle
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

        #trajectory
        self.z_ref = self.get_ref_traj("A")
        self.prev_idx = 0
        self.send_prev = 0
        self.prev_accelerations = np.array([0.0] * self.Hp)
        self.prev_deltas = np.array([0.0] * self.Hp)
        self.prev_index = 0
        self.dist = 3.5
        #storing states
        self.states = None
        self.inputs = None



    def get_ref_traj(self,type):
        """ 3 types of trajectories available
        A == A right turn with slow speed
        B == A sinusoidal path with morderate speed
        C == CPG winding track"""
        if type == "A":
            T = 81     #total simulation time
            N_steps = int(T/self.ts)
            v = np.array([0.0]*N_steps)
            # constructing the reference velocity profile
            #for 5s speed is 0
            n = int(5/self.ts)
            v[:n] = 0
            #for next 45s speed is 3m/s
            n1 = int(45/self.ts + n)
            v[n:n1] = 3
            #for next 4s speed gradualy reduces to zero
            n = int(4/self.ts)
            for i in range(n):
                v[n1+i] = v[n1+i-1]-0.75*self.ts
            # for next 4 sec speed is 0
            n1 = int(n1 + n)
            n  = int(3/self.ts +n1)
            v[n1:n] = 0.0
            # for next 12s speed is 6m/s
            n1 = int(12/self.ts + n)
            v[n:n1] = 6
            #for next 10 s the speed delecerates to 0
            n = int(10/self.ts)
            for i in range(n):
                v[n1+i] = v[n1+i-1]-0.6*self.ts
            #for next 1s speed is zero
            n1 = int(n1 + n)
            n  = int(1/self.ts)
            v[n1:n] = 0.0
            # #hence total simulation time is 5+45+4+4+12+10+1 = 81s

            # now based on the refrence velocity we get the x,v,yaw of the vehicle
            x   = np.array([0.0]*N_steps)
            y   = np.array([0.0]*N_steps)
            yaw = np.array([0.0]*N_steps)

            #first 30 secs straight road in x direction
            for i in range(int(30/self.ts)-1):
                yaw[i+1] = yaw[i]
                x[i+1] = x[i] + v[i]*cos(yaw[i])*self.ts
                y[i+1] = y[i] + v[i]*sin(yaw[i])*self.ts 
            # turining right in 10s 
            for i in range(int(30/self.ts)-1,int(40/self.ts)-1):
                yaw[i+1] = yaw[i] + pi/20*self.ts
                x[i+1] = x[i] + v[i]*cos(yaw[i])*self.ts
                y[i+1] = y[i] + v[i]*sin(yaw[i])*self.ts
            #again straight for next 31 secs in y direction
            for i in range(int(40/self.ts)-1,int(81/self.ts)-1):
                yaw[i+1] = yaw[i]
                x[i+1] = x[i] + v[i]*cos(yaw[i])*self.ts
                y[i+1] = y[i] + v[i]*sin(yaw[i])*self.ts
        return np.array([x,y,yaw,v])

    def Kinematic_model(self,yaw,steering_angle=0,dt=0.1):
        """Prediction module for vehicle states"""
        # beta = atan(self.lr*tan(steering_angle)/self.L)
        beta = steering_angle
        A = np.array([[1,0,0,cos(yaw+beta)*dt],
                      [0,1,0,sin(yaw+beta)*dt],
                      [0,0,1,sin(beta)*dt/self.lr],
                      [0,0,0,1]])
        
        B = np.array([[0,0],
                     [0,0],
                     [0,0],
                     [0,1*dt]])
        
        return A ,B

    def MPC(self,z_initial):
        """MPC solver """
        z = cvxpy.Variable((4,self.Hp),"z")
        u = cvxpy.Variable((2,self.Hp),"u")
        z_initial = np.array([self.x,self.y,self.yaw,self.v])
        cost = 0
        constraints = [z[:,0] == z_initial.flatten()]
        # constraints = []
        # self.Hp = min(self.Hp,self.z_ref.shape[1])
        if self.z_ref.shape[1] < self.Hp:
            return None, None
        for i in range(self.Hp-1):
            if i != 0:
                cost += cvxpy.quad_form(self.z_ref[:,i] - z[:,i], self.Q)
                cost += cvxpy.quad_form(u[:,i]-u[:,i-1], self.R)
            else:
                u_prev = [self.prev_deltas[0],self.prev_accelerations[0]]
                cost += cvxpy.quad_form(u[:, i] - u_prev, self.R)
            
            cost += cvxpy.quad_form(u[:,i],self.R_)
            
            #constrains

            A,B = self.Kinematic_model(self.yaw[0],self.prev_deltas[np.min([i+1,len(self.prev_deltas)-1])],self.td)
            constraints += [z[:,i+1] == A @ z[:,i] + B @ u[:,i]]

            #velocity limits
            # constraints += [z[3, i] <= self.v_max]
            # constraints += [z[3, i] >= self.v_min]

            #input limits
            constraints += [self.a_min <= u[1, i]]
            constraints += [u[1, i] <= self.a_max]
            constraints += [u[0, i] <= self.delta_f_max]
            constraints += [u[0, i] >= -self.delta_f_max]

            #rate constraints
            if i != 0:
                constraints += [(u[1, i] - u[1, i-1])/self.td <= self.a_rate_max]
                constraints += [(u[1, i] - u[1, i-1])/self.td >= -self.a_rate_max]
                constraints += [(u[0, i] - u[0, i-1])/self.td <= self.steer_rate]
                constraints += [(u[0, i] - u[0, i-1])/self.td >= -self.steer_rate]
            # else:
            #     constraints += [(u[1, i] - self.prev_accelerations[0])/self.ts  <= self.a_rate_max]
            #     constraints += [(u[1, i] - self.prev_accelerations[0]/self.ts ) >= -self.a_rate_max]
            #     constraints += [(u[0, i] - self.prev_deltas[0])/self.ts <= self.steer_rate]
            #     constraints += [(u[0, i] - self.prev_deltas[0])/self.ts >= -self.steer_rate]
        #Quad program

        qp = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        qp.solve(solver=cvxpy.ECOS_BB, verbose=False)
        print(qp.status)
        if qp.status == cvxpy.OPTIMAL or qp.status == cvxpy.OPTIMAL_INACCURATE:
            x = np.array(z.value[0, :]).flatten()
            y = np.array(z.value[1, :]).flatten()
            v = np.array(z.value[3, :]).flatten()
            yaw = np.array(z.value[2, :]).flatten()
            a = np.array(u.value[1, :]).flatten()
            delta = np.array(u.value[0, :]).flatten()
        else:
            a, delta = None, None

        return a, delta


    def solve(self,x,y,yaw,v):
        """function to call mpc solver"""
        z_intital = np.array([x,y,yaw,v])
        a,delta = self.MPC(z_intital)
        if a is None:
            self.prev_accelerations = self.prev_accelerations[1:]
            self.prev_deltas = self.prev_deltas[1:]
        else:
            self.prev_accelerations = a
            self.prev_deltas = delta
        return self.prev_accelerations[0], self.prev_deltas[0]

    def run_step(self):
        """runs for 1 trajectory"""
        # A,B = self.Kinematic_model(0,0,0)
        # z_initial = A @ np.array([self.x,self.y,self.yaw,self.v]) + B @ np.array([[0],[0]])
        z_initial = [self.x,self.y,self.yaw,self.v]
        self.states = z_initial
        self.inputs = np.array([[0],[0]])
        #step 1
        a,delta = self.solve(z_initial[0],z_initial[1],z_initial[2],z_initial[3])
        for i in range(0,int(81/self.ts)-1):
            print(i)
            print(self.z_ref[2][0]-self.yaw)
            try:
                self.z_ref = np.delete(self.z_ref,0,axis=1)
            except IndexError:
                pass

            A,B = self.Kinematic_model(self.yaw[0],delta,self.ts)
            z = A @ np.array([self.x,self.y,self.yaw,self.v]) + B @ np.array([[delta],[a]])
            self.x = z[0]
            self.y = z[1]
            self.yaw = z[2]
            self.v = z[3]
            a,delta = self.solve(z[0],z[1],z[2],z[3])
            # print(a,delta)
            self.states = np.append(self.states,z,axis=-1)
            self.inputs = np.append(self.inputs,np.array([[delta],[a]]),axis=-1)

        return self.states ,self.inputs



if __name__ == "__main__":


    Q = np.array([[  3.5,  0,  0,  0],
                  [  0,  3.5,  0,  0],
                  [  0,  0,  25,  0],
                  [  0,  0,    0,  80]])

    
    controller = MPC(Q=Q)

    states,inputs = controller.run_step()
    z_ref = controller.get_ref_traj("B")
    s = controller.inputs.shape[1]
    step=1
    fig , axes = plt.subplots(8,1,figsize=(14,85))
    axes[0].plot(list(range(0,s,step)),controller.states[3][list(range(0,s,step))])
    axes[0].set_title("Velocity tracking")
    axes[0].set(ylim=(0,7))
    axes[0].plot(list(range(0,s,step)),z_ref[3][list(range(0,s,step))],"--")
    axes[0].legend(["actual","reference"])
    axes[1].plot(list(range(0,s,step)),controller.inputs[1][list(range(0,s,step))])
    axes[1].set_title("acceleration")
    axes[1].set(ylim=(-2,1.5))
    axes[2].plot(list(range(0,s,step)),controller.inputs[0][list(range(0,s,step))])
    axes[2].set_title("steering angle delta")
    axes[2].set(ylim=(-0.4,0.4))
    axes[3].plot(list(range(0,s,step)),controller.states[2][list(range(0,s,step))])
    axes[3].set_title("Heading angle")
    axes[3].set(ylim=(-pi,pi))
    axes[4].plot(list(range(0,s,step)),controller.states[0][list(range(0,s,step))])
    axes[4].set_title("x vs t")
    axes[5].plot(list(range(0,s,step)),controller.states[1][list(range(0,s,step))])
    axes[5].set_title("y vs t")
    axes[6].plot(controller.states[0],controller.states[1][list(range(0,s,step))])
    axes[6].set_title("x vs y")
    axes[6].plot(z_ref[0],z_ref[1],"--")
    axes[6].legend(["actual","reference"])

    position_error=[]

    for i in range(s):
        position_error.append(sqrt((z_ref[0][i]-states[0][i])**2 + (z_ref[1][i]-states[1][i])**2))
    position_error=np.array(position_error)

    axes[7].plot(list(range(0,s,step)),position_error[list(range(0,s,step))])
    axes[7].set_title("position error")
    print(np.mean(position_error))

    plt.show()
