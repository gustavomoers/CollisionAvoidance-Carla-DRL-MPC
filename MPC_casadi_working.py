#!usr/bin/env python3
# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 22 August 2022


from math import cos, pi, sin
import numpy as np
import casadi as cd
import matplotlib.pyplot as plt

from Utils.CubicSpline.cubic_spline_planner import *



# Parameters
T = 0.1 #sampling time 0.1s
Td = 0.2 #discretization time
N = 10   #prediction horizon

x = cd.MX.sym("x") # x co-ordintate
y = cd.MX.sym("y") # y co-ordinate
theta = cd.MX.sym("theta") # Yaw angle
v= cd.MX.sym("v")  # longitudinal velocity

states = cd.vertcat(x,y,theta,v)
n_states = 4

#control variable
acceleration = cd.MX.sym("acceleration") # acceleration
steering = cd.MX.sym("steering") # steering angle #beta
controls = cd.vertcat(acceleration,steering) 
n_controls = 2  #no of control variables or manipulated variables

#bounds on mv
max_acc = 2 # in m/s^2 ub for acc
min_acc = -1.5 # lb for acc
max_acc_rate = 1.5 # in m/s^3 ub
min_acc_rate = -3 # lb

max_steer = 0.4317
min_steer = -0.4317
max_steer_rate = 0.1380
min_steer_rate = -0.1380

#vehicle parameters
lr = 1.738
lf = 1.105

# weight matrices
Q = np.diag([3.5,3.5,1.9,100])
# Q = np.eye(4)

# R_ = np.diag([500,500])
R_ = np.eye(2)

R = np.diag([1,1])
# R = np.eye(2)


def get_ref_traj(type):
    """ 3 types of trajectories available
    A == A right turn with slow speed
    B == A sinusoidal path with morderate speed
    C == CPG winding track"""
    if type == "A":
        Tt = 81     #total simulation time
        N_steps = int(Tt/T)
        v = np.array([0.0]*N_steps)
        # constructing the reference velocity profile
        #for 5s speed is 0
        n = int(5/T)
        v[:n] = 0
        #for next 45s speed is 3m/s
        n1 = int(45/T + n)
        v[n:n1] = 3
        #for next 4s speed gradualy reduces to zero
        n = int(4/T)+1
        for i in range(n):
            v[n1+i] = v[n1+i-1]-0.75*T
        # for next 4 sec speed is 0
        n1 = int(n1 + n)
        n  = int(3/T +n1)
        v[n1:n] = 0.0
        # for next 12s speed is 6m/s
        n1 = int(12/T + n)
        v[n:n1] = 6
        #for next 10 s the speed delecerates to 0
        n = int(10/T)
        for i in range(n):
            v[n1+i] = v[n1+i-1]-0.6*T
        #for next 1s speed is zero
        n1 = int(n1 + n)
        n  = int(1/T)
        v[n1:n] = 0.0
        # #hence total simulation time is 5+45+4+4+12+10+1 = 81s

        # now based on the refrence velocity we get the x,v,yaw of the vehicle
        x   = np.array([0.0]*N_steps)
        y   = np.array([0.0]*N_steps)
        yaw = np.array([0.0]*N_steps)

        #first 30 secs straight road in x direction
        for i in range(int(35/T)-1):
            # yaw[i+1] = yaw[i]
            x[i+1] = x[i] + v[i]*cos(yaw[i])*T
            y[i+1] = y[i] + v[i]*sin(yaw[i])*T 
        # turining right in 10s 
        for i in range(int(35/T)-1,int(45/T)-1):
            yaw[i+1] = yaw[i] - pi/20*T
            x[i+1] = x[i] + v[i]*cos(yaw[i])*T
            y[i+1] = y[i] + v[i]*sin(yaw[i])*T
        #again straight for next 31 secs in y direction
        for i in range(int(45/T)-1,int(81/T)-1):
            yaw[i+1] = yaw[i]
            # yaw[i] = -pi/2
            x[i+1] = x[i] + v[i]*cos(yaw[i])*T
            y[i+1] = y[i] + v[i]*sin(yaw[i])*T
        return np.array([x,y,yaw,v])
    
    elif type == 'B':
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



#converting Steering and Beta
steering = cd.atan(lr*cd.tan(steering)/(lr+lf))
# ^^ Beta   
# creating an Estimator function
Bicycle_model = cd.vertcat((v*cd.cos(theta+steering)),(v*cd.sin(theta+steering)),(v*cd.sin(steering)/lr),(acceleration))

Estimator = cd.Function("Estimator",[states,controls],[Bicycle_model])

#creating an Integrator for our MPC
intg_options = {"tf":Td,"simplify":True, "number_of_finite_elements":4}
# intg_options.tf = 0.8
# intg_options.simplify = True
# intg_options.number_of_finite_elements = 4

DAE = {"x":states,"p":controls,"ode":Estimator(states,controls)}
# DAE.x = states
# DAE.p = controls
# DAE.ode = Estimator(states,controls)
intg = cd.integrator("intg","rk",DAE,intg_options)
# print(intg)
res = intg(states,controls,[],[],[],[])
States_nxt = res[0]
Estimator = cd.Function("Estimator",[states,controls],[States_nxt])

intg_options1 = {"tf":T,"simplify":True, "number_of_finite_elements":4}
intg1 = cd.integrator("intg","rk",DAE,intg_options1)
# print(intg)
res1 = intg1(states,controls,[],[],[],[])
States_nxt1 = res1[0]
Estimator1 = cd.Function("Estimator1",[states,controls],[States_nxt1])

# sim = Estimator.mapaccum(10)

opti = cd.Opti()

z         = opti.variable(4,N+1)
u         = opti.variable(2,N)
z_initial = opti.parameter(4,1)
u_prev    = opti.parameter(2,1)

z_ref     = opti.parameter(4,N+2)

obj=0
for i in range(N-1):
    obj += (z[:,i+1]-z_ref[:,i]).T @ Q @ (z[:,i+1]-z_ref[:,i])  # states objective
    obj += (u[:,i+1]-u[:,i]).T @ R_ @ (u[:,i+1]-u[:,i]) # change in control variables
    obj += u[:,i].T @ R @ u[:,i] # control variables
    
opti.minimize(obj)

# initial constraint condition
opti.subject_to(z[:,0]==z_initial)
for k in range(N):
    opti.subject_to(z[:,k+1]==Estimator(z[:,k],u[:,k]))

#max and min bounds on acc and steering variable
for i in range(N):
    opti.subject_to(u[0,i] <= max_acc)
    opti.subject_to(min_acc <= u[0,i]) 
    opti.subject_to(min_steer <= u[1,i])
    opti.subject_to(u[1,i] <= max_steer)
    if i !=0:
        opti.subject_to( (u[0,i]-u[0,i-1])/Td <= max_acc_rate)
        opti.subject_to(min_acc_rate <= (u[0,i]-u[0,i-1])/Td )
        opti.subject_to( (u[1,i]-u[1,i-1])/Td <= max_steer_rate)
        opti.subject_to(min_steer_rate <= (u[1,i]-u[1,i-1])/Td )
    else:
        opti.subject_to( (u[0,i]-u_prev[0,0])/Td <= max_acc_rate)
        opti.subject_to(min_acc_rate <= (u[0,i]-u_prev[0,0])/Td )
        opti.subject_to( (u[1,i]-u_prev[1,0])/Td <= max_steer_rate)
        opti.subject_to(min_steer_rate <= (u[1,i]-u_prev[1,0])/Td )

# solver_options = {"print_header":False,"print_iteration": False, "print_time": False, "print_in":False,
#                     "print_out":False}
opts = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
opti.solver('ipopt',opts)

# opti.set_value(z_initial,[5,0,1,0])
# sol = opti.solve()

MPC = opti.to_function("MPC",[z_initial,u_prev,z_ref],[u[:,1]])
# print(MPC)
# MPC loop

z = np.array([0.0,0.0,0.0,0.0])
u_prev = np.array([0.0,0.0])
z_ref = get_ref_traj("A")
control_log = np.array([u_prev])
States_log = np.array([z.T])

for i in range(799):
    # print(z_ref[:,0:12].shape)
    u = MPC(z,u_prev,z_ref[:,0:12])
    print(i)
    u_prev = np.array(u)
    # simulate system
    z = Estimator1(z,u)
    z_ref = np.delete(z_ref,0,axis=-1)

    z=np.array(z)
    control_log = np.append(control_log,u_prev.T,axis=0)
    States_log = np.append(States_log,z.T,axis=0)

    



print(control_log.shape)
print(States_log.shape)
z_ref = get_ref_traj("A")
States_log = States_log.T
control_log = control_log.T
fig , axes = plt.subplots(5,1,figsize=(10,15))
axes[0].plot(list(range(0,799,1)),States_log[3][list(range(0,799,1))])
axes[0].set_title("Velocity tracking")
axes[0].set(ylim=(0,7))
axes[0].plot(list(range(0,799,1)),z_ref[3][list(range(0,799,1))],"--")
axes[0].legend(["actual","reference"])

# accerlation plot
axes[1].plot(list(range(0,799,1)),control_log[0][list(range(0,799,1))])
axes[1].set_title("acceleration")
axes[1].set(ylim=(-2,1.5))

# Steering Angle Beta
axes[2].plot(list(range(0,799,1)),control_log[1][list(range(0,799,1))])
# axes[2].set(ylim=(-20,20))
axes[2].set_title("Steering")

# # Heading angle
# axes[3].plot(list(range(0,799,1)),States_log[2][list(range(0,799,1))])
# axes[3].set_title("Heading angle")

# Position error
position_error = []
for i in range(799):
    position_error.append(np.sqrt((z_ref[0][i]-States_log[0][i])**2 + (z_ref[1][i]-States_log[1][i])**2))
position_error = np.array(position_error)

axes[3].plot(list(range(0,799,1)),position_error[list(range(0,799,1))])
axes[3].set_title("position error")
axes[3].plot(list(range(0,799,1)),np.mean(position_error)*np.ones((799,)),"--")
axes[3].legend(["Position Error","average"])

axes[4].plot(States_log[0][list(range(0,799,1))],States_log[1][list(range(0,799,1))])
axes[4].set_title("x vs y")
axes[4].plot(z_ref[0],z_ref[1],"--")
axes[4].legend(["actual","reference"])
fig.savefig("KONG.png")
print("Mean:",np.mean(position_error))
print("Standard deviation",np.std(position_error))

plt.show()
