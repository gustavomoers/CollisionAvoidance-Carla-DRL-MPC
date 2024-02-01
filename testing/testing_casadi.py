import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Vehicle parameters
L = 2.9  # Wheelbase of the car

# MPC parameters
N = 20  # Control horizon
dt = 0.1  # Time step

# State [x, y, v, psi]
nx = 4
# Control [delta, a]
nu = 2

# CasADi variables
X = ca.MX.sym('X', nx, N + 1)  # States over the horizon
U = ca.MX.sym('U', nu, N)      # Controls over the horizon

# Define the bicycle model
def bicycle_model(x, u, L, dt):
    x_next = x[0] + x[2] * ca.cos(x[3]) * dt
    y_next = x[1] + x[2] * ca.sin(x[3]) * dt
    v_next = x[2] + u[1] * dt
    psi_next = x[3] + x[2] / L * ca.tan(u[0]) * dt
    return ca.vertcat(x_next, y_next, v_next, psi_next)

# Reference path (straight line for simplicity)
ref_x = np.linspace(0, 10, N+1)
ref_y = np.linspace(0, 0, N+1)
ref_v = 2 * np.ones(N+1)  # Constant velocity
ref_psi = np.zeros(N+1)
ref = np.vstack((ref_x, ref_y, ref_v, ref_psi))

# Weights for the objective function
Q = np.diag([1, 1, 0.5, 0.1])
R = np.diag([0.5, 0.1])

# Formulate the NLP
obj = 0
g = []

# Initial state parameters
X0 = ca.MX.sym('X0', nx)
g.append(X[:, 0] - X0)

for k in range(N):
    # State at next time step
    x_next = bicycle_model(X[:, k], U[:, k], L, dt)
    g.append(X[:, k+1] - x_next)

    # Add to the objective
    obj += ca.mtimes([(X[:, k] - ref[:, k]).T, Q, (X[:, k] - ref[:, k])]) + \
           ca.mtimes([U[:, k].T, R, U[:, k]])

# Create an NLP solver
opts = {'ipopt.print_level': 0, 'print_time': 0}
prob = {'f': obj, 'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))), 'g': ca.vertcat(*g), 'p': X0}
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

# Solve the NLP
x0 = np.zeros((nx * (N+1) + nu * N, 1))  # Initial guess
x0_init = np.array([0, 0, 2, 0])  # Initial state [x, y, v, psi]
sol = solver(x0=x0, lbg=0, ubg=0, p=x0_init)
x_opt = sol['x'].full().flatten()

# Extract the optimized states and controls
# Total number of variables (states + controls)
total_vars = nx * (N + 1) + nu * N

# Extract the optimized states and controls
sol_values = sol['x'].full().flatten()

# States are the first (nx * (N + 1)) elements
x_opt = sol_values[:nx * (N + 1)].reshape((nx, N + 1))

# Controls follow the states in the solution array
u_opt = sol_values[nx * (N + 1):total_vars].reshape((nu, N))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(ref_x, ref_y, 'r--', label='Reference path')
plt.plot(x_opt[0, :], x_opt[1, :], label='Optimized path')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.grid(True)
plt.show()