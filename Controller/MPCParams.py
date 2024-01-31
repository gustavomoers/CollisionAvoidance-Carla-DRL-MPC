import numpy as np

class MPCParams:
    # State Cost
    # Q = np.eye(4)
    Q = np.array([[  2.5,  0,  0,  0],
                  [  0,  2.5,  0,  0],
                  [  0,  0,  1.1,  0],
                  [  0,  0,  0,  5.5]])

    # Terminal Cost
    Qf = np.array([[  3.5,  0,  0,  0],
                  [  0,  3.5,  0,  0],
                  [  0,  0,  1.5,  0],
                  [  0,  0,  0,  3.5]])

    # Control Cost 1) acceleration 2) steer rate
    R = np.eye(2)

    dist = 3.5

    # State change cost
    Rd = np.array([[1, 0],
                   [0 ,1]])

    # Horizon
    len_horizon = 10

    # Constrains
    max_steering_angle = 1

    a_max = 1

    a_min = -1.5
    
    a_rate_max = 1.5
    
    steer_rate_max = 0.3
    
    v_min = -1
    
    v_max = 100

