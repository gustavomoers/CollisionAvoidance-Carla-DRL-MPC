# Hybrid Collision Avoidance System for Autonomous Vehicles

## Overview
This project develops a hybrid collision avoidance system for autonomous vehicles, combining Deep Reinforcement Learning (DRL) with a Model Predictive Control (MPC) approach. Utilizing the CARLA simulator, the system is designed to navigate scenarios where an ego car approaches a stationary vehicle. Essentialy, the RL agent works as a path planner module and the MPC controls the vehicle.

## Model Description
The project employs Recurrent Proximal Policy Optimization (Recurrent PPO) from Stable Baselines3 for training the reinforcement learning model. The state space for the RL agent includes:
- x_dist: Lateral distance from the parked car
- y_dist: Longitudinal distance from the parked car
- current_speed
- current_acceleration
- current_yaw

The action space consists of five continuous values (x1, y1, x2, y2, desired speed). These values are used to generate a cubic spline curve that defines the path for the MPC controller, which manages vehicle navigation based on the desired speed and path.

## Results
The model demonstrates effective navigation and collision avoidance:
- **Generated Path Visualization:** GIF showing the model's path planning in real-time with red markings to denote the RL-generated path at each timestep.

  ![Model Path Visualization](https://github.com/gustavomoers/CollisionAvoidance-Carla-DRL-MPC/assets/69984472/b0fe0cac-d43e-4742-9cc0-f1b667ffff0d)


- **Performance Metrics:** Graph depicting the reward received per timestep during training, indicating model performance and learning progress.

![Reward vs. Timestep Graph](https://github.com/gustavomoers/CollisionAvoidance-Carla-DRL-MPC/assets/69984472/82b01e0b-80b6-422a-a86d-16896246b7a9)

## Dependencies
- CARLA Simulator
- Stable Baselines3
- Additional dependencies as listed in `requirements.txt`

## How to Run
To start the simulation and train the model:
1. Adjust the parameters in `train.py` and other configuration files as needed.
2. Run `train.py` to commence training and simulation.
3. Monitor outputs and adjust parameters to optimize the training and path generation.

## Contribution
Contributions to enhance the system or adapt it for different scenarios are welcome. Please fork the repository, propose changes via pull requests, or raise issues for discussion.
