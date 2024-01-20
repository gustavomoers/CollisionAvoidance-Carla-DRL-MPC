#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""
import numpy as np
from matplotlib import pyplot as plt
from Controller.MPC import MPC
from Controller.LongitudinalPID import LongitudinalPID
from Controller.MPCParams import MPCParams

class Controller(object):
    def __init__(self, waypoints = None, lf = 1.5, lr = 1.5, wheelbase = 2.89, planning_horizon = 10, time_step = 0.1):
        self._lookahead_distance = 3.0
        self._lookahead_time = 1.0
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_frame = 0
        self._current_timestamp = 0
        self._start_control_loop = False
        self._set_throttle = 0
        self._set_brake = 0
        self._set_steer = 0
        self._waypoints = waypoints
        self.stat = None
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self.controller = MPC(  x = self._current_x, y = self._current_y, yaw = self._current_yaw, v = self._current_speed, delta = 0,
                                lf = lf, lr = lr ,L = wheelbase, Q = MPCParams.Q, R = MPCParams.R, Qf = MPCParams.Qf, Rd = MPCParams.Rd, len_horizon = planning_horizon,
                                dist = MPCParams.dist, max_steering_angle = MPCParams.max_steering_angle, steer_rate_max = MPCParams.steer_rate_max, a_max = MPCParams.a_max, a_min = MPCParams.a_min, a_rate_max = MPCParams.a_rate_max, v_min = MPCParams.v_min, v_max = MPCParams.v_max, time_step=time_step)

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed
        self._current_timestamp = timestamp
        self._current_frame = frame
        self.controller.update_position(x, y)
        self.controller.update_speed(speed)
        self.controller.update_yaw(yaw)
        if self._current_frame:
            self._start_control_loop = True
        return self._start_control_loop

    def get_lookahead_index(self, lookahead_distance):
        min_idx = 0
        min_dist = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        total_dist = min_dist
        lookahead_idx = min_idx
        for i in range(min_idx + 1, len(self._waypoints)):
            if total_dist >= lookahead_distance:
                break
            total_dist += np.linalg.norm(np.array([
                self._waypoints[i][0] - self._waypoints[i-1][0],
                self._waypoints[i][1] - self._waypoints[i-1][1]]))
            lookahead_idx = i
        return lookahead_idx

    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        self._desired_speed = self._waypoints[min_idx][2]

    def smooth_yaw(self, yaws):
        for i in range(len(yaws) - 1):
            dyaw = yaws[i+1] - yaws[i]

            while dyaw >= np.pi/2.0:
                yaws[i+1] -= 2.0 * np.pi
                dyaw = yaws[i+1] - yaws[i]

            while dyaw <= -np.pi/2.0:
                yaws[i+1] += 2.0 * np.pi
                dyaw = yaws[i+1] - yaws[i]

        return yaws

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        throttle_output = 0
        steer_output = 0
        brake_output = 0

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            acceleration, steer_output, self.stat = \
                self.controller.get_inputs(x, y, yaw, v, np.array(self._waypoints).T)

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
        if acceleration > 0:
            # throttle_output = np.tanh(acceleration)
            # throttle_output = max(0.0, min(1.0, throttle_output))
            throttle_output = acceleration / MPCParams.a_max + 0.3
            brake_output = 0.0
        else:
            throttle_output = 0.0
            brake_output = acceleration / MPCParams.a_min  
        # throttle_output = acceleration / MPCParams.a_max + 0.3
        # print(f"Control input , throttle : {throttle_output}, steer outout : {steer_output}, brake : {brake_output}, acceleration : {acceleration}")
        self.set_throttle(throttle_output)  # in percent (0 to 1)
        self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
        self.set_brake(brake_output)        # in percent (0 to 1)