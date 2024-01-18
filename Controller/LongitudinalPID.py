#!/usr/bin/env python3
import numpy as np

class LongitudinalPID:
    """
    PID controller for longitudinal control
    """
    def __init__(self, v=0, L=3, Kp=1, Kd=0.01, Ki=0.01,
                 integrator_min=None, integrator_max=None):
        # States
        self.v = v
        self.prev_error = 0
        self.sum_error = 0

        # Wheel base
        self.L = L

        # Control gain
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integrator_min = integrator_min
        self.integrator_max = integrator_max

    def update_speed(self, v):
        self.v = v

    def get_throttle_input(self, v, dt, target_speed):
        self.update_speed(v)

        error = target_speed - self.v
        self.sum_error += error * dt
        if self.integrator_min is not None:
            self.sum_error = np.fmax(self.sum_error,
                                     self.integrator_min)
        if self.integrator_max is not None:
            self.sum_error = np.fmin(self.sum_error,
                                     self.integrator_max)

        throttle = self.Kp * error + \
            self.Ki * self.sum_error + \
            self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        return throttle
