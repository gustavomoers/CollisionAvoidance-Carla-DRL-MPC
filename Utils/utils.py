import os
import cv2
import pygame
import math
import numpy as np
import carla
import re



def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.
        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def correct_yaw(x):
    return(((x%360) + 360) % 360)

def create_folders(folder_names):
    for directory in folder_names:
        if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)



def wrap_angle(angle_in_degree):
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while (angle_in_rad > np.pi):
        angle_in_rad -= 2 * np.pi
    while (angle_in_rad <= -np.pi):
        angle_in_rad += 2 * np.pi
    return angle_in_rad



def get_vehicle_wheelbases(wheels, center_of_mass):
    front_left_wheel = wheels[0]
    front_right_wheel = wheels[1]
    back_left_wheel = wheels[2]
    back_right_wheel = wheels[3]
    front_x = (front_left_wheel.position.x + front_right_wheel.position.x) / 2.0
    front_y = (front_left_wheel.position.y + front_right_wheel.position.y) / 2.0
    front_z = (front_left_wheel.position.z + front_right_wheel.position.z) / 2.0
    back_x = (back_left_wheel.position.x + back_right_wheel.position.x) / 2.0
    back_y = (back_left_wheel.position.y + back_right_wheel.position.y) / 2.0
    back_z = (back_left_wheel.position.z + back_right_wheel.position.z) / 2.0
    l = np.sqrt( (front_x - back_x)**2 + (front_y - back_y)**2 + (front_z - back_z)**2  ) / 100.0
    # print(f"center of mass : {center_of_mass.x}, {center_of_mass.y}, {center_of_mass.z} wheelbase {l}")
    # return center_of_mass.x , l - center_of_mass.x, l
    return l - center_of_mass.x, center_of_mass.x, l


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name




def next_weather(self, reverse=False):
    self._weather_index += -1 if reverse else 1
    self._weather_index %= len(self._weather_presets)
    preset = self._weather_presets[self._weather_index]
    self.hud.notification('Weather: %s' % preset[1])
    self.player.get_world().set_weather(preset[0])


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]