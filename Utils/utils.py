import os
import cv2
import pygame
import math
import numpy as np
import carla
import re

def process_img(image, dim_x=128, dim_y=128):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    # scale_percent = 25
    # width = int(array.shape[1] * scale_percent/100)
    # height = int(array.shape[0] * scale_percent/100)

    # dim = (width, height)
    dim = (dim_x, dim_y)  # set same dim for now
    resized_img = cv2.resize(array, dim, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    scaledImg = img_gray/255.

    # normalize
    mean, std = 0.5, 0.5
    normalizedImg = (scaledImg - mean) / std

    return normalizedImg

def draw_image_2(self, image):
    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((self.im_height, self.im_width, 4))
    i3 = i2[:, :, :3]
    
    cv2.imshow("", i3)
    cv2.waitKey(1)


def manual_control(self):

    pygame.init() 

    size = (640, 480)
    pygame.display.set_caption("CARLA Manual Control")
    screen = pygame.display.set_mode(size)


    control = carla.VehicleControl()
    clock = pygame.time.Clock()
    done = False

    while not done:

        keys = pygame.key.get_pressed() 

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            control.throttle = min(control.throttle + 0.05, 1.0)
        else:
            control.throttle = 0.0

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            control.brake = min(control.brake + 0.2, 1.0)
        else:
            control.brake = 0.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            control.steer = max(control.steer - 0.05, -1.0)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            control.steer = min(control.steer + 0.05, 1.0)
        else:
            control.steer = 0.0

        control.hand_brake = keys[pygame.K_SPACE]

        # Apply the control to the ego vehicle and tick the simulation
        self.vehicle.apply_control(control)
        self.world.tick()

        # Update the display and check for the quit event
        pygame.display.flip()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Sleep to ensure consistent loop timing
        clock.tick(60)


def process_img2(self, image,  dim_x=128, dim_y=128):

    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((self.im_height, self.im_width, 4))
    i3 = i2[:, :, :3]
    # cv2.imwrite(f'F:/CollisionAvoidance-Carla-DRL-MPC/_out/ground/{self.global_t}.png', i3)

    img_gray = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(f'./_out/test/gray{self.global_t}.png', img_gray)

    dim = (dim_x, dim_y)  # set same dim for now
    resized_img = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)
    # cv2.imwrite(f'F:/CollisionAvoidance-Carla-DRL-MPC/_out/resized/{self.global_t}.png', resized_img)
    
    scaledImg = resized_img/255.

    # normalize
    mean, std = 0.5, 0.5
    normalizedImg = (scaledImg - mean) / std


    xx = np.matrix(resized_img)
    # print (xx.max())
    # print (xx.min())
    # print(scaledImg)

    shap = xx[:, :, np.newaxis]

    return np.array(shap)
    

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

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



def draw_waypoints(world, waypoints, z=0.5, color=(255,0,0)): # from carla/agents/tools/misc.py
    #  """
    # Draw a list of waypoints at a certain height given in z.

    # :param world: carla.world object
    # :param waypoints: list or iterable container with the waypoints to draw
    # :param z: height in meters
    # :return:
    # """
    color = carla.Color(r=color[0],g=color[1],b=color[2],a=255)
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z)
        # angle = math.radians(t.rotation.yaw)
        # end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_point(begin, size=0.05, color=color, life_time=0.1)



def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]



def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name