import pygame
import random
import carla
from Utils.synch_mode import CarlaSyncMode
import Controller.PIDController as PIDController
import Controller.MPCController as MPCController
import time
from Utils.utils import *
import math
import gym
import gymnasium as gym
from gymnasium import spaces


class World(gym.Env):
    def __init__(self, client, carla_world, hud, args):
        self.world = carla_world
        self.client = client
        self.actor_role_name = args.rolename
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        # self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = "vehicle.*"
        self._gamma = args.gamma
        self.args = args
        self.recording_start = 0
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        print(self.desired_speed)
        self.planning_horizon = args.planning_horizon
        self.time_step = args.time_step
        self.control_mode = args.control_mode
        self.controller = None
        self.control_count = 0.0
        self.random_spawn = 0
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.im_width = 640
        self.im_height = 480
        self.episode_start = 0
        self.visuals = False
        




        ## RL STABLE BASELINES
        # self.action_space = spaces.Box(low=-1, high=1,shape=(2,),dtype=np.uint8)



        if self.visuals:
            self._initiate_visuals()
        
        self.global_t = 0 # global timestep
        
    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprints = self.world.get_blueprint_library().filter(self._actor_filter)
        blueprint = None
        for blueprint_candidates in blueprints:
            # print(blueprint_candidates.id)
            if blueprint_candidates.id == self.args.vehicle_id:
                blueprint = blueprint_candidates
                break
        if blueprint is None:
            blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))

        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if self.random_spawn == 1:
                print(f"Random spawn!")
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            else:
                spawn_location = carla.Location()
                spawn_location.x = float(self.args.spawn_x)
                spawn_location.y = float(self.args.spawn_y)
                spawn_waypoint = self.map.get_waypoint(spawn_location)
                spawn_transform = spawn_waypoint.transform
                spawn_transform.location.z = 1.0
                self.player = self.world.try_spawn_actor(blueprint, spawn_transform)
            
            print('vehicle spawned')

            spectator = self.world.get_spectator()
            transform = self.player.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(y=-10,z=28.5), carla.Rotation(pitch=-90)))

            # spawn_points = self.world.get_map().get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            # self.player = self.world.try_spawn_actor(blueprint, spawn_point)

            self.blueprint_library = self.world.get_blueprint_library()
            self.vehicle_blueprint = self.blueprint_library.filter('*vehicle*')
            self.walker_blueprint = self.blueprint_library.filter('*walker.*')

            ## CAMERA RGB

            self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
            self.rgb_cam.set_attribute("image_size_x", f"{640}")
            self.rgb_cam.set_attribute("image_size_y", f"{480}")
            self.rgb_cam.set_attribute("fov", f"110")
            self.camera_rgb = self.world.spawn_actor(
                self.rgb_cam,
                carla.Transform(carla.Location(x=2, z=1), carla.Rotation(0,0,0)),
                attach_to=self.player)

            ## CAMERA FOR VIZUALIZATION

            self.vis_cam = self.blueprint_library.find('sensor.camera.rgb')
            self.vis_cam.set_attribute("image_size_x", f"{640}")
            self.vis_cam.set_attribute("image_size_y", f"{480}")
            self.vis_cam.set_attribute("fov", f"110")
            self.camera_rgb_vis = self.world.spawn_actor(
                self.vis_cam,
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.player)

            ## LANE VIZUALIZATION

            self.lane_invasion = self.world.spawn_actor(
                self.blueprint_library.find('sensor.other.lane_invasion'), 
                carla.Transform(), 
                attach_to=self.player)

            ## COLLISION SENSOR

            self.collision_sensor = self.world.spawn_actor(
                self.blueprint_library.find('sensor.other.collision'),
                carla.Transform(),
                attach_to=self.player)
            
            ## CONTROLLER

            self.control_count = 0
            if self.control_mode == "PID":
                self.controller = PIDController.Controller()
                print("Control: PID")
            elif self.control_mode == "MPC":
                physic_control = self.player.get_physics_control()
                print("Control: MPC")
                lf, lr, l = get_vehicle_wheelbases(physic_control.wheels, physic_control.center_of_mass )
                self.controller = MPCController.Controller(lf = lf, lr = lr, wheelbase=l, planning_horizon = self.planning_horizon, time_step = self.time_step)
            velocity_vec = self.player.get_velocity()
            current_transform = self.player.get_transform()
            current_location = current_transform.location
            current_roration = current_transform.rotation
            current_x = current_location.x
            current_y = current_location.y
            current_yaw = wrap_angle(current_roration.yaw)
            current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
            frame, current_timestamp = self.hud.get_simulation_information()
            print(frame)
            print(current_timestamp)
            self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.player,
            self.collision_sensor,
            self.camera_rgb,
            self.camera_rgb_vis,
            self.lane_invasion
                    ]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def _initiate_visuals(self):
        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()



    def generate_episode(self, controller):


        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.lane_invasion, self.collision_sensor, fps=10) as sync_mode:
            

            counter = 0
            snapshot, image_rgb, image_rgb_vis, lane, collision = sync_mode.tick(timeout=10.0)
            self.episode_start = time.time()
            # vehicle_location = self.player.get_location()
            # print(vehicle_location)


            image = process_img2(self,image_rgb)
            next_state = image 

            

            # controller = VehicleControl(self.world)
            clock = pygame.time.Clock()
            while True:
                
                # if self.visuals:
                #     if should_quit():
                #         return
                #     self.clock.tick_busy_loop(30)
                counter += 1
                self.global_t += 1

                # clock.tick(5)
                clock.tick_busy_loop(self.args.FPS)
                # pygame.time.dalay(500)
                if controller.parse_events(self.client, self.world, clock):
                    return
                snapshot, image_rgb, image_rgb_vis, lane, collision = sync_mode.tick(timeout=10.0)
                image = process_img2(self, image_rgb)
                # world.tick(clock)
                # time.sleep (30)

    

        
   