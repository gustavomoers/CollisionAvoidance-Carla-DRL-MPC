import pygame
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
import random
from Utils.CubicSpline.cubic_spline_planner import *
import csv
from Utils.HUD_visuals import *


class World(gym.Env):
    def __init__(self, client, carla_world, hud, args, visuals=False):
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
        self._weather_index = 0
        self._actor_filter = "vehicle.*"
        self._gamma = args.gamma
        self.args = args
        self.recording_start = 0
        self._gamma = args.gamma
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        self.planning_horizon = args.planning_horizon
        self.time_step = args.time_step
        self.control_mode = args.control_mode
        self.controller = None
        self.control_count = 0.0
        self.random_spawn = 0
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.im_width = 640
        self.im_height = 480
        self.episode_start = 0
        self.visuals = visuals
        self.episode_reward = 0
        self.player = None
        self.parked_vehicle = None
        self.collision_sensor = None
        self.camera_rgb = None
        self.camera_rgb_vis = None
        self.lane_invasion = None
        self.collision_sensor_hud = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._autopilot_enabled = False
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self.max_dist = 4.5
        self.y_values_RL =np.array([5, 10])
        self.x_values_RL = np.array([-3.5, 3.5])
        self.v_values_RL = np.array([0, 40])
        self.min_values_obs = np.array([-6, -15, 0, -1.5, -3.14])
        self.max_values_obs = np.array([6, 15, 40, 2, 3.14])
        self.counter = 0
        self.frame = None
        self.delta_seconds = 1.0 / args.FPS
        self.last_y = 0
        self.distance_parked = 35
        self.prev_action = np.array([0, 0, 0, 0, 0])
        self.ttc_trigger = 1
        self.episode_counter = 0
        self.last_v = 0
        self.save_list = []
        self.file_name = 'F:/CollisionAvoidance-Carla-DRL-MPC/logs/1709461045-recurrentPPO-90kmh-transfer/evaluation/logger_test_10hp.csv'
        self.logger = False
        self.path = []


        #VISUAL PYGAME
        self._weather_presets = find_weather_presets()
        self.visuals = visuals
        self.collision_sensor_hud = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None


        ## RL STABLE BASELINES
        self.action_space = spaces.Box(low=-1, high=1,shape=(5,),dtype="float32")
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype="float64")

        
        self.global_t = 0 # global timestep


    def append_to_csv(self,file_name, data):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)


    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)



    def reset(self, seed=None):

        self.print_path(self.path)
        time.sleep(2)
        self.path = []


        self.destroy()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1/self.args.FPS))
        self.episode_reward = 0
        self.desired_speed = self.args.desired_speed


        self.episode_counter += 1

        if self.logger:
            self.append_to_csv(file_name=self.file_name, data=self.save_list)
        self.save_list = []

        if self.visuals:
            # Keep same camera config if the camera manager exists.
            cam_index = self.camera_manager.index if self.camera_manager is not None else 0
            cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        
        self.create_actors()
  
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_roration = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y
        current_yaw = wrap_angle(current_roration.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        frame, current_timestamp = self.hud.get_simulation_information()
        self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
        self.episode_start = time.time()

        if self.visuals:
            self.collision_sensor_hud = CollisionSensor(self.player, self.hud)
            self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
            self.gnss_sensor = GnssSensor(self.player)
            self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=False)

        

        self.world.tick()
                   
        self.clock = pygame.time.Clock()

            
        ttc = self.time_to_collison()

        while ttc > self.ttc_trigger: #player_position < parked_position - self.realease_position:
            
            # snapshot, image_rgb, lane, collision = self.synch_mode.tick(timeout=10.0)

            self.clock.tick_busy_loop(self.args.FPS)

            if self.visuals:  
                self.display = pygame.display.set_mode(
                            (self.args.width, self.args.height),
                            pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.display.fill((0,0,0))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False

                self.tick(self.clock)
                self.render(self.display)
                pygame.display.update()

            if self.parse_events(clock=self.clock, action=None):
                 return
            
            velocity_vec_st = self.player.get_velocity()
            current_speed = math.sqrt(velocity_vec_st.x**2 + velocity_vec_st.y**2 + velocity_vec_st.z**2)


            ttc = self.time_to_collison()
            # print(f'ttc: {ttc}')


            snapshot, image_rgb, lane, collision = self.synch_mode.tick(timeout=10.0)

            obs = self.get_observation()
            obs = np.array(np.append(obs, self.prev_action))
  
            # img = process_img2(self, image_rgb)

        last_transform = self.player.get_transform()
        last_location = last_transform.location
        self.last_y = last_location.y
        self.last_v = current_speed
        print(current_speed)


        return obs, {}



    def tick(self, clock):
        self.hud.tick(self, clock)

    def destroy(self):

        self.world.tick()
            
        actors = [
            self.player,
            self.collision_sensor,
            self.camera_rgb,
            self.lane_invasion,
            self.parked_vehicle]

        if self.collision_sensor_hud is not None:
            actors.append(self.collision_sensor_hud.sensor)
            actors.append(self.lane_invasion_sensor.sensor)
            actors.append(self.gnss_sensor.sensor)
            actors.append(self.camera_manager.sensor)             
                           
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                    self.world.tick()
                except:
                    pass



    def step(self, action):


        self.reward = 0
        done = False
        cos_yaw_diff = 0
        dist = 0
        collision = 0
        lane = 0
        stat = 0
        traveled = 0

        if action is not None:

            self.counter += 1
            self.global_t += 1

            
            self.clock.tick_busy_loop(self.args.FPS)
            
            if self.parse_events(action, self.clock):
                return
            
            
            
            snapshot, image_rgb, lane, collision = self.synch_mode.tick(timeout=10.0)
            
            
            obs = self.get_observation()

            obs = np.array(np.append(obs, self.prev_action))
            self.prev_action = []
            self.prev_action.append(action)


            cos_yaw_diff, dist, collision, lane, stat, traveled = self.get_reward_comp(self.player, self.spawn_waypoint, collision, lane, self.controller.stat)
            
            
            self.reward = self.reward_value(cos_yaw_diff, dist, collision, lane, stat, traveled)

            if self.visuals:
                self.display.fill((0,0,0))
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False                

                self.tick(self.clock)
                self.render(self.display)
                pygame.display.flip()
 

            self.episode_reward += self.reward


            if dist > self.max_dist:
                done=True


            vehicle_location = self.player.get_location()
            y_vh = vehicle_location.y
            if y_vh > float(self.args.spawn_y)+self.distance_parked+15:
                print("episode ended by reaching goal position")
                done=True


            truncated = False
 

            if collision == 1:
                done=True
                print("Episode ended by collision")
            
            if lane == 1:
                done = True
                print("Episode ended by lane invasion")
    
            if dist > self.max_dist:
                done=True
                print(f"Episode  ended with dist from waypoint: {dist}")


            velocity_vec_st = self.player.get_velocity()
            current_speed = math.sqrt(velocity_vec_st.x**2 + velocity_vec_st.y**2 + velocity_vec_st.z**2)
            if current_speed < 0.1:
                print("Episode ended by stopping")
                done=True
                

        return obs, self.reward, done, truncated, {}
    


    def get_reward_comp(self, vehicle, waypoint, collision, lane, stat):
        vehicle_location = vehicle.get_location()
        x_wp = waypoint.transform.location.x
        y_wp = waypoint.transform.location.y

        x_vh = vehicle_location.x
        y_vh = vehicle_location.y

        wp_array = np.array([x_wp])
        vh_array = np.array([x_vh])

        dist = abs(np.linalg.norm(wp_array - vh_array))

        vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
        wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
        cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

        collision = 0 if collision is None else 1

        if lane is not None:
            lane_types = set(x.type for x in lane.crossed_lane_markings)
            text = ['%r' % str(x).split()[-1] for x in lane_types]
            lane = 1 if text[0] == "'Solid'" else 0
        
        elif lane is None:
            lane=0

        # lane = 0 if lane is None else 1

        traveled = y_vh - self.last_y
        # print(traveled)
 
        # print(stat)
        if stat is None:
            stat = 0
        elif stat == "infeasible":
            print(stat)
            stat = -1
        elif stat == "optimal":
            stat = 1

        # finish = 1 if y_vh > -40 else 0
        
        return cos_yaw_diff, dist, collision, lane, stat, traveled
    
    def reward_value(self, cos_yaw_diff, dist, collision, lane, stat, traveled, lambda_1=1, lambda_2=1, lambda_3=100, lambda_4=5, lambda_5=0.5):
    
        reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision) - (lambda_4 * lane) + (lambda_5 * traveled)
        
        return reward
    
    def parse_events(self, action, clock):

        if not self._autopilot_enabled:
            # Control loop
            # get waypoints
            current_location = self.player.get_location()
            velocity_vec = self.player.get_velocity()
            current_transform = self.player.get_transform()
            current_location = current_transform.location
            current_rotation = current_transform.rotation
            current_x = current_location.x
            current_y = current_location.y
            current_yaw = wrap_angle(current_rotation.yaw)
            current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
            # print(f"Control input : speed : {current_speed}, current position : {current_x}, {current_y}, yaw : {current_yaw}")
            frame, current_timestamp =self.hud.get_simulation_information()
            ready_to_go = self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
            
            if ready_to_go:
                if self.control_mode == "PID"and action is None:
                    current_location = self.player.get_location()
                    current_waypoint = self.map.get_waypoint(current_location).next(self.world.waypoint_resolution)[0]
                    # print(current_waypoint.transform.location.x-current_x)
                    # print(current_waypoint.transform.location.y-current_y)            
                    waypoints = []
                    for i in range(int(self.world.waypoint_lookahead_distance / self.world.waypoint_resolution)):
                        waypoints.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y, self.world.desired_speed])
                        current_waypoint = current_waypoint.next(self.world.waypoint_resolution)[0]
   

                elif self.control_mode == "MPC" and action is None:
                    road_desired_speed = self.desired_speed
                    dist = self.time_step * current_speed + 0.1
                    prev_waypoint = self.map.get_waypoint(current_location)
                    current_waypoint = prev_waypoint.next(dist)[0]
                    # print(current_waypoint)
                    waypoints = []                   
                    # road_desired_speed = world.player.get_speed_limit()/3.6*0.95
                    for i in range(self.planning_horizon):
                        if self.control_count + i <= 100:
                            desired_speed = (self.control_count + 1 + i)/100.0 * road_desired_speed
                        else:
                            desired_speed = road_desired_speed
                        dist = self.time_step * road_desired_speed
                        current_waypoint = prev_waypoint.next(dist)[0]
                        # print(f"current_waypoint: {current_waypoint}")
                        waypoints.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y, road_desired_speed, wrap_angle(current_waypoint.transform.rotation.yaw)])
                        prev_waypoint = current_waypoint

                # print(f'wp real: {waypoints}')
                if action is not None:
                    waypoints_RL = self.get_cubic_spline_path(action, current_x=current_x, current_y=current_y)
                    self.print_waypoints(waypoints_RL)
                    # print(waypoints_RL)
                    self.controller.update_waypoints(waypoints_RL)
                else:
                    self.print_waypoints(waypoints)
                    self.controller.update_waypoints(waypoints)  

                self.controller.update_controls()
                self._control.throttle, self._control.steer, self._control.brake = self.controller.get_commands()
                # print(self._control)

                self.player.apply_control(self._control)
                self.control_count += 1
            # world.player.set_transform(current_waypoint.transform)
 

    
    def print_waypoints(self, waypoints):

        for z in waypoints:
            spawn_location_r = carla.Location()
            spawn_location_r.x = float(z[0])
            spawn_location_r.y = float(z[1])
            spawn_location_r.z = 1.0
            self.world.debug.draw_string(spawn_location_r, 'O', draw_shadow=False,
                                                color=carla.Color(r=255, g=0, b=0), life_time=0.1,
                                                persistent_lines=True)


    def get_cubic_spline_path(self, action, current_x, current_y):
        # print(current_x)
        # x0 = current_x +(max(self.x_values_RL)-min(self.x_values_RL))*((action[0]+1)/2)+min(self.x_values_RL)
        # y0 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[1]+1)/2)+min(self.y_values_RL)+current_y
        y0 = current_y 
        # print(x0)

       
        x1 = (max(self.x_values_RL)-min(self.x_values_RL))*((action[0]+1)/2)+min(self.x_values_RL)+current_x
        y1 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[1]+1)/2)+min(self.y_values_RL)+y0
        # y1 = y0 + self.waypoint_lookahead_distance
        # print(x1)

        x2 = current_x +(max(self.x_values_RL)-min(self.x_values_RL))*((action[2]+1)/2)+min(self.x_values_RL)
        y2 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[3]+1)/2)+min(self.y_values_RL)+y1
        # y2 = y1 + self.waypoint_lookahead_distance
        # print(x2)

        x= [current_x ,  x1, x2]
        # print(x)
        y = [current_y + 2,  y1, y2]

        ds = self.waypoint_resolution  # [m] distance of each interpolated points
        sp = CubicSpline2D(x, y)

        s = np.arange(0, sp.s[-1], ds)
        rx, ry, ryaw= [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))


        desired_speed = (max(self.v_values_RL)-min(self.v_values_RL))*((action[4]+1)/2)+min(self.v_values_RL)
        velocity_vec = self.player.get_velocity()
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        num_steps= len(rx)

        speed_profile = [current_speed + (i / num_steps) * (desired_speed - current_speed) for i in range(num_steps + 1)]


        nw_wp = []
        for i in range(len(rx)):
            nw_wp.append([rx[i], ry[i], speed_profile[i], ryaw[i]])

        # print(nw_wp)

        return nw_wp
    
    def get_observation(self):


        # Example min and max values
        # min_values = [min_x_dist, min_y_dist, min_speed, min_acceleration, min_yaw]
        # max_values = [max_x_dist, max_y_dist, max_speed, max_acceleration, max_yaw]

        min_values = self.min_values_obs #np.array([-6, -15, 0, -1.5, -3.14])
        max_values = self.max_values_obs #np.array([6, 15, 40, 2, 3.14])

         # EGO information
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_roration = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y
        current_yaw = wrap_angle(current_roration.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        current_acceleration = self.controller._acceleration
        current_steer = self.controller._steer
        #Parked vehicle information
        parked_transform = self.parked_vehicle.get_transform()
        parked_location = parked_transform.location
        parked_x = parked_location.x
        parked_y = parked_location.y

        x_dist = current_x -parked_x
        y_dist = current_y -parked_y

        self.path.append([current_x, current_y])

        obs = [x_dist, y_dist, current_speed, current_acceleration, current_yaw]
        # print(obs)

        # Example observation data
        data = np.array([x_dist, y_dist, current_speed, current_acceleration, current_yaw])

        # Clipping the data
        clipped_data = np.clip(data, min_values, max_values)
        # print(clipped_data)

        # Normalize the data to the range [-1, 1]
        normalized_data = 2 * ((clipped_data - min_values) / (max_values - min_values)) - 1
        # print(normalized_data)

        acceleration_vec =  self.player.get_acceleration()
        sideslip = np.tanh(velocity_vec.x/np.abs(velocity_vec.y))

        # print(self.player.get_telemetry_data())


        self.save_list.append([self.episode_counter,  self.desired_speed, self.last_v, self.ttc_trigger, self.distance_parked, self.clock.get_time(), current_x, current_y, x_dist, y_dist, current_speed, current_acceleration, 
                               acceleration_vec.x, acceleration_vec.y, sideslip, current_yaw, current_steer])


        return normalized_data

    def time_to_collison(self):

         # EGO information
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_x = current_location.x
        current_y = current_location.y
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
       

        #Parked vehicle information
        parked_transform = self.parked_vehicle.get_transform()
        velocity_parked = self.parked_vehicle.get_velocity()
        parked_location = parked_transform.location
        parked_x = parked_location.x
        parked_y = parked_location.y
        parked_speed = math.sqrt(velocity_parked.x**2 + velocity_parked.y**2 + velocity_parked.z**2)

        dist = np.sqrt((parked_y-current_y)**2 + (current_x-parked_x)**2)
        rel_speed = current_speed - parked_speed

        ttc = dist/rel_speed

        return np.abs(ttc)



    def create_actors(self):

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.filter('*vehicle*')
        self.walker_blueprint = self.blueprint_library.filter('*walker.*')

        # PLAYER
    
        spawn_location = carla.Location()
        spawn_location.x = float(self.args.spawn_x)
        spawn_location.y = float(self.args.spawn_y)
        self.spawn_waypoint = self.map.get_waypoint(spawn_location)
        spawn_transform = self.spawn_waypoint.transform
        spawn_transform.location.z = 1.0
        self.player = self.world.try_spawn_actor(self.vehicle_blueprint.filter('model3')[0], spawn_transform)
        self.world.tick()   
        print('vehicle spawned')

        physic_control = self.player.get_physics_control()
        physic_control.use_sweep_wheel_collision = True

        # Turn on position lights
        current_lights = carla.VehicleLightState.NONE
        current_lights |= carla.VehicleLightState.Position
        self.player.set_light_state(carla.VehicleLightState.Position)

        # CAMERA RGB

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{640}")
        self.rgb_cam.set_attribute("image_size_y", f"{480}")
        self.rgb_cam.set_attribute("fov", f"110")
        self.camera_rgb = self.world.spawn_actor(
            self.rgb_cam,
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(0,0,0)),
            attach_to=self.player)
        self.world.tick()

        # LANE SENSOR

        self.lane_invasion = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'), 
            carla.Transform(), 
            attach_to=self.player)
        self.world.tick()

        # COLLISION SENSOR

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.player)
        self.world.tick()
        
        # SYNCH MODE CONTEXT

        self.synch_mode = CarlaSyncMode(self.world, self.camera_rgb, self.lane_invasion, self.collision_sensor)

        # STATIONARY CAR

        parking_position = carla.Transform(self.player.get_transform().location + carla.Location(-0.5, self.distance_parked, 0.5), 
                                carla.Rotation(0,90,0))
        self.parked_vehicle = self.world.spawn_actor(self.vehicle_blueprint.filter('model3')[0], parking_position) #self.vehicle_blueprint.filter('model3')[0]
        self.world.tick()


        # SPECTATOR

        spectator = self.world.get_spectator()
        if self.parked_vehicle is not None:
            transform = self.parked_vehicle.get_transform()
        else:
            transform = self.player.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(y=-10,z=28.5), carla.Rotation(pitch=-90)))
        self.world.tick()

        # CONTROLLER

        
        self.control_count = 0
        if self.control_mode == "PID":
            self.controller = PIDController.Controller()
            # print("Control: PID")
        elif self.control_mode == "MPC":
            physic_control = self.player.get_physics_control()
            physic_control.use_sweep_wheel_collision = True


            lf, lr, l = get_vehicle_wheelbases(physic_control.wheels, physic_control.center_of_mass )
            self.controller = MPCController.Controller(lf = lf, lr = lr, wheelbase=l, planning_horizon = self.planning_horizon, time_step = self.time_step)


    def print_path(self, waypoints):

        for z in waypoints:
            spawn_location_r = carla.Location()
            spawn_location_r.x = float(z[0])
            spawn_location_r.y = float(z[1])
            spawn_location_r.z = 1.0
            self.world.debug.draw_string(spawn_location_r, 'O', draw_shadow=False,
                                                color=carla.Color(r=0, g=0, b=255), life_time=5,
                                                persistent_lines=True)