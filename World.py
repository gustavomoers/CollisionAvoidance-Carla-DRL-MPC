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
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = "vehicle.*"
        self._gamma = args.gamma
        self.args = args
        self.recording_start = 0
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        # print(self.desired_speed)
        self.planning_horizon = args.planning_horizon
        self.time_step = args.time_step
        self.control_mode = args.control_mode
        self.controller = None
        self.control_count = 0.0
        self.random_spawn = 0
        # self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.im_width = 640
        self.im_height = 480
        self.episode_start = 0
        self.visuals = visuals
        self.episode_reward = 0
        self.cos_list = []
        self.dist_list = []
        self.SHOW_CAM = True
        self.player = None
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
        self.max_dist = 2.5
        self.y_values_RL =np.array([self.waypoint_lookahead_distance, 2 * self.waypoint_lookahead_distance])
        self.x_values_RL = np.array([-self.max_dist, self.max_dist])
        # self.yaw_values_RL = np.array([self.max_dist, 2.5])
        self.counter = 0
        
    


        ## RL STABLE BASELINES
        self.action_space = spaces.Box(low=-1, high=1,shape=(6,),dtype="float32")
        self.observation_space = spaces.Box(low=-0, high=255, shape=(128, 128, 1), dtype=np.uint8)


        # self.visuals = visuals
        # if self.visuals:
        #     self._initiate_visuals()
        
        self.global_t = 0 # global timestep
        
    def reset(self, seed=None):

        self.destroy()
        self.episode_reward = 0

        if self.visuals:
            # Keep same camera config if the camera manager exists.
            cam_index = self.camera_manager.index if self.camera_manager is not None else 0
            cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
            # # # Get a random blueprint.
        # blueprints = self.world.get_blueprint_library().filter(self._actor_filter)
        # blueprint = None
        # for blueprint_candidates in blueprints:
        #     # print(blueprint_candidates.id)
        #     if blueprint_candidates.id == self.args.vehicle_id:
        #         blueprint = blueprint_candidates
        #         break
        # if blueprint is None:
        #     blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))

        # blueprint.set_attribute('role_name', self.actor_role_name)
        # if blueprint.has_attribute('color'):
        #     color = random.choice(blueprint.get_attribute('color').recommended_values)
        #     blueprint.set_attribute('color', color)
        # if blueprint.has_attribute('driver_id'):
        #     driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
        #     blueprint.set_attribute('driver_id', driver_id)
        # if blueprint.has_attribute('is_invincible'):
        #     blueprint.set_attribute('is_invincible', 'true')
        # # Spawn the player.
            
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.filter('*vehicle*')
        self.walker_blueprint = self.blueprint_library.filter('*walker.*')
  
        spawn_location = carla.Location()
        spawn_location.x = float(self.args.spawn_x)
        spawn_location.y = float(self.args.spawn_y)
        self.spawn_waypoint = self.map.get_waypoint(spawn_location)
        spawn_transform = self.spawn_waypoint.transform
        spawn_transform.location.z = 1.0
        self.player = self.world.try_spawn_actor(self.vehicle_blueprint.filter('model3')[0], spawn_transform)
            
        print('vehicle spawned')

        spectator = self.world.get_spectator()
        transform = self.player.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(y=20,z=58.5), carla.Rotation(pitch=-90)))

        # spawn_points = self.world.get_map().get_spawn_points()
        # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        # self.player = self.world.try_spawn_actor(blueprint, spawn_point)


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
        # print(frame)
        # print(current_timestamp)
        self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
        self.episode_start = time.time()

        if self.visuals:
            self.collision_sensor_hud = CollisionSensor(self.player, self.hud)
            self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
            self.gnss_sensor = GnssSensor(self.player)
            self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=False)


        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.lane_invasion, self.collision_sensor, fps=10) as self.sync_mode:
            snapshot, image_rgb, image_rgb_vis, lane, collision = self.sync_mode.tick(timeout=10.0)
            # if snapshot is not None:
            #     print("snapshot 1 ok")

            img = process_img2(self, image_rgb)
            self.clock = pygame.time.Clock()
            if self.visuals:  
                self.display = pygame.display.set_mode(
                            (self.args.width, self.args.height),
                            pygame.HWSURFACE | pygame.DOUBLEBUF)
                # self.world.tick(clock)
                # self.world.render(display)
                # pygame.display.flip()

            # while current_speed < self.desired_speed - 5:
            #     if self.parse_events(clock=clock, action=None):
            #         return
                    
            #     velocity_vec_start = self.player.get_velocity()
            #     current_speed_start = math.sqrt(velocity_vec_start.x**2 + velocity_vec_start.y**2 + velocity_vec_start.z**2)
            #     # print(f"curr_speed: {current_speed_start}")
            #     if current_speed_start > 5:
            #          break

        # im = cv2.imread("F:/CollisionAvoidance-Carla-DRL-MPC/_out/resized/first.png",cv2.IMREAD_GRAYSCALE)
        # shap = im[:, :, np.newaxis]

        return img, {}

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def destroy(self):

        actors = [
            self.player,
            self.collision_sensor,
            self.camera_rgb,
            self.camera_rgb_vis,
            self.lane_invasion]

        if self.collision_sensor_hud is not None:
            actors.append(self.collision_sensor_hud.sensor)
            actors.append(self.lane_invasion_sensor.sensor)
            actors.append(self.gnss_sensor.sensor)
            actors.append(self.camera_manager.sensor)             
                           
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

    def step(self, action):


        # with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.lane_invasion, self.collision_sensor, fps=10) as sync_mode:
            

        
        snapshot, image_rgb, image_rgb_vis, lane, collision = self.sync_mode.tick(timeout=10.0)

        
        # vehicle_location = self.player.get_location()       
        # waypoint = self.map.get_waypoint(vehicle_location, project_to_road=True, 
        #             lane_type=carla.LaneType.Driving)
                
        # if snapshot is not None:
        #     print("snapshot 1 ok")

        # destroy if there is no data
        if snapshot is None or image_rgb is None:
            print("No data, skipping episode")
            # self.reset()
            return None


        self.image = process_img2(self,image_rgb)
        next_state = self.image 

        self.reward = 0
        done = 0
        cos_yaw_diff = 0
        dist = 0
        collision = 0
        lane = 0
        stat = 0
        traveled = 0

        # if self.visuals:
        #     if should_quit():
        #         return
        #     self.clock.tick_busy_loop(30)

        if action is not None:

            # Advance the simulation and wait for the data.
            state = next_state
            self.counter += 1
            self.global_t += 1
            # print(self.global_t)

            # clock.tick(5)
            self.clock.tick_busy_loop(self.args.FPS)
            # pygame.time.dalay(500)
            if self.parse_events(action, self.clock):
                return
            
            # self.world.tick(clock)
            snapshot, image_rgb, image_rgb_vis, lane, collision = self.sync_mode.tick(timeout=10.0)

            # if snapshot is not None:
            #     print("snapshot 2 ok")
            
            # print(waypoint)
            # print(collision)
            # print(lane)


            cos_yaw_diff, dist, collision, lane, stat, traveled = self.get_reward_comp(self.player, self.spawn_waypoint, collision, lane, self.controller.stat)
            self.reward = self.reward_value(cos_yaw_diff, dist, collision, lane, stat, traveled)

            fps = round(1.0 / snapshot.timestamp.delta_seconds)

            if self.visuals:
    
                self.tick(self.clock)
                self.render(self.display)
                pygame.display.flip()



            # self.cos_list.append(cos_yaw_diff)
            # self.dist_list.append(dist)
            self.episode_reward += self.reward
            

            self.image = process_img2(self, image_rgb)
            
            done = 1 if collision or lane else 0

            
            if dist > self.max_dist:
                done=1


            truncated = 0

            if self.counter > 5000:
                truncated = 1
                print(truncated)
                
            if collision == 1:
                print("Episode ended by collision")
            
            if lane == 1:
                print("Episode ended by lane invasion")
    

            if dist > self.max_dist:
                print(f"Episode  ended with dist from waypoint: {dist}")
                

        return self.image, self.reward, done, truncated, {}

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

        lane = 0 if lane is None else 1

        traveled = y_vh - float(self.args.spawn_y)
        
 
        if stat is None:
            stat = 0
        elif stat == "infeasible":
            stat = -1
        elif stat == "optimal":
            stat = 1

        # finish = 1 if y_vh > -40 else 0
        
        return cos_yaw_diff, dist, collision, lane, stat, traveled
    
    def reward_value(self, cos_yaw_diff, dist, collision, lane, stat, traveled, lambda_1=3, lambda_2=5, lambda_3=100, lambda_4=100, lambda_5=0):
    
        reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision) - (lambda_4 * lane) + (lambda_5 * stat) + traveled
        
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
                if self.control_mode == "PID":
                    current_location = self.player.get_location()
                    current_waypoint = self.map.get_waypoint(current_location).next(self.world.waypoint_resolution)[0]
                    print(current_waypoint.transform.location.x-current_x)
                    print(current_waypoint.transform.location.y-current_y)
                    # wp_draw = []
                    waypoints = []
                    for i in range(int(self.world.waypoint_lookahead_distance / self.world.waypoint_resolution)):
                        waypoints.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y, self.world.desired_speed])
                        # wp_draw.append(current_waypoint)
                        current_waypoint = current_waypoint.next(self.world.waypoint_resolution)[0]
                        
                    # print(f"number of waypoints : {len(next_waypoint_list)}")
                    # current_location = world.player.get_location()
                        

                elif self.control_mode == "MPC":

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

                    x0 = (max(self.x_values_RL)-min(self.x_values_RL))*((action[0]+1)/2)+min(self.x_values_RL)+current_x
                    y0 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[1]+1)/2)+min(self.y_values_RL)+current_y
                    location0 = carla.Location()
                    location0.x = x0
                    location0.y = y0
                    yaw0 = wrap_angle(self.map.get_waypoint(location0).transform.rotation.yaw)
                    # yaw0 = (max(self.yaw_values_RL)-min(self.yaw_values_RL))*((action[2]+1)/2)+min(self.yaw_values_RL)

                    x1 = (max(self.x_values_RL)-min(self.x_values_RL))*((action[2]+1)/2)+min(self.x_values_RL)+current_x
                    y1 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[3]+1)/2)+min(self.y_values_RL)+y0
                    location1 = carla.Location()
                    location1.x = x1
                    location1.y = y1
                    yaw1 = wrap_angle(self.map.get_waypoint(location1).transform.rotation.yaw)
                    # yaw1 = (max(self.yaw_values_RL)-min(self.yaw_values_RL))*((action[5]+1)/2)+min(self.yaw_values_RL)

                    x2 = (max(self.x_values_RL)-min(self.x_values_RL))*((action[4]+1)/2)+min(self.x_values_RL)+current_x
                    y2 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[5]+1)/2)+min(self.y_values_RL)+y1
                    location2 = carla.Location()
                    location2.x = x2
                    location2.y = y2
                    yaw2 = wrap_angle(self.map.get_waypoint(location2).transform.rotation.yaw)
                    # yaw2 = (max(self.yaw_values_RL)-min(self.yaw_values_RL))*((action[8]+1)/2)+min(self.yaw_values_RL)

                    road_desired_speed = self.desired_speed
                    waypoints_RL = [[x0, y0, road_desired_speed, yaw0], [x1, y1, road_desired_speed, yaw1], [x2, y2, road_desired_speed, yaw2]]
                    # print(f'wp RL: {waypoints_RL}')
                
                    for z in waypoints_RL:
                        spawn_location_r = carla.Location()
                        spawn_location_r.x = float(z[0])
                        spawn_location_r.y = float(z[1])
                        spawn_waypoint_r = self.map.get_waypoint(spawn_location_r)
                        spawn_transform_r = spawn_waypoint_r.transform
                        spawn_transform_r.location.z = 1.0
                        self.world.debug.draw_string(spawn_transform_r.location, 'O', draw_shadow=False,
                                                            color=carla.Color(r=255, g=0, b=0), life_time=1,
                                                            persistent_lines=True)
            

                for x in waypoints:
                    spawn_location = carla.Location()
                    spawn_location.x = float(x[0])
                    spawn_location.y = float(x[1])
                    spawn_waypoint = self.map.get_waypoint(spawn_location)
                    spawn_transform = spawn_waypoint.transform
                    spawn_transform.location.z = 1.0
                    self.world.debug.draw_string(spawn_transform.location, 'O', draw_shadow=False,
                                                        color=carla.Color(r=0, g=255, b=0), life_time=1,
                                                        persistent_lines=True)
                    
                
                # draw_waypoints(self.world, wp_draw)
                # print(waypoints[0][0]-current_x)
                # print(waypoints[1][0]-current_x)
                # print(waypoints[2][0]-current_x)

                # print(waypoints[0][1]-current_y)
                # print(waypoints[1][1]-current_y)
                # print(waypoints[2][1]-current_y)

                # print(waypoints[0][3]-current_yaw)
                # print(waypoints[1][3]-current_yaw)
                # print(waypoints[2][3]-current_yaw)
                # x0= 
                # y0=  calcular os 3 primeiros waypoints
                # w0=
                # v_desired = 

             
                if action is not None:
                    # print(waypoints_RL)
                    self.controller.update_waypoints(waypoints_RL)
                else:
                    # print(waypoints)
                    self.controller.update_waypoints(waypoints)   
                self.controller.update_controls()
                
                self._control.throttle, self._control.steer, self._control.brake = self.controller.get_commands()
                # print(self._control)
                self.player.apply_control(self._control)
                self.control_count += 1
            # world.player.set_transform(current_waypoint.transform)

        

        
   