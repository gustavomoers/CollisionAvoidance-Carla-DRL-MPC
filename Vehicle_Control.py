import carla
from Utils.utils import *


class VehicleControl(object):
    def __init__(self, world):
        self.world = world
        self._autopilot_enabled = False
        self._control = carla.VehicleControl()
        # if isinstance(self.world.player, carla.Vehicle):
        #     self._control = carla.VehicleControl()
        #     self.world.player.set_autopilot(self._autopilot_enabled)
        # elif isinstance(self.world.player, carla.Walker):
        #     self._control = carla.WalkerControl()
        #     self._autopilot_enabled = False
        #     self._rotation = self.world.player.get_transform().rotation
        # else:
            # raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        self.y_values_RL =np.array([0,5])
        self.x_values_RL = np.array([-10,10])
        self.yaw_values_RL = np.array([-3.14, 3.14])
        # world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, action):

        if not self._autopilot_enabled:
            # Control loop
            # get waypoints
            current_location = self.world.player.get_location()
            velocity_vec = self.world.player.get_velocity()
            current_transform = self.world.player.get_transform()
            current_location = current_transform.location
            current_rotation = current_transform.rotation
            current_x = current_location.x
            current_y = current_location.y
            current_yaw = wrap_angle(current_rotation.yaw)
            current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
            print(f"Control input : speed : {current_speed}, current position : {current_x}, {current_y}, yaw : {current_yaw}")
            frame, current_timestamp =self.world.hud.get_simulation_information()
            ready_to_go = self.world.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
            
            if ready_to_go:
                if self.world.control_mode == "PID":
                    print('here')
                    current_location = self.world.player.get_location()
                    current_waypoint = self.world.map.get_waypoint(current_location).next(self.world.waypoint_resolution)[0]
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
                        

                elif self.world.control_mode == "MPC":

                    x0 = (max(self.self.x_values_RL)-min(self.x_values_RL))*((action[0]+1)/2)+min(self.x_values_RL)
                    y0 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[1]+1)/2)+min(self.y_values_RL)
                    yaw0 = (max(self.yaw_values_RL)-min(self.yaw_values_RL))*((action[2]+1)/2)+min(self.yaw_values_RL)

                    x1 = (max(self.x_values_RL)-min(self.x_values_RL))*((action[3]+1)/2)+min(self.x_values_RL)
                    y1 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[4]+1)/2)+min(self.y_values_RL)+y0
                    yaw1 = (max(self.yaw_values_RL)-min(self.yaw_values_RL))*((action[5]+1)/2)+min(self.yaw_values_RL)

                    x2 = (max(self.x_values_RL)-min(self.x_values_RL))*((action[6]+1)/2)+min(self.x_values_RL)
                    y2 = (max(self.y_values_RL)-min(self.y_values_RL))*((action[7]+1)/2)+min(self.y_values_RL)+y1
                    yaw2 = (max(self.yaw_values_RL)-min(self.yaw_values_RL))*((action[8]+1)/2)+min(self.yaw_values_RL)

                    road_desired_speed = self.world.desired_speed
                    waypoints_RL = [[x0, y0, road_desired_speed, yaw0], [x1, y1, road_desired_speed, yaw1], [x2, y2, road_desired_speed, yaw2]]

                    dist = self.world.time_step * current_speed + 0.1
                    prev_waypoint = self.world.map.get_waypoint(current_location)
                    current_waypoint = prev_waypoint.next(dist)[0]
                    # print(current_waypoint)
                    waypoints = []
                    
                    # road_desired_speed = world.player.get_speed_limit()/3.6*0.95
                    for i in range(self.world.planning_horizon):
                        if self.world.control_count + i <= 100:
                            desired_speed = (self.world.control_count + 1 + i)/100.0 * road_desired_speed
                        else:
                            desired_speed = road_desired_speed
                        dist = self.world.time_step * road_desired_speed
                        current_waypoint = prev_waypoint.next(dist)[0]
                        print(f"current_waypoint: {current_waypoint}")
                        waypoints.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y, road_desired_speed, wrap_angle(current_waypoint.transform.rotation.yaw)])
                        prev_waypoint = current_waypoint

                
                
                print(f'wp real: {waypoints}')
                print(f'wp RL: {waypoints_RL}')
                
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

             
              
                self.world.controller.update_waypoints(waypoints_RL)     
                self.world.controller.update_controls()
                self._control.throttle, self._control.steer, self._control.brake = self.world.controller.get_commands()
                self.world.player.apply_control(self._control)
                self.world.control_count += 1
            # world.player.set_transform(current_waypoint.transform)
                
