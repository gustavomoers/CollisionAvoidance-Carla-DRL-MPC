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
        # world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
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
                    dist = self.world.time_step * current_speed + 0.1
                    prev_waypoint = self.world.map.get_waypoint(current_location)
                    current_waypoint = prev_waypoint.next(dist)[0]
                    # print(current_waypoint)
                    waypoints = []
                    road_desired_speed = self.world.desired_speed
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
                
                
                print(waypoints)
                # draw_waypoints(self.world, wp_draw)
                # print(current_x-waypoints[0][0])
                # print(current_x-waypoints[1][0])
                # print(current_x-waypoints[2][0])

                # print(current_y-waypoints[0][1])
                # print(current_y-waypoints[1][1])
                # print(current_y-waypoints[2][1])

                # x0= 
                # y0=  calcular os 3 primeiros waypoints
                # w0=
                # v_desired = 

             
              
                self.world.controller.update_waypoints(waypoints)     
                self.world.controller.update_controls()
                self._control.throttle, self._control.steer, self._control.brake = self.world.controller.get_commands()
                self.world.player.apply_control(self._control)
                self.world.control_count += 1
            # world.player.set_transform(current_waypoint.transform)
                

