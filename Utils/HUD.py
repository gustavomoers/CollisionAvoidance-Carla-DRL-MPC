class HUD(object):
    def __init__(self):
        self.frame = 0
        self.simulation_time = 0


    def on_world_tick(self, timestamp):
        # self._server_clock.tick()
        # self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def get_simulation_information(self):
        return self.frame, self.simulation_time

    # def tick(self, world, clock):
    #     self._notifications.tick(world, clock)
    #     if not self._show_info:
    #         return
    #     t = world.player.get_transform()
    #     v = world.player.get_velocity()
    #     c = world.player.get_control()
    #     heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
    #     heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
    #     heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
    #     heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
    #     colhist = world.collision_sensor.get_collision_history()
    #     collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
    #     max_col = max(1.0, max(collision))
    #     collision = [x / max_col for x in collision]
    #     vehicles = world.world.get_actors().filter('vehicle.*')
    #     self.record["speed"].append(math.sqrt(v.x**2 + v.y**2 + v.z**2))
    #     self.record["yaw"].append(t.rotation.yaw)
    #     self.record["time"].append(self.simulation_time)
    #     # print(self.simulation_time)
    #     pickle.dump(self.record, open( "MPC_record.pkl", "wb" ) )


    #     self._info_text = [
    #         'Server:  % 16.0f FPS' % self.server_fps,
    #         'Client:  % 16.0f FPS' % clock.get_fps(),
    #         '',
    #         'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
    #         'Map:     % 20s' % world.map.name,
    #         'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
    #         '',
    #         'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
    #         u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
    #         'Location:% 30s' % ('(% 5.1f, % 5.1f, % 5.1f)' % (t.location.x, t.location.y, t.location.z)),
    #         'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
    #         'Height:  % 18.0f m' % t.location.z,
    #         '']
    #     if isinstance(c, carla.VehicleControl):
    #         self._info_text += [
    #             ('Throttle:', c.throttle, 0.0, 1.0),
    #             ('Steer:', c.steer, -1.0, 1.0),
    #             ('Brake:', c.brake, 0.0, 1.0),
    #             ('Reverse:', c.reverse),
    #             ('Hand brake:', c.hand_brake),
    #             ('Manual:', c.manual_gear_shift),
    #             'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
    #     elif isinstance(c, carla.WalkerControl):
    #         self._info_text += [
    #             ('Speed:', c.speed, 0.0, 5.556),
    #             ('Jump:', c.jump)]
    #     self._info_text += [
    #         '',
    #         'Collision:',
    #         collision,
    #         '',
    #         'Number of vehicles: % 8d' % len(vehicles)]
    #     if len(vehicles) > 1:
    #         self._info_text += ['Nearby vehicles:']
    #         distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
    #         vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
    #         for d, vehicle in sorted(vehicles):
    #             if d > 200.0:
    #                 break
    #             vehicle_type = get_actor_display_name(vehicle, truncate=22)
    #             self._info_text.append('% 4dm %s' % (d, vehicle_type))

