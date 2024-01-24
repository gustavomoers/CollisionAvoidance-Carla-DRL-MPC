import carla
import time
from Utils.utils import *
from Utils.HUD import HUD as HUD
from World import World
import argparse
import logging
from stable_baselines3 import PPO #PPO
import os
from stable_baselines3.common.callbacks import CallbackList
from callbacks import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


# new_logger = configure(logdir, ["stdout", "csv", "tensorboard"])

def game_loop(args): 
    world=None   
    try: 

        client = carla.Client(args.host, args.port)
        client.set_timeout(100.0)
        hud = HUD()
        # carla_world = client.load_world(args.map)
        carla_world = client.get_world()
        # carla_world.apply_settings(carla.WorldSettings(
        #     no_rendering_mode=False,
        #     synchronous_mode=True,
        #     fixed_delta_seconds=1/args.FPS))
        world = World(client, carla_world, hud, args)
        world = Monitor(world, logdir)
        world.reset()
        model = PPO('CnnPolicy', world, verbose=2, learning_rate=0.0003, n_steps=640, n_epochs=30, batch_size=32, ent_coef=0.01,
                     tensorboard_log=logdir) # tensorboard_log=logdir
        # Create Callback
        save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir, verbose=1) 
        tensor = TensorboardCallback()  
        # logger = HParamCallback()
        # printer = MeticLogger()
        # plotter = PlottingCallback(log_dir=logdir)
        # checkpoint = CheckpointCallback(save_freq=500, save_path=models_dir)
       

        TIMESTEPS = 500000 # how long is each training iteration - individual steps
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", progress_bar=True, 
                        callback = CallbackList([tensor, save_callback])) 
                
    finally:

            if world is not None:
                world.destroy()        







# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--map',
        metavar='NAME',
        default='Town04',
        help='simulation map (default: "Town04")')
    argparser.add_argument(
        '--spawn_x',
        metavar='x',
        default='-16.75', #town04 = -16.75
        help='x position to spawn the agent')
    argparser.add_argument(
        '--spawn_y',
        metavar='y',
        default='-223.55', #town04 = -223.55
        help='y position to spawn the agent')
    argparser.add_argument(
        '--random_spawn',        
        metavar='RS',
        default='0',
        type=int,
        help='Random spawn agent')
    argparser.add_argument(
        '--vehicle_id',
        metavar='NAME',
        # default='vehicle.jeep.wrangler_rubicon',
        default='vehicle.tesla.model3',
        help='vehicle to spawn, available options : vehicle.audi.a2 vehicle.audi.tt vehicle.carlamotors.carlacola vehicle.citroen.c3 vehicle.dodge_charger.police vehicle.jeep.wrangler_rubicon vehicle.yamaha.yzf vehicle.nissan.patrol vehicle.gazelle.omafiets vehicle.bh.crossbike vehicle.ford.mustang vehicle.bmw.isetta vehicle.audi.etron vehicle.harley-davidson.low rider vehicle.mercedes-benz.coupe vehicle.bmw.grandtourer vehicle.toyota.prius vehicle.diamondback.century vehicle.tesla.model3 vehicle.seat.leon vehicle.lincoln.mkz2017 vehicle.kawasaki.ninja vehicle.volkswagen.t2 vehicle.nissan.micra vehicle.chevrolet.impala vehicle.mini.cooperst')
    argparser.add_argument(
        '--vehicle_wheelbase',
        metavar='NAME',
        type=float,
        default='2.89',
        help='vehicle wheelbase used for model predict control')
    argparser.add_argument(
        '--waypoint_resolution',
        metavar='WR',
        default='0.5',
        type=float,
        help='waypoint resulution for control')
    argparser.add_argument(
        '--waypoint_lookahead_distance',
        metavar='WLD',
        default='5.0',
        type=float,
        help='waypoint look ahead distance for control')
    argparser.add_argument(
        '--desired_speed',
        metavar='SPEED',
        default='13.89',
        type=float,
        help='desired speed for highway driving')
    argparser.add_argument(
        '--control_mode',
        metavar='CONT',
        default='MPC',
        help='Controller')
    argparser.add_argument(
        '--planning_horizon',
        metavar='HORIZON',
        type=int,
        default='3',
        help='Planning horizon for MPC')
    argparser.add_argument(
        '--time_step',
        metavar='DT',
        default='0.4',
        type=float,
        help='Planning time step for MPC')
    argparser.add_argument(
        '--FPS',
        metavar='FPS',
        default='15',
        type=int,
        help='Frame per second for simulation')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
