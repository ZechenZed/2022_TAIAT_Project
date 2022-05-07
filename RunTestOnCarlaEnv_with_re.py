import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from reinforce_continuous import REINFORCE

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')[0])
except IndexError:
    print("carla egg not found")
    pass

import carla

# import gym_carla
import gym
import random
import time
from carlaEnv import carlaEnv

try:
    sys.path.append('../PythonAPI/carla/agents/navigation/')
except IndexError:
    print("navigation not found")
    pass
from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO


# How we test CarlaEnv:
def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 1,
        'number_of_walkers': 1,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.01,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'collision_distance': 2.0,  # distance between pedestrian and ego-vehicle
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1600,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
        'client': carla.Client('localhost', 2000),
        # XYZ, carla's rotation: __init__(self, pitch=0.0, yaw=0.0, roll=0.0)
        'ego_vehicle_init_state': [-30.4, -135.6, 1.0, 0.0, 0.0, 0.0],
        # 1) [20, 134.5, 1.0, 0.0, 0.0, 0.0]
        # 2) [38.1, -133.8, 2.0](straight) [-25.0,-134.0,1.0] [7.5, -179.3, 1.0](left-turn)
        # 3) 1. [6.7, -150.0, 1.0] 2.[21.0, -134.4, 1.0] 3.[-1.0, -121.1, 1.0]
        'ego_vehicle_end_state': [7.5, -179.3, 1.0],
        'ego_person_init_state': [-15, -128, 1.0, 90.0, 0.0, 0.0],
        'ego_person_end_state': [-20, 134.5, 1.0, 0.0, 0.0, 0.0],
        'xscope_min': -12,
        'xscope_max': 17,
        'yscope_min': -148,
        'yscope_max': -128,
    }

    lr = 0.005  # learning rate
    gamma = 0.0  # reward decay rate
    num_epoch = 10000  # training epoch number
    agent = REINFORCE(lr=lr, gamma=gamma)  # reinforce algorithm
    agent.save_model()

    num_runs = 32  # number of simulation runs before parameter update

    # cache
    rewards = dict()

    # get the trajectory
    env = carlaEnv(params)
    ego_init_pos = params['ego_vehicle_init_state']
    ego_spawn_loc = carla.Location(x=ego_init_pos[0], y=ego_init_pos[1], z=ego_init_pos[2])
    ego_end_pos = params['ego_vehicle_end_state']
    ego_end_loc = carla.Location(x=ego_end_pos[0], y=ego_end_pos[1], z=ego_end_pos[2])
    ego_trajectory = env.generateWaypoints(ego_spawn_loc, ego_end_loc)
    env.reset()

    Temp = []
    for waypoint in ego_trajectory:
        waypoint_x = waypoint[0].transform.location.x
        waypoint_y = waypoint[0].transform.location.y
        Temp.append([waypoint_x, waypoint_y])
    route = np.array(Temp)
    plt.plot(route[:, 0], route[:, 1])
    plt.show()

    for t_i in range(num_epoch):

        reward_b = []
        log_prob_b = []
        entropy_b = []
        exist = False

        for b_i in range(num_runs):
            # trajectory for env step
            ego_action = [2.0, 0.0, ego_trajectory, env.ego_vehicle]

            # select pre action
            target_speed = 0.4  # TODO: car speed

            route_norm = normalize_routes(route[0:30, :])
            route_norm = np.concatenate((route_norm, [[target_speed]]), axis=0)
            route_norm = route_norm.astype('float32')
            ped_action, log_prob, entropy = agent.select_action(route_norm)

            # TODO: check -- find the matched action in memory dictionary
            a_i = []
            for i in range(len(ped_action)):
                # try generate a key ?
                a_i.append(str(int(ped_action[i] * 1000) / 1000))  # .1
            a_i = "".join(a_i)

            # if the action is already generated
            if a_i in rewards.keys():
                reward = rewards[a_i]

            # run env + store to a cache dictionary

            else:
                # if the position is too close to the route, give a panelty
                route_penalty = check_route_distance(ped_action, route)
                if route_penalty:
                    reward = -40
                    print("spawned too close to trajectory!")
                else:
                    # initialize carlaEnv
                    # ped_action = [x_spwan_pos, y_spwan_pos, rotation, trigger_distance]
                    ped_action[0], ped_action[1], ped_action[3] = rescale_spwan_pos(ped_action)
                    # env = carlaEnv(params, ped_action)
                    obs = env.reset()
                    env.spawn_walker(ped_action)
                    print("spawn at: x: ", ped_action[0], "y: ", ped_action[1], "trigger distance: ", ped_action[3])
                    # clear up + set collision sensor
                    # set world timestamp
                    # settings = env.world.get_settings()
                    # settings.synchronous_mode = True
                    # env.world.apply_settings(settings)

                    while True:
                        # one run
                        collision, reward, done = env.step(ego_trajectory, env.ego_vehicle)
                        if done:
                            # settings = env.world.get_settings()
                            # settings.synchronous_mode = False
                            # env.world.apply_settings(settings)
                            state = env.reset()
                            # env.destroy()
                            # print('one round end !!')

                            break

                rewards[a_i] = reward

            reward_b.append(reward)
            log_prob_b.append(log_prob)
            entropy_b.append(entropy)

        # update parameters based on historical runs
        agent.update_parameters(reward_b, log_prob_b, entropy_b)

        # eval
        if (t_i) % 2 == 0:
            # output a deterministic action
            target_speed = 0.4  # ? car speed ? ped speed ?
            route_norm = normalize_routes(route[0:30, :])
            route_norm = np.concatenate((route_norm, [[target_speed]]), axis=0)
            route_norm = route_norm.astype('float32')
            action_det = agent.deterministic_action(route_norm)  # gaussian -- mean

            action_det_1 = action_det[0]  # X
            action_det_2 = action_det[1]  # Y
            action_det_3 = action_det[2]  # orientation
            action_det = [action_det_1, action_det_2, action_det_3]
            agent.save_model()

            print('[{}/{}] Reward: {}, Action: {}'.format(t_i + 1, num_epoch, rewards[a_i], action_det))
        print('-------------------------')

    # 10 trials ... mean
    #


if __name__ == '__main__':
    main()

