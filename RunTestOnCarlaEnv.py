import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    sys.path.append(
        glob.glob('//media/userzed/UbuntuStore/TAIAT/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')[
            0])
except IndexError:
    print("carla .egg not found")
    pass

import carla

# import gym_carla
# import gym
# import random
# import time
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
    'collision_distance': 2.0, # distance between pedestrian and ego-vehicle
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 100000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 10,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'client': carla.Client('localhost', 2000),
    # XYZ, carla's rotation: __init__(self, pitch=0.0, yaw=0.0, roll=0.0)
    # 1) [-20, 134.5, 1.0, 0.0, 0.0, 0.0]
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

  # Set gym-carla environment
  # gym.envs.register(
  #     id='carla-v0',
  #     #entry_point='gym.envs.classic_control:carlaEnv',
  #     #max_episode_steps=150,
  #     #kwargs={'size': 1, 'init_state': 10., 'state_bound': np.inf},
  # )
  # env = carlaEnv(params)
  # obs = env.reset()
  #
  # ego_trajectory = env.generateWaypoints(params['ego_vehicle_init_state'][:2], params['ego_vehicle_end_state'][:2])
  # print(ego_trajectory)
  # print(np.asarray(ego_trajectory))
  #
  # while True:
  #   action = [2.0, 0.0, ego_trajectory, env.ego_vehicle]
  #   obs,r,done,info = env.step(action)
  #
  #   if done:
  #     obs = env.reset()

  env = carlaEnv(params)
  obs = env.reset()

  ego_init_pos = params['ego_vehicle_init_state']
  ego_spawn_loc = carla.Location(x=ego_init_pos[0], y=ego_init_pos[1], z=ego_init_pos[2])
  ego_end_pos = params['ego_vehicle_end_state']
  ego_end_loc = carla.Location(x=ego_end_pos[0], y=ego_end_pos[1], z=ego_end_pos[2])
  ego_trajectory = env.generateWaypoints(ego_spawn_loc, ego_end_loc)
  # print(ego_trajectory)
  # print(np.asarray(ego_trajectory))

  Temp = []
  for waypoint in ego_trajectory:
    waypoint_x = waypoint[0].transform.location.x
    waypoint_y = waypoint[0].transform.location.y
    Temp.append([waypoint_x, waypoint_y])
  # trajectory = np.asarray(trajectory)
  trajectory = np.array(Temp)

  plt.plot(trajectory[:, 0], trajectory[:, 1])
  plt.show()

  while True:
    # action = [2.0, 0.0]
    # action = [2.0, 0.0, ego_trajectory, env.ego_vehicle]
    # obs, r, done, info = env.step(ego_trajectory, env.ego_vehicle)
    done, ego_state, ped_state = env.step(ego_trajectory, env.ego_vehicle)

    if done:
      # obs = env.reset()
      env.reset()

if __name__ == '__main__':
    main()