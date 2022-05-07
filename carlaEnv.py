import pygame
import sys
import matplotlib as plt

try:
    sys.path.append('/opt/carla-simulator/PythonAPI/carla/agents/navigation/')
except IndexError:
    print("navigation not found")
    pass
from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO
import numpy as np
import random
import glob
from utils import *
from reinforce_continuous import REINFORCE

try:
    sys.path.append(
        glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')[
            0])
except IndexError:
    print("carla .egg not found")
    pass
import carla
import gym
from gym import spaces
from gym.utils import seeding


class carlaEnv(gym.Env):
    def __init__(self, params):
        # parameters
        # self.display_size = params['display_size']  # rendering screen size
        self.collision_distance = params['collision_distance']
        self.max_past_step = params['max_past_step']
        self.delT = params['dt']
        self.max_time_episode = params['max_time_episode']
        # self.max_waypt = params['max_waypt']
        # self.obs_range = params['obs_range']
        # self.lidar_bin = params['lidar_bin']
        # self.d_behind = params['d_behind']
        # self.obs_size = int(self.obs_range/self.lidar_bin)
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route']
        self.total_step = 0

        # action and observation spaces
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'], params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
                                                     params['continuous_steer_range'][0]]),
                                           np.array([params['continuous_accel_range'][1],
                                                     params['continuous_steer_range'][1]]),
                                           dtype=np.float32)  # acc, steer

        # For PID control, we need route planner to generate waypoints, and vehicle's or persons' rotation.yaw
        self.integralPsiError = 0
        self.previousPsiError = 0
        self.integralPsiError = 0
        self.previousXdotError = 0
        self.integralXdotError = 0

        # for reward -- calculating distance threshold
        self.collision_thre = 3.0

        self.actor_list = []

        # 0. client that will send the requests to the simulator.
        self.client = params['client']  # port 2000
        self.client.set_timeout(200.0)  # timeout for connecting client

        # 1. world
        self.world = self.client.get_world()

        # set world timestamp
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.delT  # 20 fps, 1ms
        # settings.synchronous_mode = True
        self.world.apply_settings(settings)

        # 2. defining car
        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        self.blueprint_library = self.world.get_blueprint_library()
        # temp target car
        ego_vehicle_bp = self.blueprint_library.find('vehicle.mercedes-benz.coupe')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')

        # 3. spawn car
        ego_init_pos = params['ego_vehicle_init_state']
        self.destinations = [params['ego_vehicle_end_state']]
        self.ego_spawn_point = carla.Transform(
            carla.Location(x=ego_init_pos[0], y=ego_init_pos[1], z=ego_init_pos[2]),
            carla.Rotation(roll=ego_init_pos[3], pitch=ego_init_pos[4], yaw=ego_init_pos[5]))  # , carla.Rotation())
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, self.ego_spawn_point)
        self.actor_list.append(self.ego_vehicle)
        # Create Wheels Physics Control
        # tried 2.0
        front_left_wheel = carla.WheelPhysicsControl(tire_friction=5.0, damping_rate=1.5, max_steer_angle=70.0)
        front_right_wheel = carla.WheelPhysicsControl(tire_friction=5.0, damping_rate=1.5, max_steer_angle=70.0)
        rear_left_wheel = carla.WheelPhysicsControl(tire_friction=10.0, damping_rate=1.5, max_steer_angle=0.0)
        rear_right_wheel = carla.WheelPhysicsControl(tire_friction=10.0, damping_rate=1.5, max_steer_angle=0.0)

        wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
        # Change Vehicle Physics Control parameters of the vehicle
        physics_control = self.ego_vehicle.get_physics_control()
        physics_control.wheels = wheels
        self.ego_vehicle.apply_physics_control(physics_control)
        # self.ego_vehicle.set_autopilot(True)

        # print('created %s' % self.ego_vehicle.type_id)

        # 4. sensor
        # camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(camera)
        # print('created %s' % camera.type_id)
        # Now we register the callback
        camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))

        # Add collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # 5. spector -- let car in spectator
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))

        # 7. Trajectory PID
        # self.trajectory = ego_init_pos[0:2];

        # 8. Distance between Vehicle and Pedestrian
        self.relative_distance = 0
        self.ped_con_count = 0
        self.ped_y_control = 0
        self.stop_start_time = 0

    def spawn_walker(self, ped_action):
        # 6.  spawn pedestrian
        bp = random.choice(self.blueprint_library.filter('walker'))
        # bp = random.choice(self.world.get_blueprint_library.find('controller.ai.walker'))
        # ped_init_pos = params['ego_person_init_state']

        # ped_action = [x_spwan_pos, y_spwan_pos, rotation, trigger_distance]
        # TODO: ped z spawn point
        # ped_rotation = ped_action[2] * np.pi
        # if ped_rotation > np.pi:
        #     ped_rotation = np.pi
        # elif ped_rotation < -np.pi:
        #     ped_rotation = -np.pi
        # print(ped_action)
        self.ped_transform = carla.Transform(
            carla.Location(x=ped_action[0], y=ped_action[1], z=1.0),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=wrapToPi(ped_action[2])))  # )
        self.pedestrian = self.world.spawn_actor(bp, self.ped_transform)

        self.trigger_distance = ped_action[3]
        self.actor_list.append(self.pedestrian)
        # print('created %s' % self.pedestrian.type_id)

    def generateWaypoints(self, carla_location_start, carla_location_goal):
        '''
        Example input:
        a = carla.Location(x=96.0, y=4.45, z=0)
        b = carla.Location(x=215.0, y=6.23, z=0)
        '''
        # https://github.com/carla-simulator/carla/issues/1673
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 2.0)  # don't know what is 2.0
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        return grp.trace_route(carla_location_start, carla_location_goal)  # ? don't know data structure and shape

    def wrapToPi(self, a):
        # Wraps the input angle to between 0 and pi
        return (a + np.pi) % (2 * np.pi) - np.pi

    def clamp(self, n, minValue, maxValue):
        # Clamps to range [minValue, maxValue]
        return max(min(maxValue, n), minValue)

    def closestNode(self, X, Y, trajectory):
        # Finds the closest waypoints in the trajectory
        # with respect to the point [X, Y] given in as input
        point = np.array([X, Y])
        # Temp = []
        # for waypoint in trajectory:
        #     waypoint_x = waypoint[0].transform.location.x
        #     waypoint_y = waypoint[0].transform.location.y
        #     Temp.append([waypoint_x,waypoint_y])
        # # trajectory = np.asarray(trajectory)
        # trajectory = np.array(Temp)
        dist = point - trajectory
        # print(trajectory,dist)
        distSquared = np.sum(dist ** 2, axis=1)
        # distSquared = np.sum(dist ** 2)
        minIndex = np.argmin(distSquared)
        # print(minIndex)
        return np.sqrt(distSquared[minIndex]), minIndex

    def get_control_input_PID(self, trajectory, ego):
        delT = self.delT
        ego_trans = ego.get_transform()
        X = ego_trans.location.x
        Y = ego_trans.location.y
        psi = ego_trans.rotation.yaw / 180 * np.pi
        xdot = ego.get_velocity().x

        # Trajectory
        Temp = []
        for waypoint in trajectory:
            waypoint_x = waypoint[0].transform.location.x
            waypoint_y = waypoint[0].transform.location.y
            Temp.append([waypoint_x, waypoint_y])
        # trajectory = np.asarray(trajectory)
        trajectory = np.array(Temp)

        # Find the closest node to the vehicle
        _, node = self.closestNode(X, Y, trajectory)

        # Choose a node that is ahead of our current node based on index
        forwardIndex = 8  # tried 5,35, 50
        # We use a try-except so we don't attempt to grab an index that is out of scope
        try:
            psiDesired = np.arctan2(trajectory[node + forwardIndex, 1] - Y, trajectory[node + forwardIndex, 0] - X)
        except:
            psiDesired = np.arctan2(trajectory[-1, 1] - Y, trajectory[-1, 0] - X)
        # PID gains

        # Lateral
        kp = 0.405  # tried 1.0, 0.8, 0.5(worked-large), 0.3(worked-small)
        ki = 0.065  # tried 0.005
        kd = 0.001  # tried 0.001
        # Calculate difference between desired and actual heading angle
        psiError = self.wrapToPi(psiDesired - psi)
        self.integralPsiError += psiError
        derivativePsiError = psiError - self.previousPsiError
        delta = kp * psiError + ki * self.integralPsiError * delT + \
                kd * derivativePsiError / delT
        delta = -self.wrapToPi(delta)  # steering angle

        # Longitudinal
        kp = 200
        ki = 10
        kd = 30
        desiredVelocity = self.desired_speed  # tried 2 ,20

        xdotError = desiredVelocity - xdot
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError
        F = kp * xdotError + ki * self.integralXdotError * delT + \
            kd * derivativeXdotError / delT
        # if self.relative_distance < 10:
        #     self.ped_con_count += 1
        #
        # if F == 0 and self.time_step > 80:
        #     F = 0
        #     brake = 10000
        # elif self.ped_con_count != 0 and self.relative_distance > 3:
        #     F = kp * xdotError + ki * self.integralXdotError * delT + \
        #         kd * derivativeXdotError / delT

        # if self.relative_distance < 15:
        #     return -(F*20), delta
        # if self.relative_distance < 15 and self.ped_con_count == 0:
        #     self.ped_con_count += 1
        #     return -(F*20), delta
        # else:
        #     return F, delta
        if self.relative_distance < 10:
            self.ped_con_count = 1
        if self.relative_distance >= 6:
            self.ped_con_count = 0

        return F, delta

    def step(self, trajectory, ego):

        # trajectory = self.trajectory
        # ego = self.ego_vehicle

        # PID control for ego_vehicle
        throttle, steer = self.get_control_input_PID(trajectory, self.ego_vehicle)

        # brake = 0
        if self.ped_con_count == 1:
            throttle = -(throttle * 20)
            brake = np.absolute(throttle) if throttle < 0 else 0
            throttle = 0.0
        else:
            brake = np.absolute(throttle) if throttle < 0 else 0

        # print("throttle: ", throttle, "steer: ", steer, "brake: ", brake)
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego_vehicle.apply_control(act)

        # Carla Control for Pedestrian
        # pedestrian
        # direction (carla.Vector3D)
        # actp = carla.WalkerControl(direction=[1.0, 0.0, 0.0], speed=0.0, jump=False)
        # Critical scenario generation
        self.ped_y_control = -0.5
        if self.relative_distance <= 15.0:
            self.ped_y_control = -1.0
            self.ped_con_count += 1
        elif self.ped_con_count != 0:
            self.ped_y_control = -0.5
        else:
            self.ped_y_control = 0.0

        # actp = carla.WalkerControl(self.pedestrian.get_transform().get_forward_vector(), speed=3.0, jump=False)

        # print(self.pedestrian.get_transform().get_forward_vector())
        # self.pedestrian.apply_control(actp)

        # TODO: KEEP MODIFYING -- APPLY CONTROL IF DIST < TRIGGER DISTANCE # ??

        if ego_ped_distance(self.ego_vehicle, self.pedestrian) < self.trigger_distance:
            # print(self.pedestrian)
            actp = carla.WalkerControl(self.pedestrian.get_transform().get_forward_vector(), speed=3.0, jump=False)
            # print(actp)
            try:
                self.pedestrian.apply_control(actp)
                if self.control_verbose:
                    print("apply walker control")
                    self.control_verbose = False
            except:
                if self.control_verbose:
                    print("apply walker control exception! try skip")
                    self.control_verbose = False
                pass

        self.world.tick()

        # Update timestep
        self.time_step += 1
        self.total_step += 1
        # print(self.time_step)

        collision = len(self.collision_hist) > 0

        reward = self.get_reward()

        # collision, reward, done, info = env.step(ego_action)
        return collision, reward, self.terminal(self.ego_vehicle, self.pedestrian, trajectory)

    # TODO:
    def terminal(self, vehicle, pedestrian, trajectory):
        '''
        What are our termination condition?
        1. Collision (sensor-based)
        2. Time steps maxed out
        3. The ego-vehicle is close to one of the destination
        4. Ego-vehicle out of lane (DO WE REALLY NEED IT???)
        5.
        '''

        # Calculate whether to terminate the current episode.
        # Get ego state
        trans = vehicle.get_transform()
        ego_x = trans.location.x
        ego_y = trans.location.y
        ego_z = trans.location.z

        # Get ped state
        trans_ped = pedestrian.get_transform()
        ped_x = trans_ped.location.x
        ped_y = trans_ped.location.y
        ped_z = trans_ped.location.z

        self.relative_distance = np.sqrt((ego_x - ped_x) ** 2 + (ego_y - ped_y) ** 2 + (ego_z - ped_z) ** 2)
        # print('Distance between ego and ped:', self.relative_distance)

        ego_roll = trans.rotation.roll
        ego_pitch = trans.rotation.pitch
        ego_yaw = trans.rotation.yaw
        ped_roll = trans_ped.rotation.roll
        ped_pitch = trans_ped.rotation.pitch
        ped_yaw = trans_ped.rotation.yaw

        # 1. If collides -- Collision sensor
        if len(self.collision_hist) > 0:
            print('collision')
            return True  # ,[ego_x, ego_y, ego_z, ego_pitch, ego_yaw, ego_roll],\
            # [ped_x, ped_y, ped_z, ped_pitch, ped_yaw, ped_roll]

        # 2. If reach maximum timestep # need global var
        if self.time_step > self.max_time_episode:
            print('timestep maxed out')
            return True, [ego_x, ego_y, ego_z, ego_pitch, ego_yaw, ego_roll], \
                   [ped_x, ped_y, ped_z, ped_pitch, ped_yaw, ped_roll]

        # If at destination # set destination to ?
        # 3. If the ego-vehicle is close to one of the destination, we terminate.
        if self.destinations is not None:  # If at destination
            for destination in self.destinations:
                if np.sqrt((ego_x - destination[0]) ** 2 + (ego_y - destination[1]) ** 2) < 4:
                    print('destination')
                    return True  # ,[ego_x, ego_y, ego_z, ego_pitch, ego_yaw, ego_roll],\
                    # [ped_x, ped_y, ped_z, ped_pitch, ped_yaw, ped_roll]

        # 4. If out of lane
        Temp = []
        for waypoint in trajectory:
            waypoint_x = waypoint[0].transform.location.x
            waypoint_y = waypoint[0].transform.location.y
            Temp.append([waypoint_x, waypoint_y])
        trajectory = np.array(Temp)
        dis, _ = self.closestNode(ego_x, ego_y, trajectory)
        # print('Difference with planned trajectory :',dis)
        # print('trajectory is :',trajectory)
        # print(trajectory.shape)
        # 1.5 was used but too small for lane departure, 300 worked
        if abs(dis) > 4.0 * self.out_lane_thres:
            print('out of lane')
            return True  # ,[ego_x, ego_y, ego_z, ego_pitch, ego_yaw, ego_roll],\
            # [ped_x, ped_y, ped_z, ped_pitch, ped_yaw, ped_roll]

        # # if ego-vehicle is too close to pedestrian (Temporarily not used)
        # if relative_distance < self.collision_distance:
        #     return True

        # angle: pitch yaw roll in degrees, left hand rule, shit rule
        return False  # ,[ego_x, ego_y, ego_z, ego_pitch, ego_yaw, ego_roll],\
        # [ped_x, ped_y, ped_z, ped_pitch, ped_yaw, ped_roll]

    def destroy(self):
        # Clear sensor objects
        self.collision_sensor = None
        self.camera_sensor = None
        for actor in self.actor_list:
            # if actor.is_alive:
            actor.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

    def reset(self):

        # Delete sensors, vehicles and walkers
        filters = ['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*',
                   'controller.ai.walker', 'walker.*']
        # for actor in self.world.get_actors().filter(filters):
        # self.pedestrian.stop()
        # self.ego_vehicle.stop()
        for actor in self.actor_list:
            # if actor.is_alive:
            actor.destroy()
            self.actor_list = []
        # Clear sensor objects
        self.collision_sensor = None
        self.camera_sensor = None

        # Disable sync mode
        # self._set_synchronous_mode(False)

        # Spawn the ego vehicle
        # temp target car
        ego_vehicle_bp = self.blueprint_library.find('vehicle.mercedes-benz.coupe')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, self.ego_spawn_point)
        self.actor_list.append(self.ego_vehicle)

        # Add camera sensor
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(camera)
        # Now we register the callback
        # camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))

        # Add collision sensor
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        self.actor_list.append(self.collision_sensor)

        # Add pedestrian
        # bp = random.choice(self.blueprint_library.filter('walker'))
        # self.pedestrian = self.world.spawn_actor(bp, self.ped_transform)
        # self.actor_list.append(self.pedestrian)

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Update timesteps
        self.time_step = 0
        # self.reset_step += 1

        self.control_verbose = True

        # Enable sync mode
        # self.settings.synchronous_mode = True
        # self.world.apply_settings(self.settings)

        # self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        # self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Set ego information for render
        # self.birdeye_render.set_hero(self.ego, self.ego.id)

        return  # return states...

    def get_reward(self):

        # TODO: find scale for relative distance MODIFYING...
        rd = ego_ped_distance(self.ego_vehicle, self.pedestrian)

        # TODO: NEED TO DEFINE COLLISION THRESHOLD (tmp:3)
        if len(self.collision_hist) > 0 or ego_ped_distance(self.ego_vehicle, self.pedestrian) < self.collision_thre:
            rb = 10
        else:
            rb = 0

        # rp is defined in the main loop in RunTestOnCarlaEnv_with_re.py

        return -rd + rb

