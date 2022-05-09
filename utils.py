# Utils
import numpy as np


# RL - route_penalty
# constraint on ped spwan point
def constraint(x, x_min, x_max):
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    else:
        return x


def rescale_spwan_pos(actions):
    x_min = -15
    x_max = 15
    x_scale = (x_max - x_min) / 2
    x_mean = (x_max + x_min) / 2

    y_min = -148
    y_max = -128
    y_scale = (y_max - y_min) / 2
    y_mean = (y_max + y_min) / 2

    x = actions[0] * x_scale + x_mean
    y = actions[1] * y_scale + y_mean

    # d = min(max(1.0, actions[3]), 0.0)
    d = max(min(0.8, actions[3]), 0.2)
    d = d * np.sqrt(y_scale ** 2 + x_scale ** 2)
    # d = min(max(15.0, d), 2.0)

    x = constraint(x, x_min, x_max)
    y = constraint(y, y_min, y_max)

    return x, y, d


# RL - route_penalty
# check if the ped spwan position is too close to the route
def check_route_distance(actions, route):
    # route : list of numpy waypoints
    x_min = -15
    x_max = 15
    x_scale = (x_max - x_min) / 2
    x_mean = (x_max + x_min) / 2

    y_min = -148
    y_max = -128
    y_scale = (y_max - y_min) / 2
    y_mean = (y_max + y_min) / 2

    x = actions[0] * x_scale + x_mean
    y = actions[1] * y_scale + y_mean

    x = constraint(x, x_min, x_max)
    y = constraint(y, y_min, y_max)

    distance = np.inf
    for r_i in range(len(route)):
        r_x = route[r_i, 0]
        r_y = route[r_i, 1]
        dist_ = ((x - r_x) ** 2 + (y - r_y) ** 2) ** 0.5

        distance = dist_ if dist_ < distance else distance

    penalty_threshold = 2
    if distance < penalty_threshold:
        return True
    else:
        return False


# RL -- mian -- normalize the route
def normalize_routes(routes):
    mean_x = np.mean(routes[:, 0:1])
    max_x = np.max(np.abs(routes[:, 0:1]))
    x_1_2 = (routes[:, 0:1] - mean_x) / max_x

    mean_y = np.mean(routes[:, 1:2])
    max_y = np.max(np.abs(routes[:, 1:2]))
    y_1_2 = (routes[:, 1:2] - mean_y) / max_y

    route = np.concatenate([x_1_2, y_1_2], axis=0)
    return route


def wrapToPi(a):
    # Wraps the input angle to between 0 and pi
    return (a + np.pi) % (2 * np.pi) - np.pi


# CarlaEnv distance btw ego_vehicle and pedestrian
# TODO: NEED TO CHECK DISTANCE AND TRIGGER DISTANCE SCALE LATER
def ego_ped_distance(vehicle, pedestrian):
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

    relative_distance = np.sqrt((ego_x - ped_x) ** 2 + (ego_y - ped_y) ** 2 + (ego_z - ped_z) ** 2)
    # print('Distance between ego and ped:', self.relative_distance)

    return relative_distance
