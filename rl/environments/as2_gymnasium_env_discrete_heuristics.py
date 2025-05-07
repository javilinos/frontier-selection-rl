import subprocess
import os

import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from std_srvs.srv import SetBool, Empty

import gymnasium as gym
from gymnasium.spaces import Discrete

from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from typing import Any, List, Type
import math
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2quat
import random
import time
from copy import deepcopy

import xml.etree.ElementTree as ET
import pandas as pd

import argparse

from observation.observation import MultiChannelImageObservationWithFrontierFeatures as Observation
from action.heuristic_action import NearestFrontierAction, HybridAction, RandomAction, TAREAction


class AS2GymnasiumEnv(VecEnv):

    def __init__(self, world_name, world_size, grid_size, min_distance, num_envs, policy_type: str, method: str) -> None:
        # ROS 2 related stuff
        self.drone_interface_list = [
            DroneInterfaceTeleop(drone_id=f"drone{n}", use_sim_time=True)
            for n in range(num_envs)
        ]
        self.set_pose_client = self.drone_interface_list[0].create_client(
            SetPoseWithID, f"/world/{world_name}/set_pose"
        )
        self.set_pose_client.wait_for_service(timeout_sec=5.0)
        self.world_control_client = self.drone_interface_list[0].create_client(
            ControlWorld, f"/world/{world_name}/control"
        )

        self.activate_scan_srv = self.drone_interface_list[0].create_client(
            SetBool, f"{self.drone_interface_list[0].get_namespace()}/activate_scan_to_occ_grid"
        )

        self.clear_map_srv = self.drone_interface_list[0].create_client(
            Empty, "/map_server/clear_map"
        )

        self.render_mode = []
        for _ in range(num_envs):
            self.render_mode.append(["rgb_array"])

        self.world_name = world_name
        self.world_size = world_size
        self.min_distance = min_distance
        self.grid_size = grid_size

        # Environment observation
        self.observation_manager = Observation(
            grid_size, num_envs, self.drone_interface_list, policy_type)
        observation_space = self.observation_manager.observation_space

        # Environment action
        if method == "nearest":
            self.action_manager = NearestFrontierAction(
                self.drone_interface_list, self.grid_size)
        elif method == "random":
            self.action_manager = RandomAction(
                self.drone_interface_list, self.grid_size)
        elif method == "hybrid":
            self.action_manager = HybridAction(
                self.drone_interface_list, self.grid_size)
        elif method == "tare":
            self.action_manager = TAREAction(
                self.drone_interface_list, self.grid_size)
        else:
            raise ValueError("Invalid method")
        # self.action_manager = Action(self.drone_interface_list, self.grid_size)
        action_space = self.action_manager.action_space

        super().__init__(num_envs, observation_space, action_space)

        self.keys = self.observation_manager.keys
        self.buf_obs = self.observation_manager.buf_obs
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        # Make a drone interface with functionality to control the internal state of the drone with rl env methods

        # Other stuff
        self.obstacles = self.parse_xml(f"assets/worlds/{world_name}.sdf")
        self.cum_path_length = []
        self.episode_path = []
        self.area_explored = 0
        self.total_path_length = 0
        self.path_length = 0

    def pause_physics(self) -> bool:
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = True
        pause_physics_res = self.world_control_client.call(pause_physics_req)
        return pause_physics_res.success

    def unpause_physics(self) -> bool:
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = False
        pause_physics_res = self.world_control_client.call(pause_physics_req)
        return pause_physics_res.success

    def set_random_pose(self, model_name) -> tuple[bool, Pose]:
        set_model_pose_req = SetPoseWithID.Request()
        set_model_pose_req.pose.id = model_name
        x = round(random.uniform(-self.world_size, self.world_size), 2)
        y = round(random.uniform(-self.world_size, self.world_size), 2)
        while True:
            too_close = any(
                self.distance((x, y), obstacle) < self.min_distance for obstacle in self.obstacles
            )
            if not too_close:
                break
            else:
                x = round(random.uniform(-self.world_size, self.world_size), 2)
                y = round(random.uniform(-self.world_size, self.world_size), 2)

        set_model_pose_req.pose.pose.position.x = x
        set_model_pose_req.pose.pose.position.y = y
        set_model_pose_req.pose.pose.position.z = 1.0
        yaw = round(random.uniform(0, 2 * math.pi), 2)
        quat = euler2quat(0, 0, yaw)
        set_model_pose_req.pose.pose.orientation.x = quat[1]
        set_model_pose_req.pose.pose.orientation.y = quat[2]
        set_model_pose_req.pose.pose.orientation.z = quat[3]
        set_model_pose_req.pose.pose.orientation.w = quat[0]

        set_model_pose_res = self.set_pose_client.call(set_model_pose_req)
        # Return success and position
        return set_model_pose_res.success, set_model_pose_req.pose.pose

    def set_pose(self, model_name, x, y) -> tuple[bool, Pose]:
        set_model_pose_req = SetPoseWithID.Request()
        set_model_pose_req.pose.id = model_name
        set_model_pose_req.pose.pose.position.x = x
        set_model_pose_req.pose.pose.position.y = y
        set_model_pose_req.pose.pose.position.z = 1.0
        set_model_pose_req.pose.pose.orientation.x = 0.0
        set_model_pose_req.pose.pose.orientation.y = 0.0
        set_model_pose_req.pose.pose.orientation.z = 0.0
        set_model_pose_req.pose.pose.orientation.w = 1.0

        set_model_pose_res = self.set_pose_client.call(set_model_pose_req)
        # Return success and position
        return set_model_pose_res.success, set_model_pose_req.pose.pose

    def set_pose_with_cli(self, model_name, x, y):

        command = (
            '''gz service -s /world/''' + self.world_name + '''/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1000 -r "name: ''' +
            "'" + f'{model_name}' + "'" + ''', position: {x: ''' + str(x) + ''', y: ''' + str(y) +
            ''', z: ''' + str(1.0) + '''}, orientation: {x: 0, y: 0, z: 0, w: 1}"'''
        )

        pro = subprocess.Popen("exec " + command, stdout=subprocess.PIPE,
                               shell=True, preexec_fn=os.setsid)
        pro.communicate()

        pro.wait()
        pro.kill()
        # Return success and position
        return

    def randomize_scenario(self):
        models = []
        for i in range(len(self.obstacles)):
            x = round(random.uniform(-self.world_size, self.world_size), 2)
            y = round(random.uniform(-self.world_size, self.world_size), 2)
            self.set_pose(f"pole{i + 1}", x, y)
            models.append((x, y))
        return models

    def reset_single_env(self, env_idx):
        self.total_path_length = 0
        self.area_explored = 0
        self.activate_scan_srv.call(SetBool.Request(data=False))
        self.pause_physics()
        self.clear_map_srv.call(Empty.Request())
        self.obstacles = self.randomize_scenario()
        print("Resetting drone", self.drone_interface_list[env_idx].drone_id)
        self.set_random_pose(self.drone_interface_list[env_idx].drone_id)
        # self.set_pose(self.drone_interface_list[env_idx].drone_id, 0.0, 0.0)
        self.unpause_physics()
        self.clear_map_srv.call(Empty.Request())
        time.sleep(1.0)
        self.activate_scan_srv.call(SetBool.Request(data=True))
        self.wait_for_map()
        # self.observation_manager.call_get_frontiers_with_msg(env_id=env_idx)
        # while self.observation_manager.wait_for_frontiers == 0:
        #     pass
        frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
            env_idx)
        self.area_explored = discovered_area
        if len(frontiers) == 0:
            return self.reset_single_env(env_idx)
        obs = self._get_obs(env_idx)
        self._save_obs(env_idx, obs)
        self.reset_counter = 0
        return obs

    def reset(self, **kwargs) -> VecEnvObs:
        for idx, _ in enumerate(self.drone_interface_list):
            self.reset_single_env(idx)
        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        self.action_manager.actions = actions

    def step_wait(self) -> None:
        for idx, drone in enumerate(self.drone_interface_list):
            # self.action_manager.actions = self.action_manager.generate_random_action()
            frontier, self.path_length, result, nav_path = self.action_manager.take_action(
                self.observation_manager.frontiers, self.observation_manager.position_frontiers, idx, self.observation_manager.grid_matrix)
            # Go to closest frontier

            while not result:
                print("Failed to reach goal")
                self.buf_dones[idx] = False
                self.wait_for_map()
                frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
                    idx)
                frontier_index = self.observation_manager.frontiers.index(frontier)
                self.observation_manager.frontiers.pop(frontier_index)
                self.observation_manager.position_frontiers.pop(frontier_index)
                frontier, self.path_length, result, nav_path = self.action_manager.take_action(
                    self.observation_manager.frontiers, self.observation_manager.position_frontiers, idx, self.observation_manager.grid_matrix)
                if len(frontiers) == 0:  # No frontiers left, episode ends
                    self.buf_dones[idx] = True
                    self.cum_path_length.append(self.total_path_length)
                    self.area_explored = discovered_area
                    # self.buf_rews[idx] = 10.0
                    self.reset_single_env(idx)
                    break

            self.episode_path.append(nav_path)
            self.activate_scan_srv.call(SetBool.Request(data=False))
            self.set_pose(drone.drone_id, frontier[0], frontier[1])
            self.activate_scan_srv.call(SetBool.Request(data=True))
            self.wait_for_map()
            self.wait_for_map()
            frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
                idx)
            obs = self._get_obs(idx)
            self._save_obs(idx, obs)
            self.buf_infos[idx] = {}  # TODO: Add info

            max_distance = math.sqrt((self.world_size * 2)**2 + (self.world_size * 2)**2)

            self.buf_rews[idx] = -(self.path_length / max_distance)
            self.buf_dones[idx] = False
            self.total_path_length += self.path_length
            self.area_explored = discovered_area
            print(self.area_explored)
            if len(frontiers) == 0:  # No frontiers left, episode ends
                self.buf_dones[idx] = True
                self.cum_path_length.append(self.total_path_length)
                # self.buf_rews[idx] = 10.0
                # self.reset_single_env(idx)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def close(self):
        return

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        try:
            return getattr(self, attr_name)
        except AttributeError:
            return None

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        return

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ):
        if method_name == "action_masks":
            return self.action_masks()
        return

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return

    def _obs_from_buf(self) -> VecEnvObs:
        # return all observations from all environments
        return self.observation_manager._obs_from_buf()

    def _save_obs(self, env_id, obs: VecEnvObs):
        # save the observation for the specified environment
        self.observation_manager._save_obs(obs, env_id)

    def _get_obs(self, env_id) -> VecEnvObs:
        # get the observation for the specified environment
        return self.observation_manager._get_obs(env_id)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        return [True] * self.num_envs

    def parse_xml(self, filename: str) -> List[tuple[float, float]]:
        """Parse XML file and return pole positions"""
        world_tree = ET.parse(filename).getroot()
        models = []
        for model in world_tree.iter('include'):
            if model.find('uri').text == 'model://pole':
                x, y, *_ = model.find('pose').text.split(' ')
                models.append((float(x), float(y)))

        return models

    def distance(self, point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """
        Calculate the euclidean distance between 2 points
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def wait_for_map(self):
        self.observation_manager.wait_for_map = 0
        while self.observation_manager.wait_for_map == 0:
            pass
        return

    def action_masks(self):
        return self.observation_manager.get_action_mask(0)

    def frontier_features(self):
        frontier_features = self.observation_manager.get_frontier_features()
        self.action_space = Discrete(len(frontier_features))
        return frontier_features


def fill_data(matrix):
    '''Grab a list of lists and make all the lists the same length, given the longest list, by filling with 1.0'''
    max_len = max(len(lst) for lst in matrix)
    for lst in matrix:
        while len(lst) < max_len:
            lst.append(1.0)


def get_std_dev(matrix):
    '''Grab a list of lists and return the standard deviation of the elements of each index in a new list'''
    std_vector = []
    for i in range(len(matrix[0])):
        column = [lst[i] for lst in matrix]
        std_vector.append(np.std(column))

    return std_vector


def get_mean(matrix):
    '''Grab a list of lists and return the mean of the elements of each index in a new list'''
    mean_vector = []
    for i in range(len(matrix[0])):
        column = [lst[i] for lst in matrix]
        mean_vector.append(np.mean(column))

    return mean_vector


def plot_path(obstacles, paths):
    GRID_SIZE = 20
    HALF = GRID_SIZE / 2

    fig, ax = plt.subplots(figsize=(6, 6))

    # 1) major ticks only at -10, 0, +10
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])

    # 2) minor ticks at every integer
    ax.set_xticks(range(-int(HALF), int(HALF) + 1), minor=True)
    ax.set_yticks(range(-int(HALF), int(HALF) + 1), minor=True)

    # 3) draw minor-grid (dashed) and major-grid (solid)
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.grid(which='major', linestyle='-', linewidth=1)

    # 4) hide all spines (no painted axis lines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 5) label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 6) plot obstacles (smaller circles)
    xo, yo = zip(*obstacles)
    ax.scatter(xo, yo,
               s=50,
               color='black',
               zorder=3)

    # 7) plot paths: blue, thicker, darker overlaps
    for path in paths:
        xs, ys = zip(*path)
        ax.plot(xs, ys,
                color='blue',
                linewidth=5,
                alpha=0.3,
                zorder=1)

    # 8) enforce aspect & limits
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform testing of heuristic methods")
    parser.add_argument(
        "--num_episodes", type=int, default=10,
        help="Number of episodes to test on"
    )
    parser.add_argument(
        "--method", type=str, default="nearest",
        help="Method to test on", choices=["nearest", "random", "hybrid", "tare"], required=True
    )
    parser.add_argument(
        "--world_name", type=str, default="world_low_density",
        help="World name to test on"
    )

    parser.add_argument(
        "--plot_path", type=bool, default=False,
        help="Plot the path taken by the drone"
    )

    args = parser.parse_args()

    rclpy.init()
    env = AS2GymnasiumEnv(world_name=args.world_name, world_size=10.0,
                          grid_size=200, min_distance=1.0, num_envs=1, policy_type="MultiInputPolicy", method=args.method)
    env.reset()
    num_episodes = args.num_episodes
    episodes = list(range(1, num_episodes + 1))
    episode_count = 0
    ###### matrix initialization #######
    area_explored_matrix = []
    cum_path_length_matrix = []
    # _rewards = []
    step_path_length = []
    area_explored = []
    std_vector = []
    path_length_per_episode = []
    episode_reward = 0.0
    step_path_length.append(0)
    area_explored.append(env.area_explored)

    for _ in range(num_episodes):
        done = False
        while not done:
            observations, rewards, dones, infos = env.step_wait()
            # episode_reward += rewards[0]
            done = dones[0]
            step_path_length.append(env.path_length)
            area_explored.append(env.area_explored)
        # _rewards.append(episode_reward)++
        # episode_reward = 0.0
        area_explored_matrix.append(area_explored)
        accumulated_path = np.cumsum(step_path_length)
        cum_path_length_matrix.append(accumulated_path)
        episode_path_length = np.sum(step_path_length)
        path_length_per_episode.append(episode_path_length)
        path_to_plot = env.episode_path
        env.reset()
        step_path_length = []
        area_explored = []
        episode_reward = 0.0
        step_path_length.append(0)
        area_explored.append(env.area_explored)
        episode_count += 1
        print("Episode:", episode_count)
    # print("Mean reward:", np.mean(_rewards))
    # Save to CSV
    if args.plot_path:
        plot_path(env.obstacles, path_to_plot)

    fill_data(area_explored_matrix)
    mean_area_explored = get_mean(area_explored_matrix)
    std_area_explored = get_std_dev(area_explored_matrix)
    # just return the biggest length list in the matrix cum_path_length_matrix
    distance = max(cum_path_length_matrix, key=len)

    df = pd.DataFrame({
        'distance': distance,
        'mean_area_explored': mean_area_explored,
        'std_area_explored': std_area_explored
    })
    df2 = pd.DataFrame({
        'episode': episodes,
        'path_length': path_length_per_episode
    })

    # df.to_csv('csv/time_graphics_10_episodes/nearest.csv', index=False)
    # df2.to_csv('csv/bars_graphic_cum_mean_path_length/enormous_density/nearest.csv', index=False)

    # # Optional: plot the data
    fig, ax = plt.subplots()

    # Plot mean (darkest color)
    ax.plot(df['distance'], df['mean_area_explored'],
            color='blue', alpha=1.0, label='Mean Area Explored')

    # Plot standard deviation as transparent band
    ax.fill_between(
        df['distance'],
        df['mean_area_explored'] - df['std_area_explored'],
        df['mean_area_explored'] + df['std_area_explored'],
        color='blue', alpha=0.3, label='Std Dev'
    )

    ax.set_xlabel('Distance')
    ax.set_ylabel('Area Explored')
    ax.set_title('Mean Area Explored with Standard Deviation')
    ax.legend()

    plt.show()

    env.drone_interface_list[0].shutdown()
    rclpy.shutdown()
