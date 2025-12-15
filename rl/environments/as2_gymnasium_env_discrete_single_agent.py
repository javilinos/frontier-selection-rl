import subprocess
import os

import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from nav_msgs.msg import Path
from std_srvs.srv import SetBool, Empty

import gymnasium as gym
from gymnasium.spaces import Discrete

from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from typing import Any, List, Type
import math
import numpy as np
from transforms3d.euler import euler2quat
import random
import time
from copy import deepcopy

import xml.etree.ElementTree as ET

from .observation.observation import MultiChannelImageObservationWithFrontierFeatures as Observation
from .action.action import DiscreteCoordinateAction as Action
# from frontiers import get_frontiers, paint_frontiers


class AS2GymnasiumEnv(VecEnv):

    def __init__(self, world_name, world_size, grid_size, min_distance, num_envs, policy_type: str, testing: bool = False) -> None:
        # ROS 2 related stuff
        self.drone_interface_list = [
            DroneInterfaceTeleop(drone_id=f"drone{n}", use_sim_time=True)
            for n in range(num_envs)
        ]
        self.set_pose_client = self.drone_interface_list[0].create_client(
            SetPoseWithID, f"/world/{world_name}/set_pose"
        )
        self.world_control_client = self.drone_interface_list[0].create_client(
            ControlWorld, f"/world/{world_name}/control"
        )

        self.activate_scan_srv = self.drone_interface_list[0].create_client(
            SetBool, f"{self.drone_interface_list[0].get_namespace()}/activate_scan_to_occ_grid"
        )

        self.clear_map_srv = self.drone_interface_list[0].create_client(
            Empty, "/map_server/clear_map"
        )

        if testing:
            self.rotate_srv = self.drone_interface_list[0].create_client(
                SetBool, f"{self.drone_interface_list[0].get_namespace()}/rotate_in_place")

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
        self.action_manager = Action(self.drone_interface_list, self.grid_size)
        action_space = self.action_manager.action_space

        super().__init__(num_envs, observation_space, action_space)

        self.keys = self.observation_manager.keys
        self.buf_obs = self.observation_manager.buf_obs
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.testing = testing
        # Make a drone interface with functionality to control the internal state of the drone with rl env methods

        # Other stuff
        self.obstacles = self.parse_xml(f"assets/worlds/{world_name}.sdf")
        self.cum_path_length = []
        self.episode_path = []
        self.area_explored = 0
        self.total_path_length = 0
        self.path_length = 0
        print(self.obstacles)

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

    def set_random_pose(self, model_name, z=1.0) -> tuple[bool, Pose]:
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
        set_model_pose_req.pose.pose.position.z = z
        yaw = round(random.uniform(0, 2 * math.pi), 2)
        quat = euler2quat(0, 0, yaw)
        set_model_pose_req.pose.pose.orientation.x = quat[1]
        set_model_pose_req.pose.pose.orientation.y = quat[2]
        set_model_pose_req.pose.pose.orientation.z = quat[3]
        set_model_pose_req.pose.pose.orientation.w = quat[0]

        set_model_pose_res = self.set_pose_client.call(set_model_pose_req)
        # Return success and position
        return set_model_pose_res.success, set_model_pose_req.pose.pose

    def set_random_pose_with_cli(self, model_name):

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

        yaw = round(random.uniform(0, 2 * math.pi), 2)
        quat = euler2quat(0, 0, yaw)

        command = (
            '''gz service -s /world/''' + self.world_name + '''/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1000 -r "name: ''' +
            "'" + f'{model_name}' + "'" + ''', position: {x: ''' + str(x) + ''', y: ''' + str(y) +
            ''', z: ''' + str(1.0) + '''}, orientation: {x: 0, y: 0, z: 0, w: 1}"'''
        )
        print(command)

        pro = subprocess.Popen("exec " + command, stdout=subprocess.PIPE,
                               shell=True, preexec_fn=os.setsid)
        pro.communicate()

        pro.wait()
        pro.kill()
        # Return success and position
        return

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

    def set_pose_with_motion(self, path):
        path_to_follow = Path()
        path_to_follow.header.frame_id = "earth"
        for point in path:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "earth"
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 1.0
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_to_follow.poses.append(pose_stamped)
        self.drone_interface_list[0].follow_path.follow_path_with_keep_yaw(
            path_to_follow, speed=1.0)
        return

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
        self.obstacles = self.randomize_scenario()
        self.activate_scan_srv.call(SetBool.Request(data=False))
        # self.pause_physics()
        print("Resetting drone", self.drone_interface_list[env_idx].drone_id)
        self.set_random_pose(self.drone_interface_list[env_idx].drone_id)
        time.sleep(1.0)
        # self.unpause_physics()
        if self.testing:
            print('Arm')
            success = self.drone_interface_list[0].arm()
            print(f'Arm success: {success}')
            # Offboard
            print('Offboard')
            success = self.drone_interface_list[0].offboard()
            print(f'Offboard success: {success}')
            self.drone_interface_list[0].takeoff(1.0, 0.5)
        self.activate_scan_srv.call(SetBool.Request(data=True))
        self.clear_map_srv.call(Empty.Request())
        self.wait_for_map()
        if self.testing:
            self.rotate_srv.call(SetBool.Request(data=True))
            self.wait_for_map()
        frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
            env_idx)
        self.area_explored = discovered_area
        if len(frontiers) == 0:
            return self.reset_single_env(env_idx)
        obs = self._get_obs(env_idx)
        self._save_obs(env_idx, obs)
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
                self.observation_manager.frontiers, self.observation_manager.position_frontiers, idx)

            if not result:
                print("Failed to reach goal")
                self.buf_dones[idx] = False
                self.wait_for_map()
                frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
                    idx)
                if len(frontiers) == 0:  # No frontiers left, episode ends
                    self.buf_dones[idx] = True
                    self.area_explored = discovered_area
                    # self.buf_rews[idx] = 10.0
                    self.reset_single_env(idx)
                    break
            else:
                # old_map = np.copy(self.observation_manager.grid_matrix[0])
                # self.episode_path.append(nav_path)
                self.activate_scan_srv.call(SetBool.Request(data=False))
                if self.testing:
                    self.set_pose_with_motion(nav_path)
                    self.rotate_srv.call(SetBool.Request(data=True))
                else:
                    self.set_pose(drone.drone_id, frontier[0], frontier[1])
                self.activate_scan_srv.call(SetBool.Request(data=True))
                self.wait_for_map()

                frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
                    idx)

                obs = self._get_obs(idx)
                self._save_obs(idx, obs)
                self.buf_infos[idx] = {"nav_path": nav_path}

                max_distance = math.sqrt((self.world_size * 2)**2 + (self.world_size * 2)**2)

                self.buf_rews[idx] = -(self.path_length / max_distance)
                self.buf_dones[idx] = False
                self.total_path_length += self.path_length
                self.area_explored = discovered_area

                if len(frontiers) == 0:  # No frontiers left, episode ends
                    self.buf_dones[idx] = True
                    self.cum_path_length.append(self.total_path_length)
                    if not self.testing:
                        self.reset_single_env(idx)

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


if __name__ == "__main__":
    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world_low_density", world_size=10.0,
                          grid_size=200, min_distance=1.0, num_envs=1, policy_type="MultiInputPolicy")
    env.reset()
    env.wait_for_map()
    env.observation_manager._get_obs(0)
    for i in range(10):
        time.sleep(1.0)
        env.reset_single_env(0)
    env.drone_interface_list[0].shutdown()
    rclpy.shutdown()
