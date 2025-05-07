import subprocess
import os

import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from std_srvs.srv import SetBool, Empty

import gymnasium as gym

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

from observation import Observation
from action import ActionCartessianCoordinateVector as Action
from frontiers import get_frontiers, paint_frontiers


class AS2GymnasiumEnv(VecEnv):

    def __init__(self, world_name, world_size, grid_size, min_distance, num_envs, policy_type: str) -> None:
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
        self.action_manager = Action(self.drone_interface_list)
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
        self.obstacles = self.parse_xml("assets/worlds/world1.sdf")
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

    def reset_single_env(self, env_idx):
        self.activate_scan_srv.call(SetBool.Request(data=False))
        self.pause_physics()
        self.clear_map_srv.call(Empty.Request())
        print("Resetting drone", self.drone_interface_list[env_idx].drone_id)
        self.set_random_pose(self.drone_interface_list[env_idx].drone_id)

        self.unpause_physics()
        self.activate_scan_srv.call(SetBool.Request(data=True))
        self.wait_for_map()
        # self.observation_manager.call_get_frontiers_with_msg(env_id=env_idx)
        # while self.observation_manager.wait_for_frontiers == 0:
        #     pass
        frontiers, _ = self.observation_manager.get_frontiers_and_position(
            env_idx)
        if len(frontiers) == 0:
            return self.reset_single_env(env_idx)
        obs = self._get_obs(env_idx)
        self._save_obs(env_idx, obs)
        return obs

    def reset(self, seed=None, **kwargs) -> VecEnvObs:
        for idx, _ in enumerate(self.drone_interface_list):
            self.reset_single_env(idx)
        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        self.action_manager.actions = actions

    def step_wait(self) -> None:
        for idx, drone in enumerate(self.drone_interface_list):
            # self.action_manager.actions = self.action_manager.generate_random_action()
            frontier, path_length, closest_distance, result = self.action_manager.take_action(
                self.observation_manager.frontiers, self.world_size, idx)

            distance_reward = (0.5 - closest_distance / math.sqrt(2)) * 2

            self.set_pose(drone.drone_id, frontier[0], frontier[1])
            self.wait_for_map()
            # self.observation_manager.call_get_frontiers_with_msg(env_id=idx)
            # while self.observation_manager.wait_for_frontiers == 0:
            #     pass
            frontiers, _ = self.observation_manager.get_frontiers_and_position(
                idx)
            obs = self._get_obs(idx)
            self._save_obs(idx, obs)
            self.buf_infos[idx] = {}  # TODO: Add info
            self.buf_rews[idx] = -path_length * 0.1
            self.buf_rews[idx] += distance_reward
            self.buf_dones[idx] = False
            if len(frontiers) == 0:  # No frontiers left, episode ends
                self.buf_dones[idx] = True
                self.buf_rews[idx] = 100.0
                self.reset_single_env(idx)
            if not result:
                print("Failed to reach goal")
                self.buf_dones[idx] = True
                self.buf_rews[idx] = -100.0
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
    ) -> List[Any]:
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


if __name__ == "__main__":
    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world1", world_size=2.5,
                          grid_size=50, min_distance=1.0, num_envs=1, policy_type="MlpPolicy")
    while (True):
        env.observation_manager._get_obs(0)
    # print("Start mission")
    # #### ARM OFFBOARD #####
    # print("Arm")
    # env.drone_interface_list[0].offboard()
    # time.sleep(1.0)
    # print("Offboard")
    # env.drone_interface_list[0].arm()
    # time.sleep(1.0)

    # ##### TAKE OFF #####
    # print("Take Off")
    # env.drone_interface_list[0].takeoff(1.0, speed=1.0)
    # time.sleep(1.0)

    # env.reset()
    # for i in range(10):
    #     env.step_wait()
    #     # print('number of frontiers:', len(env.observation_manager.frontiers))
    #     # env.observation_manager.show_image_with_frontiers()
    #     # time.sleep(2.0)
    # rclpy.shutdown()
