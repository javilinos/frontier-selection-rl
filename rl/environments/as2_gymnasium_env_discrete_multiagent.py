import subprocess
import os
from threading import BrokenBarrierError

import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_srvs.srv import SetBool, Empty
from std_msgs.msg import Bool

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

from observation import ObservationAsync as Observation
from action import DiscreteCoordinateActionSingleEnv as Action
from frontiers import get_frontiers, paint_frontiers


class AS2GymnasiumEnv(VecEnv):

    def __init__(self, world_name, world_size, grid_size, min_distance, num_envs, num_drones, policy_type: str,
                 env_index: int = 0,
                 shared_frontiers: list = None, lock=None, barrier_reset=None, barrier_step=None,
                 condition=None, queue=None, drones_initial_position=None, vec_sync: list = None,
                 step_lengths: list = None) -> None:
        # ROS 2 related stuff
        self.drone_interface_list = [
            DroneInterfaceTeleop(drone_id=f"drone{env_index}", use_sim_time=True)
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
            grid_size, num_envs, num_drones, env_index, self.drone_interface_list, policy_type)
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
        # Make a drone interface with functionality to control the internal state of the drone with rl env methods

        # Other stuff
        self.obstacles = self.parse_xml(f"assets/worlds/{world_name}.sdf")
        print(self.obstacles)
        self.env_index = env_index
        self.num_drones = num_drones

        self.shared_frontiers = shared_frontiers
        self.lock = lock
        self.barrier_reset = barrier_reset
        self.barrier_step = barrier_step
        self.condition = condition
        self.queue = queue

        self.drones_initial_position: List = drones_initial_position
        self.vec_sync = vec_sync
        self.step_lengths = step_lengths

        self._action_masks = np.zeros(self.action_manager.grid_size *
                                      self.action_manager.grid_size, dtype=bool)

        self.invalid_frontiers = []

        self.sync_step = True

    def pause_physics(self) -> bool:
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = True
        future = self.world_control_client.call_async(pause_physics_req)
        while rclpy.ok():
            if future.done():
                break
        return future.result().success

    def unpause_physics(self) -> bool:
        pause_physics_req = ControlWorld.Request()
        pause_physics_req.world_control.pause = False
        future = self.world_control_client.call_async(pause_physics_req)
        while rclpy.ok():
            if future.done():
                break
        return future.result().success

    def set_random_pose(self, model_name, drones: list[tuple[float, float]]) -> tuple[bool, Point]:
        set_model_pose_req = SetPoseWithID.Request()
        set_model_pose_req.pose.id = model_name
        x = round(random.uniform(-self.world_size / 1.5, self.world_size / 1.5), 2)
        y = round(random.uniform(-self.world_size / 1.5, self.world_size / 1.5), 2)
        drone_copy = []
        for drone in drones:
            drone_copy.append(drone)

        while True:
            too_close = any(
                self.distance((x, y), obstacle) < self.min_distance for obstacle in (self.obstacles + drone_copy)
            )
            if not too_close:
                break
            else:
                x = round(random.uniform(-self.world_size / 1.5, self.world_size / 1.5), 2)
                y = round(random.uniform(-self.world_size / 1.5, self.world_size / 1.5), 2)

        set_model_pose_req.pose.pose.position.x = x
        set_model_pose_req.pose.pose.position.y = y
        set_model_pose_req.pose.pose.position.z = 1.0
        yaw = round(random.uniform(0, 2 * math.pi), 2)
        quat = euler2quat(0, 0, yaw)
        set_model_pose_req.pose.pose.orientation.x = 0.0
        set_model_pose_req.pose.pose.orientation.y = 0.0
        set_model_pose_req.pose.pose.orientation.z = 0.0
        set_model_pose_req.pose.pose.orientation.w = 1.0

        success = False
        future = self.set_pose_client.call_async(request=set_model_pose_req)
        while rclpy.ok():
            if future.done():
                if future.result() is not None:
                    success = future.result().success
                break
        # Return success and position
        return success, set_model_pose_req.pose.pose.position

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

        # set_model_pose_res = self.set_pose_client.call(request=set_model_pose_req)
        success = False
        future = self.set_pose_client.call_async(request=set_model_pose_req)
        while rclpy.ok():
            if future.done():
                if future.result() is not None:
                    success = future.result().success
                break
        # Return success and position
        return success, set_model_pose_req.pose.pose

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
        self.barrier_reset.wait()
        self.barrier_step.reset()
        print("Resetting drone", self.drone_interface_list[env_idx].drone_id)
        self.sync_step = True

        if self.env_index == 0:
            while not self.queue.empty():
                self.queue.get()

        print("queue cleared")
        self.lock.acquire(timeout=5)
        try:
            future = self.activate_scan_srv.call_async(SetBool.Request(data=False))
            while rclpy.ok():
                if future.done():
                    break
        finally:
            self.lock.release()

        print("scan deactivated")
        self.barrier_reset.wait()

        if self.env_index == 0:
            self.clear_map_srv.call_async(Empty.Request())
            while rclpy.ok():
                if future.done():
                    break

        print("map cleared")
        self.lock.acquire(timeout=5)
        try:
            _, position = self.set_random_pose(
                self.drone_interface_list[env_idx].drone_id, self.drones_initial_position)
            print("lo intenta al menos")

            print("entra aqui al menos")
            self.drones_initial_position.append((position.x, position.y))
            if len(self.drones_initial_position) == self.num_drones:
                for initial_position in self.drones_initial_position:
                    self.drones_initial_position.remove(initial_position)
            for _ in self.shared_frontiers:
                self.shared_frontiers.pop(0)
        finally:
            self.lock.release()
        # with self.lock:
        #     self.drones_initial_position.append((position.x, position.y))

        #     if len(self.drones_initial_position) == self.num_drones:
        #         for initial_position in self.drones_initial_position:
        #             self.drones_initial_position.remove(initial_position)

        #     for _ in self.shared_frontiers:
        #         self.shared_frontiers.pop(0)

        print("Drone", self.drone_interface_list[env_idx].drone_id, " Half reset done")

        self.lock.acquire(timeout=5)
        future = self.activate_scan_srv.call_async(SetBool.Request(data=True))
        try:
            while rclpy.ok():
                if future.done():
                    break
        finally:
            self.lock.release()

        self.barrier_reset.wait()

        print("Drone", self.drone_interface_list[env_idx].drone_id, " Waiting for map")
        self.wait_for_map()

        print("Drone", self.drone_interface_list[env_idx].drone_id, " Getting frontiers")

        frontiers, position_frontiers = self.observation_manager.get_frontiers_and_position(
            self.env_index)

        print("Drone", self.drone_interface_list[env_idx].drone_id, " Frontiers gotten")

        obs = self._get_obs(env_idx)

        print("Drone", self.drone_interface_list[env_idx].drone_id, " Observations gotten")

        self._save_obs(env_idx, obs)
        print("Drone", self.drone_interface_list[env_idx].drone_id, " Calculating action masks")
        self.calculate_action_masks()

        self.barrier_reset.wait()

        print("Reset done: ", self.drone_interface_list[env_idx].drone_id)

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
            print(f"taking action drone {drone.drone_id}")
            frontier, position_frontier, path_length, path, result = self.action_manager.take_action(
                self.observation_manager.frontiers, self.observation_manager.position_frontiers, 0)

            if not result:
                print(f"Failed to reach goal (Invalid action) for drone {drone.drone_id}")
                self.buf_rews[idx] = -1.0
                self.wait_for_map()

                frontiers, position_frontiers = self.observation_manager.get_frontiers_and_position(
                    self.env_index)

                self.calculate_action_masks()

                self.invalid_frontiers.append(position_frontier)

                self.add_frontier_to_action_mask(self.invalid_frontiers)

                obs = self._get_obs(idx)
                self._save_obs(idx, obs)
                self.buf_infos[idx] = {}  # TODO: Add info
                self.buf_dones[idx] = False

                # Checks if all actions are masked, if so, ends the episode for that drone
                self.check_end_episode_cond(idx, drone)

                return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

            self.invalid_frontiers = []

            future = self.activate_scan_srv.call_async(SetBool.Request(data=False))
            while rclpy.ok():
                if future.done():
                    break

            with self.lock:
                if position_frontier not in self.shared_frontiers:
                    self.shared_frontiers.append(
                        position_frontier)

                print("drone", drone.drone_id, " shared frontiers: ", self.shared_frontiers)
                self.step_lengths[self.env_index] = len(path)

            if self.sync_step:
                print("queue empty")
                self.barrier_step.wait()

            self.sync_step = False

            with self.condition:
                self.condition.notify_all()

            while True:
                try:
                    self.barrier_step.wait()
                except BrokenBarrierError as e:
                    print("Barrier broken")

                print("Drone", drone.drone_id, " substep started")
                with self.condition:
                    self.condition.wait_for(lambda: min(self.step_lengths) != 0)
                length = min(self.step_lengths)
                print("Drone", drone.drone_id, " before set pose: length",
                      length, " path length: ", len(path))
                self.set_pose(drone.drone_id, path[length - 1][0], path[length - 1][1])
                print("Drone", drone.drone_id, " after set pose to ", path[length - 1])
                try:
                    self.barrier_step.wait()
                except BrokenBarrierError as e:
                    print("Barrier broken")
                print("Drone", drone.drone_id, " substep done")
                if all(step_length == self.step_lengths[0] for step_length in self.step_lengths):
                    break
                elif length == len(path):
                    print("Drone", drone.drone_id, " donete ")
                    with self.lock:
                        if not any((step_length == 0) for step_length in self.step_lengths):
                            for i in range(self.num_drones):
                                self.step_lengths[i] -= length
                    print("Drone", drone.drone_id, " donete despues")
                    break
                else:
                    path = path[length:]
                    with self.condition:
                        print(f"Drone {drone.drone_id} waiting for other drones")
                        self.condition.wait(timeout=5.0)
                        print(f"Drone {drone.drone_id} woke up")
            self.lock.acquire(timeout=5)
            try:
                self.wait_for_map()
                old_map = self.observation_manager.grid_matrix
                future = self.activate_scan_srv.call_async(SetBool.Request(data=True))
                then = time.time()
                while rclpy.ok():
                    if future.done():
                        if future.result() is not None:
                            activate_scan_res = future.result()
                            if activate_scan_res.success:
                                break
                    if time.time() - then > 1.0:
                        print(f"Drone{self.env_index} service call timeout, calling again...")
                        future = self.activate_scan_srv.call_async(
                            SetBool.Request(data=True))
                        then = time.time()

                self.wait_for_map()
                new_map = self.observation_manager.grid_matrix
            finally:
                self.lock.release()

            print("Drone", drone.drone_id, " pre remove frontiers")
            with self.lock:
                if position_frontier in self.shared_frontiers:
                    self.shared_frontiers.remove(position_frontier)
            print("Drone", drone.drone_id, " pre wait for map")

            frontiers, position_frontiers = self.observation_manager.get_frontiers_and_position(
                self.env_index)

            obs = self._get_obs(idx)
            self._save_obs(idx, obs)

            max_distance = math.sqrt((self.world_size * 2)**2 + (self.world_size * 2)**2)

            max_area = self.grid_size * self.grid_size

            discovered_area_rew = (np.sum(old_map == self.observation_manager.UNKNOWN) -
                                   np.sum(new_map == self.observation_manager.UNKNOWN)
                                   ) / max_area  # Reward based on discovered area

            print(f"drone {drone.drone_id} discovered area reward: {discovered_area_rew}")

            path_length_rew = -(path_length / max_distance)  # Reward based on path length

            distance_to_closest_drone_rew = max_distance

            for drone_id, drone_position in self.observation_manager.swarm_position.items():
                if drone_id != drone.drone_id:
                    dist = self.distance((drone_position[0], drone_position[1]),
                                         (drone.position[0], drone.position[1]))
                    if dist < distance_to_closest_drone_rew:
                        distance_to_closest_drone_rew = dist

            distance_to_closest_drone_rew = distance_to_closest_drone_rew / \
                max_distance  # Reward based on distance to closest drone

            self.buf_infos[idx] = {}  # TODO: Add info
            self.buf_rews[idx] = path_length_rew + \
                distance_to_closest_drone_rew + discovered_area_rew
            self.buf_dones[idx] = False

            # with self.lock:
            #     index = next((i for i, _ in enumerate(self.shared_frontiers)
            #                   if self.shared_frontiers[i][0] == position_frontier[0] and self.shared_frontiers[i][1] == position_frontier[1]), None)
            #     print("Index", index)
            #     self.shared_frontiers.pop(index)
            #     print("Shared frontiers", self.shared_frontiers)
            #     index

            # with self.lock:
            #     for shared_frontier in self.shared_frontiers:
            #         if shared_frontier in position_frontiers:
            #             position_frontiers.remove(shared_frontier)

            for frontier in self.observation_manager.position_frontiers:
                if frontier in self.shared_frontiers:
                    print("Frontier already chosen")
                    index = self.observation_manager.position_frontiers.index(frontier)
                    self.observation_manager.position_frontiers.pop(index)
                    self.observation_manager.frontiers.pop(index)

            self.calculate_action_masks()

            self.check_end_episode_cond(idx, drone)

            print("Drone", drone.drone_id, " step done")

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

    def check_end_episode_cond(self, idx, drone: DroneInterfaceTeleop):
        if all(self._action_masks == False):
            print(f"Drone{self.drone_interface_list[0].drone_id} All actions masked")
            print(self.observation_manager.position_frontiers)

            self.buf_dones[idx] = True

            self.lock.acquire(timeout=5)
            try:
                if not self.queue.full():
                    self.queue.put(drone.drone_id)
            finally:
                self.lock.release()

            self.lock.acquire(timeout=5)
            try:
                self.step_lengths[self.env_index] = 10000  # Arbitrary large number
            finally:
                self.lock.release()

            if not self.barrier_step.broken:
                self.barrier_step.abort()

            with self.condition:
                self.condition.notify_all()

            print("Drone", drone.drone_id, " before reset")
            self.reset_single_env(idx)

            print("Drone", drone.drone_id, "done")

    def calculate_action_masks(self):
        self._action_masks = np.zeros(self.action_manager.grid_size *
                                      self.action_manager.grid_size, dtype=bool)
        self.lock.acquire(timeout=5)
        try:
            print("in action masking, drone", self.env_index,
                  " shared frontiers: ", self.shared_frontiers)
            print("in action masking, drone", self.env_index,
                  " position frontiers: ", self.observation_manager.position_frontiers)
            for frontier in self.observation_manager.position_frontiers:
                if (0 > frontier[0] > self.grid_size - 1) or (0 > frontier[1] > self.grid_size - 1):
                    print("Frontier out of bounds")
                    self.observation_manager.position_frontiers.remove(frontier)
                else:
                    if frontier in self.shared_frontiers:
                        print("Frontier already chosen")
                        self._action_masks[frontier[1] *
                                           self.action_manager.grid_size + frontier[0]] = False
                    else:
                        self._action_masks[frontier[1] *
                                           self.action_manager.grid_size + frontier[0]] = True
            print("calculated action masks")
        finally:
            self.lock.release()
            print("released lock")

    def add_frontier_to_action_mask(self, frontiers):
        for frontier in frontiers:
            self._action_masks[frontier[1] * self.action_manager.grid_size + frontier[0]] = False

    def action_masks(self):
        # action_masks = np.zeros(self.action_manager.grid_size *
        #                         self.action_manager.grid_size, dtype=bool)
        # with self.lock:
        #     for frontier in self.observation_manager.position_frontiers:
        #         if frontier in self.shared_frontiers:
        #             print("Frontier already chosen")
        #             action_masks[frontier[1] * self.action_manager.grid_size + frontier[0]] = False
        #         else:
        #             action_masks[frontier[1] * self.action_manager.grid_size + frontier[0]] = True
        # if all(action_masks == False):
        #     print(f"Drone{self.drone_interface_list[0].drone_id} All actions masked")
        #     print(self.observation_manager.position_frontiers)
        return self._action_masks

    # def sync_reset(self, phase: int = 0):
    #     release = False
    #     while not release:
    #         with self.lock:
    #             if self.env_index == 0:
    #                 self.vec_sync[phase][0] = True
    #             if self.env_index == 1 and self.vec_sync[phase][0]:
    #                 self.vec_sync[phase][1] = True
    #             if self.env_index == 2 and self.vec_sync[phase][1]:
    #                 self.vec_sync[phase][2] = True
    #             if self.env_index == 3 and self.vec_sync[phase][2]:
    #                 self.vec_sync[phase][3] = True
    #             release = all(self.vec_sync[phase])
    #         time.sleep(0.1)

    # def sync_step(self, phase: int = 3):
    #     release = False
    #     while not release:
    #         with self.lock:
    #             if self.env_index == 0:
    #                 self.vec_sync[phase][0] = True
    #             if self.env_index == 1 and self.vec_sync[phase][0]:
    #                 self.vec_sync[phase][1] = True
    #             if self.env_index == 2 and self.vec_sync[phase][1]:
    #                 self.vec_sync[phase][2] = True
    #             if self.env_index == 3 and self.vec_sync[phase][2]:
    #                 self.vec_sync[phase][3] = True
    #             release = all(self.vec_sync[phase])
    #         time.sleep(0.1)

    # def reset_reset_syncers(self):
    #     for i in range(3):
    #         for j in range(self.num_drones):
    #             self.vec_sync[i][j] = False
    #     return

    # def reset_step_syncers(self):
    #     for i in range(3, 5):
    #         for j in range(self.num_drones):
    #             self.vec_sync[i][j] = False
    #     return


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
