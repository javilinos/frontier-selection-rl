import subprocess
import os
from threading import BrokenBarrierError

import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPoseWithID
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point
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

from .observation.observation import MultiChannelImageObservationWithFrontierFeaturesAsync as Observation
from .action.action import DiscreteFrontierIndexAction as Action
# from frontiers import get_frontiers, paint_frontiers


class AS2GymnasiumEnv(VecEnv):

    def __init__(self, world_name, world_size, grid_size, min_distance, num_envs, num_drones, policy_type: str,
                 env_index: int = 0,
                 shared_frontiers: list = None, lock=None, barrier_reset=None, barrier_step=None,
                 condition=None, queue=None, drones_initial_position=None, vec_sync: list = None,
                 step_lengths: list = None, rollout_remaining=None, testing: bool = False) -> None:
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
            grid_size, num_envs, self.drone_interface_list, policy_type, num_drones=num_drones)
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
        self.rollout_remaining = rollout_remaining

        self.cum_path_length = []
        self.episode_path = []
        self.invalid_frontiers = []
        self.area_explored = 0
        self.total_path_length = 0
        self.path_length = 0
        self.sync_step = True
        self.max_range_limit = 3.0
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
        set_model_pose_req.pose.pose.orientation.x = quat[1]
        set_model_pose_req.pose.pose.orientation.y = quat[2]
        set_model_pose_req.pose.pose.orientation.z = quat[3]
        set_model_pose_req.pose.pose.orientation.w = quat[0]

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
        self.barrier_reset.wait()
        # self.barrier_step.reset()
        self.barrier_step.increment()
        self.barrier_step.print_status()
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
            future = self.clear_map_srv.call_async(Empty.Request())
            while rclpy.ok():
                if future.done():
                    print("map cleared service done")
                    break

        print("map cleared")
        self.lock.acquire(timeout=5)
        try:
            _, position = self.set_random_pose(
                self.drone_interface_list[env_idx].drone_id, self.drones_initial_position)
            self.drones_initial_position.append((position.x, position.y))
            if len(self.drones_initial_position) == self.num_drones:
                for initial_position in self.drones_initial_position:
                    self.drones_initial_position.remove(initial_position)
            for _ in self.shared_frontiers:
                self.shared_frontiers.pop(0)
        finally:
            self.lock.release()

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

        frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
            self.env_index)
        self.area_explored = discovered_area

        print("Drone", self.drone_interface_list[env_idx].drone_id, " Frontiers gotten")

        # self.remove_shared_frontiers_from_observation()
        # if len(self.observation_manager.frontiers) == 0:
        #     return self.reset_single_env(env_idx)

        obs = self._get_obs(env_idx)
        self._save_obs(env_idx, obs)

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
            print(f"taking action drone {drone.drone_id}")
            frontier, path_length, result, path = self.action_manager.take_action(
                self.observation_manager.frontiers, idx)
            if not result:
                print(f"Failed to reach goal (Invalid action) for drone {drone.drone_id}")
                self.buf_dones[idx] = False
                self.wait_for_map()
                print(f"map stamp: {self.observation_manager.last_map_header_.stamp.nanosec}")
                frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
                    idx)
                self.invalid_frontiers.append(frontier)
                for invalid_frontier in self.invalid_frontiers:
                    try:
                        invalid_frontier_idx = self.observation_manager.frontiers.index(invalid_frontier)
                        self.observation_manager.frontiers.pop(invalid_frontier_idx)
                        self.observation_manager.position_frontiers.pop(invalid_frontier_idx)
                    except ValueError as e:
                        print(f"Drone {drone.drone_id} Frontier not found")

                print(f"Invalid frontier for drone {drone.drone_id}")
                
                obs = self._get_obs(idx)
                self._save_obs(idx, obs)
                self.buf_infos[idx] = {"distance": 0.0}  # TODO: Add info

                if not self.check_end_episode_cond(idx, drone):
                    self.buf_rews[idx] = -10.0
                    self.buf_dones[idx] = False
                else:
                    self.buf_rews[idx] = 0.0

                return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
            
            self.invalid_frontiers = []
            self.wait_for_map()
            future = self.activate_scan_srv.call_async(SetBool.Request(data=False))

            while rclpy.ok():
                if future.done():
                    break

            with self.lock:
                if frontier not in self.shared_frontiers:
                    self.shared_frontiers.append(
                        frontier)
                self.step_lengths[self.env_index] = len(path)

            # if self.sync_step:
            #     print("queue empty")
            # self.barrier_step.wait()

            self.sync_step = False

            with self.condition:
                self.condition.notify_all()  

            while True:
                try:
                    self.barrier_step.print_status()
                    self.barrier_step.wait()
                except BrokenBarrierError as e:
                    print("Barrier broken")
                print("Drone", drone.drone_id, " substep started")
                with self.condition:
                    self.condition.wait_for(
                        lambda: min(self.step_lengths) != 0
                    )
                    length = min(self.step_lengths)
                                       
                print("Drone", drone.drone_id, " before set pose: length",
                      length, " path length: ", len(path))
                self.set_pose(drone.drone_id, path[length - 1][0], path[length - 1][1])
                print("Drone", drone.drone_id, " after set pose to ", path[length - 1])
                try:
                    self.barrier_step.print_status()
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

            print(f"Drone {drone.drone_id} pre lock acquire")
            self.lock.acquire(timeout=5)
            print(f"Drone {drone.drone_id} post lock acquire")
            try:
                old_map = np.copy(self.observation_manager.grid_matrix[2])    
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
                new_map = np.copy(self.observation_manager.grid_matrix[2])
            finally:
                self.lock.release()

            print("Drone", drone.drone_id, " pre remove frontiers")
            with self.lock:
                try:
                    if frontier in self.shared_frontiers:
                        self.shared_frontiers.remove(frontier)
                except ValueError as e:
                    print(f"Drone {drone.drone_id} Frontier not found in shared frontiers")
                    
            print("Drone", drone.drone_id, " pre wait for map")

            frontiers, position_frontiers, discovered_area = self.observation_manager.get_frontiers_and_position(
                self.env_index)
            
            for frontier in self.observation_manager.frontiers:
                if frontier in self.shared_frontiers:
                    print("Frontier already chosen")
                    shared_frontier_index = self.observation_manager.frontiers.index(frontier)
                    self.observation_manager.position_frontiers.pop(shared_frontier_index)
                    self.observation_manager.frontiers.pop(shared_frontier_index)

            obs = self._get_obs(idx)
            self._save_obs(idx, obs)

            max_distance = math.sqrt((self.world_size * 2)**2 + (self.world_size * 2)**2)

            max_area = self.grid_size * self.grid_size

            discovered_area_rew = (np.sum(new_map == 0) - 
                                   np.sum(old_map == 0)) / max_area

            path_length_rew = -(path_length / max_distance)  # Reward based on path length

            distance_to_closest_drone_rew = 0.0

            for drone_id, drone_position in self.observation_manager.swarm_position_real.items():
                if drone_id != drone.drone_id:
                    dist = self.distance((drone_position[0], drone_position[1]),
                                         (drone.position[0], drone.position[1]))
                    if dist < distance_to_closest_drone_rew or distance_to_closest_drone_rew == 0.0:
                        distance_to_closest_drone_rew = dist

            distance_to_closest_drone_rew = - (max_distance - distance_to_closest_drone_rew) / max_distance  # Reward based on distance to closest drone, The closer the worse, to encourage spreading out

            self.buf_infos[idx] = {"distance": path_length}  # TODO: Add info
            self.buf_rews[idx] = path_length_rew + \
                distance_to_closest_drone_rew + discovered_area_rew
            self.buf_dones[idx] = False

            self.check_end_episode_cond(idx, drone)

            print(f"finished step: Drone {drone.drone_id}")
            with self.condition:
                self.step_lengths[self.env_index] = 10000
                self.condition.notify_all()

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

    def wait_for_map(self, timeout_s: float | None = None) -> bool:
        self.observation_manager.wait_for_map = 0
        start = time.time()
        while self.observation_manager.wait_for_map == 0:
            if timeout_s is not None and (time.time() - start) > timeout_s:
                print("wait_for_map timeout")
                return False
            time.sleep(0.01)
        return True

    def _min_positive_step_length(self) -> int:
        positive_lengths = [length for length in self.step_lengths if length > 0]
        if not positive_lengths:
            return 0
        return min(positive_lengths)

    def _all_step_lengths_zero(self) -> bool:
        return all(length == 0 for length in self.step_lengths)

    def remove_frontier_from_observation(self, position_frontier):
        if position_frontier in self.observation_manager.position_frontiers:
            index = self.observation_manager.position_frontiers.index(position_frontier)
            self.observation_manager.position_frontiers.pop(index)
            self.observation_manager.frontiers.pop(index)

    def remove_shared_frontiers_from_observation(self):
        for shared_frontier in list(self.shared_frontiers):
            if shared_frontier in self.observation_manager.position_frontiers:
                index = self.observation_manager.position_frontiers.index(shared_frontier)
                self.observation_manager.position_frontiers.pop(index)
                self.observation_manager.frontiers.pop(index)
            

    def check_end_episode_cond(self, idx, drone: DroneInterfaceTeleop):
        if len(self.observation_manager.frontiers) <= 10:
            print(f"drone {self.env_index} printing observation manager frontiers: {self.observation_manager.frontiers}")
            with self.lock:
                for shared_frontier in self.shared_frontiers:
                    print(f"drone {self.env_index} printing shared frontier ({shared_frontier[0]}, {shared_frontier[1]})")
                    for frontier in self.observation_manager.frontiers:
                        if self.distance((shared_frontier[0], shared_frontier[1]), (frontier[0], frontier[1])) <= self.max_range_limit:
                            frontier_idx = self.observation_manager.frontiers.index(frontier)
                            self.observation_manager.frontiers.pop(frontier_idx)
                            self.observation_manager.position_frontiers.pop(frontier_idx)
                            print(f"shared frontier ({shared_frontier[0]}, {shared_frontier[1]}), within range of frontier ({frontier[0]}, {frontier[1]})")

        finish = len(self.observation_manager.frontiers) == 0
        if finish:
            print(f"Drone{self.drone_interface_list[0].drone_id} No frontiers left")

            self.buf_dones[idx] = True

            self.lock.acquire(timeout=5)
            try:
                if not self.queue.full():
                    self.queue.put(drone.drone_id)
            finally:
                self.lock.release()

            # self.lock.acquire(timeout=5)
            # try:
                # Mark this env as inactive so _min_positive_step_length ignores it.
            # finally:
            #     self.lock.release()

            with self.condition:
                self.step_lengths[self.env_index] = 10000
                self.condition.notify_all()

            # if not self.barrier_step.broken:
            #     self.barrier_step.abort()
            self.barrier_step.decrement()
            self.barrier_step.print_status()


            print("Drone", drone.drone_id, " before reset")
            self.reset_single_env(idx)

            print("Drone", drone.drone_id, "done")

        return finish


    def frontier_features(self):
        self.remove_shared_frontiers_from_observation()
        frontier_features = self.observation_manager.get_frontier_features()
        self.action_space = Discrete(len(frontier_features))
        return frontier_features


if __name__ == "__main__":
    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world_low_density", world_size=10.0,
                          grid_size=200, min_distance=1.0, num_envs=1, policy_type="MultiInputPolicy")
