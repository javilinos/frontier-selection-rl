from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
from as2_msgs.action import NavigateToPoint
from rclpy.action import ActionClient
import random
import math
from geometry_msgs.msg import PointStamped, Point
# from rdp import rdp
import time


class PathActionClient:
    def __init__(self, drone_interface):
        self._action_client = ActionClient(
            drone_interface,
            NavigateToPoint,
            f"{drone_interface.get_namespace()}/navigate_to_point",
        )
        self.drone_interface = drone_interface

    def send_goal(self, point: list[float, float]):
        goal_msg = NavigateToPoint.Goal()
        goal_msg.point.point.x = point[0]
        goal_msg.point.point.y = point[1]
        goal_msg.point.header.frame_id = "earth"
        self._action_client.wait_for_server()
        result = self._action_client.send_goal(goal_msg)
        return result.result.success, result.result.path_length.data, result.result.path


class RandomAction:
    def __init__(self, drone_interface_list, grid_size):
        self.dims = [grid_size, grid_size]
        self.action_space = Discrete(grid_size * grid_size)
        self.drone_interface_list = drone_interface_list
        self.actions = []
        self.generate_path_action_client_list = []
        self.grid_size = grid_size
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def convert_pose_to_grid_position(self, pose: list[float]):
        desp = (self.grid_size) / 2
        x = -(round(pose[1] * 10, 0) - desp)
        y = -(round(pose[0] * 10, 0) - desp)
        return np.array([x, y], dtype=np.int32)

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]], env_id, occupancy_grid) -> tuple:
        position = self.convert_pose_to_grid_position(self.drone_interface_list[0].position)
        # Select random action index
        action = random.randint(0, len(grid_frontier_list) - 1)

        frontier = frontier_list[action]
        # result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(frontier)
        result, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            frontier)
        nav_path = []
        if result:
            for point in path:
                nav_path.append([point.x, point.y])
            # path_simplified = rdp(nav_path, epsilon=0.1)
            path_length = self.path_length(nav_path)

        return frontier, path_length, result, nav_path

    def path_length(self, path):
        points = np.array(path)

        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))


class NearestFrontierAction:
    def __init__(self, drone_interface_list, grid_size):
        self.dims = [grid_size, grid_size]
        self.action_space = Discrete(grid_size * grid_size)
        self.drone_interface_list = drone_interface_list
        self.actions = []
        self.generate_path_action_client_list = []
        self.grid_size = grid_size
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def convert_pose_to_grid_position(self, pose: list[float]):
        desp = (self.grid_size) / 2
        x = -(round(pose[1] * 10, 0) - desp)
        y = -(round(pose[0] * 10, 0) - desp)
        return np.array([x, y], dtype=np.int32)

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]], env_id, occupancy_grid) -> tuple:
        position = self.convert_pose_to_grid_position(self.drone_interface_list[0].position)
        # Get closest frontier position based on euclidean distance
        closest_distance = np.inf
        for i in range(len(grid_frontier_list)):
            grid_frontier = grid_frontier_list[i]
            grid_frontier = np.array(grid_frontier)
            distance = np.linalg.norm(np.array([position[0], position[1]]) - grid_frontier)
            if distance < closest_distance:
                closest_distance = distance
                action = i
        # Find the index of the closest frontier

        frontier = frontier_list[action]
        # result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(frontier)
        result, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            frontier)
        nav_path = []
        if result:
            for point in path:
                nav_path.append([point.x, point.y])
            # path_simplified = rdp(nav_path, epsilon=0.1)
            path_length = self.path_length(nav_path)

        return frontier, path_length, result, nav_path

    def generate_random_action(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def path_length(self, path):
        points = np.array(path)

        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))


class HybridAction:
    def __init__(self, drone_interface_list, grid_size):
        self.dims = [grid_size, grid_size]
        self.action_space = Discrete(grid_size * grid_size)
        self.drone_interface_list = drone_interface_list
        self.actions = []
        self.generate_path_action_client_list = []
        self.grid_size = grid_size
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def compute_information_gain(self, frontier, occupancy_grid, sensor_range):
        ig = 0
        for dx in range(-sensor_range, sensor_range + 1):
            for dy in range(-sensor_range, sensor_range + 1):
                if dx**2 + dy**2 <= sensor_range**2:
                    x, y = frontier[0] + dx, frontier[1] + dy
                    if 0 <= x < occupancy_grid[2].shape[0] and 0 <= y < occupancy_grid[2].shape[1]:
                        if occupancy_grid[2][x, y] == 255:
                            ig += 1

        max_ig = np.pi * (sensor_range ** 2)
        return ig / max_ig  # returns a value between 0 and 1

    def convert_pose_to_grid_position(self, pose: list[float]):
        desp = (self.grid_size) / 2
        x = -(round(pose[1] * 10, 0) - desp)
        y = -(round(pose[0] * 10, 0) - desp)
        return np.array([x, y], dtype=np.int32)

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]], env_id, occupancy_grid, sensor_range=10, lambda_weight=1.0) -> tuple:
        position = self.convert_pose_to_grid_position(self.drone_interface_list[0].position)
        best_utility = -np.inf
        best_frontier_idx = None
        best_info_gain = 0
        best_distance = 0

        for i, grid_frontier in enumerate(grid_frontier_list):
            grid_frontier = np.array(grid_frontier)
            distance = np.linalg.norm(position - grid_frontier) / (math.sqrt(2) * self.grid_size)
            info_gain = self.compute_information_gain(grid_frontier, occupancy_grid, sensor_range)

            # utility = info_gain * np.exp(-lambda_weight * distance)
            utility = (info_gain * 0.5) - (distance * 0.5)

            if utility > best_utility:
                best_utility = utility
                best_frontier_idx = i
                best_info_gain = info_gain
                best_distance = distance

        frontier = frontier_list[best_frontier_idx]
        result, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            frontier)

        nav_path = []
        if result:
            for point in path:
                nav_path.append([point.x, point.y])
            path_length = self.path_length(nav_path)

        return frontier, path_length, result, nav_path

    def generate_random_action(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def path_length(self, path):
        points = np.array(path)

        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))


class TAREAction:
    def __init__(self, drone_interface_list, grid_size):
        self.dims = [grid_size, grid_size]
        self.action_space = Discrete(grid_size * grid_size)
        self.drone_interface_list = drone_interface_list
        self.actions = []
        self.generate_path_action_client_list = []
        self.grid_size = grid_size
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def compute_information_gain(self, frontier, occupancy_grid, sensor_range):
        ig = 0
        for dx in range(-sensor_range, sensor_range + 1):
            for dy in range(-sensor_range, sensor_range + 1):
                if dx**2 + dy**2 <= sensor_range**2:
                    x, y = frontier[0] + dx, frontier[1] + dy
                    if 0 <= x < occupancy_grid[2].shape[0] and 0 <= y < occupancy_grid[2].shape[1]:
                        if occupancy_grid[2][x, y] == 255:
                            ig += 1

        max_ig = np.pi * (sensor_range ** 2)
        return ig / max_ig  # returns a value between 0 and 1

    def convert_pose_to_grid_position(self, pose: list[float]):
        desp = (self.grid_size) / 2
        x = -(round(pose[1] * 10, 0) - desp)
        y = -(round(pose[0] * 10, 0) - desp)
        return np.array([x, y], dtype=np.int32)

    def is_within_horizon(self, frontier, position, horizon_radius):
        dx = frontier[0] - position[0]
        dy = frontier[1] - position[1]
        return dx**2 + dy**2 <= horizon_radius**2

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]],
                    env_id, occupancy_grid, sensor_range=10, horizon_radius=60,
                    w_info=1.0, w_dist=1.0):
        position = self.convert_pose_to_grid_position(self.drone_interface_list[0].position)
        best_utility = -np.inf
        best_frontier = None
        best_index = None

        for i, grid_frontier in enumerate(grid_frontier_list):

            grid_frontier = np.array(grid_frontier)
            distance = np.linalg.norm(position - grid_frontier) / (math.sqrt(2) * self.grid_size)
            info_gain = self.compute_information_gain(grid_frontier, occupancy_grid, sensor_range)

            # utility = info_gain * np.exp(-lambda_weight * distance)
            utility = (info_gain * 0.25) - (distance * 0.75) + \
                (10000 if self.is_within_horizon(grid_frontier, position, horizon_radius) else 0)

            if utility > best_utility:
                best_utility = utility
                best_index = i
                best_frontier = grid_frontier

        if best_frontier is None:
            print("No local frontier found within horizon.")
            return None, 0, False  # No local frontier found
        best_frontier = frontier_list[best_index]
        # Send goal and retrieve path
        result, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            best_frontier)
        nav_path = []
        if result:
            for point in path:
                nav_path.append([point.x, point.y])
            path_length = self.path_length(nav_path)

        return best_frontier, path_length, result, nav_path

    def generate_random_action(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def path_length(self, path):
        points = np.array(path)

        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
