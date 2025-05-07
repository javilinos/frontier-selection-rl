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


class ActionSingleValue:
    def __init__(self, drone_interface_list):
        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.drone_interface_list = drone_interface_list
        self.actions = None
        self.generate_path_action_client_list = []
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )

    def take_action(self, frontier_list, env_id) -> tuple:
        action = self.actions[env_id]
        frontier = self.select_frontier(frontier_list, action)

        result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(frontier)

        return frontier, path_length, result

    def generate_random_action(self):
        return [np.array([random.uniform(0, 1)])]

    def select_frontier(self, frontier_list, action):
        index = int(round(action[0] * (len(frontier_list) - 1)))

        return frontier_list[index]


class ActionScalarVector:
    def __init__(self, drone_interface_list):
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.drone_interface_list = drone_interface_list
        self.actions = None
        self.generate_path_action_client_list = []
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def take_action(self, frontiers, world_size, env_id):
        action = self.actions[env_id]

        angle_action = math.atan2(action[1], action[0])
        magnitude_action = math.sqrt(action[0] ** 2 + action[1] ** 2)

        # for frontier in frontier_positions:
        #     frontier[0] = frontier[0] - grid_size / 2
        #     frontier[1] = -(frontier[1] - grid_size / 2)

        frontier_index, closest_distance = self.select_frontier(
            angle_action, magnitude_action, frontiers, world_size)

        result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(
            frontiers[frontier_index])

        return frontiers[frontier_index], path_length, closest_distance, result

    def select_frontier(self, angle_action, magnitude_action, frontiers, world_size):
        closest_index = None
        closest_distance = float("inf")
        x_action = magnitude_action * math.cos(angle_action)
        y_action = magnitude_action * math.sin(angle_action)

        for index, frontier in enumerate(frontiers):
            x_frontier = frontier[0] / world_size
            y_frontier = frontier[1] / world_size

            point_distance = math.sqrt((x_frontier - x_action) ** 2 + (y_frontier - y_action) ** 2)

            if point_distance < closest_distance:
                closest_distance = point_distance
                closest_index = index

        chosen_action = PointStamped()
        chosen_action.header.frame_id = "earth"
        chosen_action.point.x = x_action * world_size
        chosen_action.point.y = y_action * world_size
        self.chosen_action_pub.publish(chosen_action)

        return closest_index, closest_distance


class ActionCartessianCoordinateVector:
    def __init__(self, drone_interface_list):
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.drone_interface_list = drone_interface_list
        self.actions = None
        self.generate_path_action_client_list = []
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def take_action(self, frontiers, world_size, env_id):
        action = self.actions[env_id]

        x = action[0]
        y = action[1]

        # for frontier in frontier_positions:
        #     frontier[0] = frontier[0] - grid_size / 2
        #     frontier[1] = -(frontier[1] - grid_size / 2)

        frontier_index, closest_distance = self.select_frontier(
            x, y, frontiers, world_size)

        result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(
            frontiers[frontier_index])

        return frontiers[frontier_index], path_length, closest_distance, result

    def select_frontier(self, x_action, y_action, frontiers, world_size):
        closest_index = None
        closest_distance = float("inf")

        for index, frontier in enumerate(frontiers):
            x_frontier = frontier[0] / world_size
            y_frontier = frontier[1] / world_size

            point_distance = math.sqrt((x_frontier - x_action) ** 2 + (y_frontier - y_action) ** 2)

            if point_distance < closest_distance:
                closest_distance = point_distance
                closest_index = index

        chosen_action = PointStamped()
        chosen_action.header.frame_id = "earth"
        chosen_action.point.x = x_action * world_size
        chosen_action.point.y = y_action * world_size
        self.chosen_action_pub.publish(chosen_action)

        return closest_index, closest_distance


class DiscreteValueAction:  # To be used with MaskablePPO
    def __init__(self, drone_interface_list):
        self.action_space = Discrete(16)
        self.drone_interface_list = drone_interface_list
        self.actions = None
        self.generate_path_action_client_list = []
        for drone_interface in self.drone_interface_list:
            self.generate_path_action_client_list.append(
                PathActionClient(
                    drone_interface
                )
            )
        self.chosen_action_pub = self.drone_interface_list[0].create_publisher(
            PointStamped, "/chosen_action", 10
        )

    def take_action(self, frontier_list, env_id) -> tuple:
        action = self.actions[env_id]
        frontier = self.select_frontier(frontier_list, action)
        result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(frontier)

        return frontier, path_length, result

    def generate_random_action(self):
        return random.randint(0, 15)

    def select_frontier(self, frontier_list, action):
        chosen_action = PointStamped()
        chosen_action.header.frame_id = "earth"
        # chosen_action.point.x = frontier_list[action][0]
        # chosen_action.point.y = frontier_list[action][1]
        # self.chosen_action_pub.publish(chosen_action)
        return frontier_list[action]


class DiscreteFrontierIndexAction:  # To be used with MaskablePPO
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

    def convert_grid_position_to_pose(self, grid_position: np.ndarray) -> list[float]:
        desp = self.grid_size / 2
        # grid_position[0] corresponds to x (derived from pose[1])
        # grid_position[1] corresponds to y (derived from pose[0])
        pose_1 = (desp - grid_position[0]) / 10.0  # This recovers pose[1]
        pose_0 = (desp - grid_position[1]) / 10.0  # This recovers pose[0]
        return [pose_0, pose_1]

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]], env_id) -> tuple:
        action = self.actions[env_id]
        # action_coord = np.array([action % self.grid_size, action // self.grid_size])
        # action = self.convert_grid_position_to_pose(action_coord)
        # frontier = frontier_list[action_index]
        # result, path_length, _ = self.generate_path_action_client_list[env_id].send_goal(frontier)
        frontier = grid_frontier_list[action]
        frontier = self.convert_grid_position_to_pose(frontier)
        result, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            frontier)
        nav_path = []
        if result:
            for point in path:
                nav_path.append([point.x, point.y])
            # path_simplified = rdp(nav_path, epsilon=0.1)
            path_length = self.path_length(nav_path)

        return action, path_length, result

    def generate_random_action(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def path_length(self, path):
        points = np.array(path)

        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))


class DiscreteCoordinateAction:  # To be used with MaskablePPO
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

    def convert_grid_position_to_pose(self, grid_position: np.ndarray) -> list[float]:
        desp = self.grid_size / 2
        # grid_position[0] corresponds to x (derived from pose[1])
        # grid_position[1] corresponds to y (derived from pose[0])
        pose_1 = (desp - grid_position[0]) / 10.0  # This recovers pose[1]
        pose_0 = (desp - grid_position[1]) / 10.0  # This recovers pose[0]
        return [pose_0, pose_1]

    def take_action(self, frontier_list, grid_frontier_list: list[list[int]], env_id) -> tuple:
        action = self.actions[env_id]
        # action_coord = np.array([action % self.grid_size, action // self.grid_size])
        # action = self.convert_grid_position_to_pose(action_coord)
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


class DiscreteCoordinateActionSingleEnv:  # To be used with MaskablePPO
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

    def take_action(self, frontier_list: list[(int, int)], position_frontier_list: list[(int, int)], env_id) -> tuple:
        action = self.actions[env_id]
        print(f"drone {self.drone_interface_list[0].drone_id}, action {action}")
        action_coord = (action % self.grid_size, action // self.grid_size)
        print(f"drone {self.drone_interface_list[0].drone_id}, action_coord {action_coord}")
        print(
            f"drone {self.drone_interface_list[0].drone_id}, position_frontier_list {position_frontier_list}")
        try:
            action_index = position_frontier_list.index(action_coord)
        except ValueError:
            return None, action_coord, None, None, False

        print(f"drone {self.drone_interface_list[0].drone_id}, index {action_index}")
        # action_coord = np.array([action % self.grid_size, action // self.grid_size])
        # action_index = np.where(np.all(position_frontier_list == action_coord, axis=1))[0][0]
        frontier = frontier_list[action_index]
        position_frontier = position_frontier_list[action_index]
        print(f"drone {self.drone_interface_list[0].drone_id}, frontier: {frontier}")
        success, path_length, path = self.generate_path_action_client_list[env_id].send_goal(
            frontier)
        nav_path = []
        if success:
            for point in path:
                nav_path.append([point.x, point.y])
            # path_simplified = rdp(nav_path, epsilon=0.1)
            path_length = self.path_length(nav_path)
        return frontier, position_frontier, path_length, nav_path, success

    def generate_random_action(self):
        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def path_length(self, path):
        points = np.array(path)
        # Compute Euclidean distances between consecutive points
        return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
