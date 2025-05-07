#!/bin/python3
"""explore.py"""

import sys
from time import sleep
import rclpy
from rclpy.task import Future
from rclpy import logging
from rclpy.qos import qos_profile_sensor_data
from as2_python_api.shared_data.twist_data import TwistData
from as2_python_api.shared_data.pose_data import PoseData
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.tools.utils import euler_from_quaternion
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import TwistStamped, PointStamped, PoseStamped


class Explorer(DroneInterface):
    """An UAV born to explore the world"""

    def __init__(self, drone_id: str = "drone0", verbose: bool = False,
                 use_sim_time: bool = False) -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.namespace = drone_id
        self.crashed = False

        self.explore_client = self.create_client(SetBool, "start_exploration")

        # Overriding twist methods to get path length
        self._last_timestamp: float = None
        self.path_length: float = 0.0

        self.__twist = TwistData(1.0)
        self.twist_sub = self.create_subscription(
            TwistStamped, 'self_localization/twist', self.twist_cbk, qos_profile_sensor_data)

        # Overriding pose methods to republish to evaluator
        self.__pose = PoseData()
        self.pose_sub = self.create_subscription(
            PoseStamped, 'self_localization/pose', self.pose_cbk, qos_profile_sensor_data)

        # PUBLISHERS FOR EVALUATOR
        # Using PointStamped msg and avoiding custom msg
        self.path_length_pub = self.create_publisher(
            PointStamped, "/eval/path_length", 10)
        self.poses_pub = self.create_publisher(
            PointStamped, "/eval/poses", 10)

    def explore(self) -> Future:
        """Call exploration service asynchronously and return a future"""
        request = SetBool.Request()
        request.data = True

        return self.explore_client.call_async(request)

    @property
    def connected(self) -> bool:
        """Check if the drone is connected"""
        return self.info["connected"]

    @property
    def speed(self) -> list[float]:
        """Get drone speed (vx, vy, vz) in m/s.

        :rtype: List[float]
        """
        return self.__twist.twist

    def twist_cbk(self, twist_msg: TwistStamped) -> None:
        """twist stamped callback"""
        self.__twist.twist = [twist_msg.twist.linear.x,
                              twist_msg.twist.linear.y,
                              twist_msg.twist.linear.z]

        # Path length
        timestamp = twist_msg.header.stamp.sec + \
            twist_msg.header.stamp.nanosec * 1e-9
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return
        delta = timestamp - self._last_timestamp
        self.path_length += (abs(self.speed[0]) + abs(self.speed[1])) * delta

        msg = PointStamped()
        msg.header = twist_msg.header
        msg.point.x = self.path_length
        self.path_length_pub.publish(msg)

        self._last_timestamp = timestamp

    # TODO: not able to super() this method since it's private. Really needed to be private?
    def pose_cbk(self, pose_msg: PoseStamped) -> None:
        """pose stamped callback"""
        if round(pose_msg.pose.position.z, 1) == 0.0 and pose_msg.header.stamp.sec > 100:
            self.crashed = True

        self.__pose.position = [pose_msg.pose.position.x,
                                pose_msg.pose.position.y,
                                pose_msg.pose.position.z]

        self.__pose.orientation = [
            *euler_from_quaternion(
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w)]

        msg = PointStamped()
        msg.header = pose_msg.header
        # NOTE: frame_id is not overwritten in the evaluator with earth.
        # Not a good practice, just to avoid creating new msg
        msg.header.frame_id = self.namespace
        msg.point = pose_msg.pose.position
        self.poses_pub.publish(msg)


if __name__ == '__main__':
    rclpy.init()

    scouts: list[Explorer] = []
    scouts.append(Explorer(drone_id="drone0",
                  verbose=False, use_sim_time=True))
    # scouts.append(Explorer(drone_id="drone1",
    #               verbose=False, use_sim_time=True))
    # scouts.append(Explorer(drone_id="drone2",
    #               verbose=False, use_sim_time=True))
    # scouts.append(Explorer(drone_id="drone3",
    #               verbose=False, use_sim_time=True))
    # scouts.append(Explorer(drone_id="drone4",
    #               verbose=False, use_sim_time=True))
    # scouts.append(Explorer(drone_id="drone5",
    #               verbose=False, use_sim_time=True))
    # scouts.append(Explorer(drone_id="drone6",
    #               verbose=False, use_sim_time=True))

    sleep(5)
    # Only keep connected drones
    for scout in scouts.copy():
        if not scout.connected:
            logging.get_logger("rclpy").info(
                f"{scout.drone_id} not connected, removing from list")
            scouts.remove(scout)
            scout.shutdown()
    if not scouts:
        logging.get_logger("rclpy").info("No connected drones")
        rclpy.shutdown()
        sys.exit(1)
    evaluator_start = scouts[0].create_client(Trigger, "/evaluator/start")

    for scout in scouts:
        scout.offboard()
        scout.arm()

    logging.get_logger("rclpy").info("Taking off")
    for scout in scouts:
        wait = scout == scouts[-1]
        scout.takeoff(1.0, wait=wait)

    sleep(5)
    logging.get_logger("rclpy").info("Starting evaluator")
    evaluator_start.call_async(Trigger.Request())

    timeout: int = 6000

    futures: list[Future] = [scout.explore() for scout in scouts]
    logging.get_logger("rclpy").info("Exploring")
    while not all((fut.done() for fut in futures)):
        sleep(0.5)
        if any((scout.crashed for scout in scouts)):
            logging.get_logger("rclpy").error(
                f"Crash detected for {scout.drone_id}")
            for fut in futures:
                fut.cancel()
            break

        if scouts[0].get_clock().now().seconds_nanoseconds()[0] > timeout:
            logging.get_logger("rclpy").error("Timeout")
            for fut in futures:
                fut.cancel()
            break

    if not any(scout.crashed for scout in scouts):
        logging.get_logger("rclpy").info("Landing")
        for scout in scouts:
            wait = scout == scouts[-1]
            scout.land(wait=wait)

    for scout in scouts:
        scout.disarm()
        scout.shutdown()
    rclpy.shutdown()

    print("Clean exit")
    sys.exit(0)
