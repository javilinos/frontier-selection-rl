from environments.as2_gymnasium_env_discrete_single_agent import AS2GymnasiumEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from algorithms.policies.custom_policy_attention import ActorCriticCnnPolicy, ActorCriticPolicy
from algorithms.custom_ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from evaluate_policy import evaluate_policy
import torch as th
import pstats
import cProfile
import numpy as np
import time
import rclpy
from torch import nn
from gymnasium import spaces
import gymnasium as gym
import argparse


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        # Access the actual environment from the callback's training environment
        pass

    def _on_step(self) -> bool:
        # Get the done flags and rewards for all environments
        pass

        return True

    def _on_rollout_end(self) -> None:
        # Compute and log the means of the rewards and lengths
        pass


class Test:
    def __init__(self, env: AS2GymnasiumEnv, custom_callback: CustomCallback, path: str):
        self.env = env
        self.custom_callback = custom_callback
        self.model = PPO.load(path, self.env, device='cpu')

    def test(self, num_episodes=10):
        mean_reward, std_reward = evaluate_policy(self.model.policy, self.env, num_episodes)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        self.env.drone_interface_list[0].shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test RL policy')
    parser.add_argument(
        "--world_type", type=str, default="low_density",
        help="World name to test on", choices=["low_density", "medium_density", "high_density"]
    )
    parser.add_argument('--num_episodes', type=int, default=1, help='Size of the world')

    args = parser.parse_args()

    rclpy.init()
    env = AS2GymnasiumEnv(world_name=f"world_{args.world_type}", world_size=2.5,
                          grid_size=200, min_distance=1.0, num_envs=1, policy_type="CnnPolicy", testing=True)
    env = VecMonitor(env)
    custom_callback = CustomCallback()
    test = Test(env, custom_callback, "trained_policy/ppo_as2_gymnasium.zip")
    test.test(args.num_episodes)
    rclpy.shutdown()
