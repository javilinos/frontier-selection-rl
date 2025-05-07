from async_vector_env import AsyncPPO
from as2_gymnasium_env_discrete_multiagent import AS2GymnasiumEnv
import argparse
from RL_ENV.policies.custom_cnn import CustomCombinedExtractor
import time
import rclpy
import cProfile
import pstats
import torch
import numpy as np
from torch.multiprocessing import Manager, Lock, Barrier, Condition, Queue
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

# multiprocessing.set_start_method('forkserver', force=True)
# multiprocessing.set_start_method('spawn', force=True)


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


class Training:
    def __init__(self, env: AS2GymnasiumEnv, n_steps: int = 128,
                 batch_size: int = 32, n_epochs: int = 5, learning_rate: float = 0.00005,
                 pi_net_arch: list = [128, 128], vf_net_arch: list = [128, 128]):
        self.env = env
        # torch.cuda.is_available = lambda: False  # Disable cuda
        self.model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            self.env,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            policy_kwargs=dict(
                pi_net_arch=pi_net_arch,
                vf_net_arch=vf_net_arch,
                features_extractor_class=CustomCombinedExtractor,
            ),

        )
        print(f"Training with n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, \
              learning_rate={learning_rate}, pi_net_arch={pi_net_arch}, vf_net_arch={vf_net_arch}")

    # def mask_fn(env: AS2GymnasiumEnv) -> np.ndarray:
    #     # Do whatever you'd like in this function to return the action mask
    #     # for the current env. In this example, we assume the env has a
    #     # helpful method we can rely on.
    #     return env.valid_action_mask()

    def train(self):
        print("Training the model...")
        self.model.learn(
            total_timesteps=100000
        )

        ##### PROFILING #####

        # with cProfile.Profile() as pr:

        #     model.learn(
        #         total_timesteps=10,
        #         callback=self.custom_callback,
        #     )

        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME)
        # # stats.print_stats()
        # stats.dump_stats(
        #     filename='profilings/needs_profiling_no_clock_sim_time.prof')

        ######################

        self.model.save(f"ppo_as2_gymnasium{self.env.env_index}")

    def close(self):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform training of the model")
    parser.add_argument("--n_steps", type=int, default=128,
                        help="Number of steps in the environment")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--pi_net_arch", type=list,
                        default=[128, 128], help="Policy network architecture")
    parser.add_argument("--vf_net_arch", type=list,
                        default=[128, 128], help="Value function network architecture")
    parser.add_argument("--world_name", type=str, default="world2", help="Name of the world")
    parser.add_argument("--world_size", type=float, default=10.0, help="Size of the world")
    parser.add_argument("--grid_size", type=int, default=200, help="Size of the grid")
    parser.add_argument("--num_drones", type=int, default=3, help="Number of drones")
    args = parser.parse_args()

    num_drones = args.num_drones
    world_name = args.world_name
    world_size = args.world_size
    grid_size = args.grid_size

    manager = Manager()
    lock = Lock()
    barrier_reset = Barrier(num_drones)
    barrier_step = Barrier(num_drones)
    condition = Condition()
    queue = Queue()
    shared_frontiers = manager.list()
    drones_initial_positions = manager.list()
    vec_sync = manager.list([False] * num_drones)
    # First three for reset

    step_lengths = manager.list([-1.0] * num_drones)

    def takeoff(env):
        print("Start mission")
        #### ARM OFFBOARD #####
        print("Arm")
        env.drone_interface_list[0].offboard()
        time.sleep(1.0)
        print("Offboard")
        env.drone_interface_list[0].arm()
        time.sleep(1.0)

        ##### TAKE OFF #####
        print("Take Off")
        env.drone_interface_list[0].takeoff(1.0, speed=1.0)
        time.sleep(1.0)

    def make_env_0():
        env = AS2GymnasiumEnv(world_name=world_name, world_size=world_size,
                              grid_size=grid_size, min_distance=1.0, num_envs=1, num_drones=num_drones, env_index=0, policy_type="MultiInputPolicy",
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env)

    def make_env_1():
        env = AS2GymnasiumEnv(world_name=world_name, world_size=world_size,
                              grid_size=grid_size, min_distance=1.0, num_envs=1, num_drones=num_drones, env_index=1, policy_type="MultiInputPolicy",
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env)

    def make_env_2():
        env = AS2GymnasiumEnv(world_name=world_name, world_size=world_size,
                              grid_size=grid_size, min_distance=1.0, num_envs=1, num_drones=num_drones, env_index=2, policy_type="MultiInputPolicy",
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env)

    def make_env_3():
        env = AS2GymnasiumEnv(world_name=world_name, world_size=world_size,
                              grid_size=grid_size, min_distance=1.0, num_envs=1, num_drones=num_drones, env_index=3, policy_type="MultiInputPolicy",
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env)

    rclpy.init()

    ppos = AsyncPPO([make_env_0, make_env_1, make_env_2])
    ppos.call_async("train")
    ppos.call_wait()

    rclpy.shutdown()
