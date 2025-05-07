from async_vector_env import AsyncPPO
from as2_gymnasium_env_discrete_multiagent import AS2GymnasiumEnv
from gymnasium.spaces import Box, Dict, Discrete
import argparse
from RL_ENV.policies.custom_cnn import CustomCombinedExtractor
import time
import rclpy
import cProfile
import pstats
import torch
from torch.distributions.constraints import Constraint
from torch.distributions import constraints
import numpy as np
from torch.multiprocessing import Manager, Lock, Barrier, Condition, Queue
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy, MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from concurrent_ppo_mask import MaskablePPO
from stable_baselines3.common.utils import get_schedule_fn
from multiprocessing.managers import BaseManager


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


class SharedPolicyManager(BaseManager):
    pass


class CustomSimplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """

    event_dim = 1

    def check(self, value):

        return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-4)


class Training:
    def __init__(self, env: AS2GymnasiumEnv, policy: MaskableMultiInputActorCriticPolicy, policy_lock: None,
                 n_steps: int = 128,
                 batch_size: int = 32, n_epochs: int = 5, learning_rate: float = 0.00005,
                 pi_net_arch: list = [128, 128], vf_net_arch: list = [128, 128]):
        self.env = env
        # torch.cuda.is_available = lambda: False  # Disable cuda
        self.model = MaskablePPO(
            policy,
            self.env,
            verbose=1,
            lock=policy_lock,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device="cpu",
            tensorboard_log=f"./tensorboard/{self.env.env_index}",
            policy_kwargs=dict(
                net_arch=dict(pi=pi_net_arch, vf=vf_net_arch),
                features_extractor_class=CustomCombinedExtractor,
            )
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
            total_timesteps=50000
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

# def mask_fn(env: AS2GymnasiumEnv) -> np.ndarray:
#     # Do whatever you'd like in this function to return the action mask
#     # for the current env. In this example, we assume the env has a
#     # helpful method we can rely on.

#     return env.action_masks()


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
    parser.add_argument("--policy_type", type=str, default="MultiInputPolicy", help="Policy type")
    args = parser.parse_args()

    action_space = Discrete(200 * 200)

    manager = Manager()
    lock = Lock()
    policy_lock = Lock()
    barrier_reset = Barrier(4)
    barrier_step = Barrier(4)
    condition = Condition()
    queue = Queue(maxsize=3)
    shared_frontiers = manager.list()
    drones_initial_positions = manager.list()
    vec_sync = manager.list([False, False, False, False])
    # First three for reset

    step_lengths = manager.list([10000.0, 10000.0, 10000.0, 10000.0])

    torch.distributions.Categorical.arg_constraints = {
        "probs": CustomSimplex(), "logits": constraints.real_vector}  # Modify simplex constrain to be less restrictive

    rclpy.init()

    policy_class = args.policy_type
    learning_rate = 0.00005

    if policy_class == "MultiInputPolicy":
        observation_space = Dict(
            {
                "image": Box(
                    low=0, high=255, shape=(1, 200, 200), dtype=np.uint8
                ),  # Ocuppancy grid map: 0: free, 1: occupied, 2: unknown, 3: frontier point
                "position": Box(low=0, high=200 - 1, shape=(2,), dtype=np.int32),
                # Position of the drone in the grid
            }
        )

        SharedPolicyManager.register('MaskableMultiInputActorCriticPolicy',
                                     MaskableMultiInputActorCriticPolicy)
        manager = SharedPolicyManager()
        manager.start()

        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            features_extractor_class=CustomCombinedExtractor
        )

        # multiprocessing.set_start_method('spawn', force=True)

        policy = manager.MaskableMultiInputActorCriticPolicy(  # type: ignore[assignment]
            observation_space,
            action_space,
            (0.0, learning_rate),
            **policy_kwargs,
        )

    elif policy_class == "MlpPolicy":

        observation_space = Box(low=0, high=1, shape=(
            200 * 200 + 2,), dtype=np.float32)
        SharedPolicyManager.register('MaskableActorCriticPolicy',
                                     MaskableActorCriticPolicy)
        manager = SharedPolicyManager()
        manager.start()

        policy_kwargs = dict(
            # activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )

        policy = manager.MaskableActorCriticPolicy(  # type: ignore[assignment]
            observation_space,
            action_space,
            (0.0, learning_rate),
            **policy_kwargs,
        )

    policy = policy.to("cpu")
    policy.share_memory()

    # policy.share_memory()

    def make_training_0():
        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=0, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, policy_lock=policy_lock)

    def make_training_1():
        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=1, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, policy_lock=policy_lock)

    def make_training_2():
        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=2, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, policy_lock=policy_lock)

    def make_training_3():
        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=3, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, policy_lock=policy_lock)

    # env = AS2GymnasiumEnv(world_name="world2", world_size=10.0,
    #                       grid_size=200, min_distance=1.0, policy_type="MultiInputPolicy", namespace="drone0")

    # env = VecMonitor(env)
    # env = ActionMasker(env.venv, action_mask_fn=Training.mask_fn)
    # multiprocessing.set_start_method('fork', force=True)
    ppos = AsyncPPO([make_training_0, make_training_1, make_training_2, make_training_3])
    ppos.call_async("train")
    ppos.call_wait()
    # env0 = make_env_0()
    # # env1 = make_env_1()
    # print("Start mission")
    # custom_callback = CustomCallback()
    # training0 = Training(env0, custom_callback)
    # # training1 = Training(env1, custom_callback)
    # processes = []
    # processes.append(Process(target=training0.train, args=()))
    # # processes.append(Process(target=training1.train, args=()))
    # # training0.train()
    # for p in processes:
    #     p.start()

    # for p in processes:
    #     p.join()

    rclpy.shutdown()
