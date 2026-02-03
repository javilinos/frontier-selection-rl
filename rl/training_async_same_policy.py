from environments.async_vector_env import AsyncPPO
from environments.as2_gymnasium_env_discrete_multiagent_attention import AS2GymnasiumEnv
from gymnasium.spaces import Box, Discrete
import argparse
from algorithms.policies.features_extractors.custom_cnn import NatureCNN_Mod
import time
import rclpy
import cProfile
import pstats
import torch
import signal
import sys
from torch.distributions.constraints import Constraint
from torch.distributions import constraints
import numpy as np
from torch.multiprocessing import Manager, Lock, Barrier, Condition, Queue
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from algorithms.policies.custom_policy_attention import ActorCriticPolicy
from algorithms.concurrent_custom_ppo import PPO
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


class ConstantSchedule:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, _progress_remaining: float) -> float:
        return float(self.value)


class Training:
    def __init__(self, env: AS2GymnasiumEnv, policy: ActorCriticPolicy, policy_lock: None,
                 n_steps: int = 128,
                 batch_size: int = 32, n_epochs: int = 5, learning_rate: float = 0.00005,
                 pi_net_arch: list = [128, 128], vf_net_arch: list = [128, 128]):
        self.env = env
        # torch.cuda.is_available = lambda: False  # Disable cuda
        self.model = PPO(
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
                features_extractor_class=NatureCNN_Mod,
                share_features_extractor=True,
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
    parser.add_argument("--policy_type", type=str, default="CnnPolicy", help="Policy type")
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

    step_lengths = manager.list([-1.0, -1.0, -1.0, -1.0])

    torch.distributions.Categorical.arg_constraints = {
        "probs": CustomSimplex(), "logits": constraints.real_vector}  # Modify simplex constrain to be less restrictive

    rclpy.init()

    policy_class = args.policy_type
    learning_rate = 0.0003

    if policy_class != "CnnPolicy":
        raise ValueError("Async attention training only supports CnnPolicy.")

    observation_space = Box(low=0, high=255, shape=(5, 200, 200), dtype=np.uint8)

    SharedPolicyManager.register('ActorCriticPolicy', ActorCriticPolicy)
    manager = SharedPolicyManager()
    manager.start()

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        features_extractor_class=NatureCNN_Mod,
        share_features_extractor=True,
    )

    policy = manager.ActorCriticPolicy(  # type: ignore[assignment]
        observation_space,
        action_space,
        ConstantSchedule(learning_rate),
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

    def _shutdown(signum=None, frame=None):
        print("Shutdown requested, terminating worker processes...")
        try:
            ppos.close_extras(terminate=True)
        finally:
            try:
                rclpy.shutdown()
            finally:
                sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        ppos.call_async("train")
        ppos.call_wait()
    except KeyboardInterrupt:
        _shutdown()
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
