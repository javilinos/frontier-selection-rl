from typing import Optional
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from algorithms.policies.custom_policy_attention import ActorCriticPolicy
from custom_buffer import RolloutBuffer
from algorithms.concurrent_custom_ppo import PPO
from multiprocessing.managers import BaseManager
from multiprocessing import Value
import ctypes


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
    


class DynamicEarlyExitBarrier:
    def __init__(self, parties: int, rollout_remaining, lock=None):
        if parties < 1:
            raise ValueError("parties must be >= 1")

        self.rollout_remaining = rollout_remaining
        self.max_parties = parties                          # for sanity checks only
        self.parties = Value(ctypes.c_int, parties)      # dynamic threshold
        self.count = Value(ctypes.c_int, 0)              # arrivals in current generation
        self.generation = Value(ctypes.c_int, 0)         # generation number
        self.cond = Condition(lock)

    def wait(self):
        with self.cond:
            gen = self.generation.value

            # Early-exit: if we're done collecting, never block
            # if self.rollout_remaining.value <= 0:
            #     self._release_locked()
            #     return False

            # If parties was shrunk to 0, don't block
            if self.parties.value <= 0:
                return False

            # Arrive
            self.count.value += 1

            # Normal release condition
            if self.count.value >= self.parties.value:
                self._release_locked()
                return True  # "leader"
            else:
                # Wait until this generation releases OR early-exit triggers
                self.cond.wait_for(
                    lambda: (
                        self.generation.value != gen
                        # or self.rollout_remaining.value <= 0
                        or self.parties.value <= 0
                    )
                )
                return False

    def increment(self, n: int = 1):
        """Increase parties for *future* waits (or this gen if you really know what you're doing)."""
        if n < 1:
            return
        with self.cond:
            if self.parties.value + n > self.max_parties:
                print("Warning: trying to increment barrier parties above initial number. Capping to max_parties.")
                return
            self.parties.value += n
            # No notify required; increasing doesn't unblock anyone.

    def decrement(self, n: int = 1):
        """
        Decrease parties. If enough processes are already waiting, release them immediately.
        This makes '3 waiting, 4th leaves => 3 pass' work.
        """
        if n < 1:
            return
        with self.cond:
            new_parties = self.parties.value - n
            self.parties.value = new_parties

            # If threshold becomes <= 0, release everyone and make waits non-blocking
            if self.parties.value <= 0:
                self.parties.value = 0
                self._release_locked()
                return

            # KEY FEATURE:
            # if already-waiting arrivals meet the new threshold, release right now
            if self.count.value >= self.parties.value:
                self._release_locked()

    def _release_locked(self):
        """Release the current generation. Must be called with self.cond acquired."""
        self.count.value = 0
        self.generation.value += 1
        self.cond.notify_all()

    def print_status(self):
        with self.cond:
            print(f"Barrier status: parties={self.parties.value}, count={self.count.value}, generation={self.generation.value}")


class Training:
    def __init__(self, env: AS2GymnasiumEnv, policy: ActorCriticPolicy, shared_buffer: Optional[RolloutBuffer], policy_lock: None,
                 n_steps: int = 128, rollout_remaining=None, global_timesteps=None, batch_id=None,
                 barrier_collect_done=None, barrier_train_done=None, rollout_pool=None, env_barrier_step=None,
                 env_barrier_reset=None,
                 batch_size: int = 32, n_epochs: int = 5, learning_rate: float = 0.0003,
                 pi_net_arch: list = [128, 128], vf_net_arch: list = [128, 128]):
        self.env = env
        # torch.cuda.is_available = lambda: False  # Disable cuda
        self.model = PPO(
            policy=policy,
            shared_buffer=shared_buffer,
            env=self.env,
            verbose=1,
            lock=policy_lock,
            rollout_remaining=rollout_remaining,
            global_timesteps=global_timesteps,
            batch_id=batch_id,
            env_barrier_step=env_barrier_step,
            env_barrier_reset=env_barrier_reset,
            barrier_collect_done=barrier_collect_done,
            barrier_train_done=barrier_train_done,
            is_learner=(self.env.env_index == 0),
            rollout_pool=rollout_pool,
            n_envs_total=N_ENVS,
            n_steps_total=N_TOTAL,
            n_steps=N_TOTAL,
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
        self.checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=f'./models/{self.env.env_index}', name_prefix='ppo_model')
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
            total_timesteps=200000,
            callback=self.checkpoint_callback,
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
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--pi_net_arch", type=list,
                        default=[128, 128], help="Policy network architecture")
    parser.add_argument("--vf_net_arch", type=list,
                        default=[128, 128], help="Value function network architecture")
    parser.add_argument("--policy_type", type=str, default="CnnPolicy", help="Policy type")
    args = parser.parse_args()

    N_ENVS = 4
    N_TOTAL = args.n_steps * N_ENVS  # e.g. 128*4

    rollout_remaining = Value(ctypes.c_int, N_TOTAL)   # how many total events left to collect in this batch
    global_timesteps = Value(ctypes.c_long, 0)         # optional: global counter for logging/stop
    batch_id = Value(ctypes.c_int, 0)                  # optional: increments each PPO update

    barrier_collect_done = Barrier(N_ENVS)             # all workers finished collecting for this batch
    barrier_train_done = Barrier(N_ENVS)               # all workers wait until training is done

    action_space = Discrete(200 * 200)

    manager = Manager()
    lock = Lock()
    policy_lock = Lock()
    barrier_reset = DynamicEarlyExitBarrier(N_ENVS, rollout_remaining, lock=lock)
    # barrier_step = Barrier(N_ENVS)

    barrier_step = DynamicEarlyExitBarrier(N_ENVS, rollout_remaining, lock=lock)  # all workers sync at each step, but if rollout_remaining hits 0, they can stop blocking and just keep going to finish the episode

    condition = Condition(lock)
    queue = Queue(maxsize=3)
    shared_frontiers = manager.list()
    drones_initial_positions = manager.list()
    vec_sync = manager.list([False] * N_ENVS)
    rollout_pool = manager.list([None] * N_ENVS)

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
    # SharedPolicyManager.register('RolloutBuffer', RolloutBuffer)

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

    # shared_buffer = manager.RolloutBuffer(
    #     buffer_size=N_TOTAL,              # total decision-events across all agents before update
    #     observation_space=observation_space,
    #     action_space=action_space,
    #     device="cpu",
    # )

    def make_training_0():
        local_shared_buffer = RolloutBuffer(
            buffer_size=N_TOTAL,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            gamma=0.99,        # set explicitly (match PPO)
            gae_lambda=0.95,   # set explicitly (match PPO)
            n_envs=1,
            distance_unit=1.0, # set to your chosen unit (or speed if discount per second)
)

        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=0, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, shared_buffer=local_shared_buffer,  policy_lock=lock, rollout_remaining=rollout_remaining, global_timesteps=global_timesteps, 
                        batch_id=batch_id, barrier_collect_done=barrier_collect_done, barrier_train_done=barrier_train_done, env_barrier_step=barrier_step, env_barrier_reset=barrier_reset,
                        rollout_pool=rollout_pool)

    def make_training_1():
        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=1, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, shared_buffer=None,  policy_lock=lock, rollout_remaining=rollout_remaining, global_timesteps=global_timesteps, 
                        batch_id=batch_id, barrier_collect_done=barrier_collect_done, barrier_train_done=barrier_train_done, env_barrier_step=barrier_step, env_barrier_reset=barrier_reset,
                        rollout_pool=rollout_pool)

    def make_training_2():
        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=2, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, shared_buffer=None,  policy_lock=lock, rollout_remaining=rollout_remaining, global_timesteps=global_timesteps, 
                        batch_id=batch_id, barrier_collect_done=barrier_collect_done, barrier_train_done=barrier_train_done, env_barrier_step=barrier_step, env_barrier_reset=barrier_reset,
                        rollout_pool=rollout_pool)

    def make_training_3():
        env = AS2GymnasiumEnv(world_name="world3", world_size=10.0,
                              grid_size=200, min_distance=3.0, num_envs=1, num_drones=4, env_index=3, policy_type=args.policy_type,
                              shared_frontiers=shared_frontiers, lock=lock, barrier_reset=barrier_reset, barrier_step=barrier_step, condition=condition, queue=queue,
                              drones_initial_position=drones_initial_positions, vec_sync=vec_sync, step_lengths=step_lengths
                              )
        env = VecMonitor(env)
        return Training(env=env, policy=policy, shared_buffer=None,  policy_lock=lock, rollout_remaining=rollout_remaining, global_timesteps=global_timesteps, 
                        batch_id=batch_id, barrier_collect_done=barrier_collect_done, barrier_train_done=barrier_train_done, env_barrier_step=barrier_step, env_barrier_reset=barrier_reset,
                        rollout_pool=rollout_pool)

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
