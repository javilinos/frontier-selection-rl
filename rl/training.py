import rclpy
import cProfile
import pstats
import torch as th
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from environments.as2_gymnasium_env_discrete_free import AS2GymnasiumEnv
from algorithms.policies.features_extractors.custom_cnn import CustomCombinedExtractor, NatureCNN_Mod

import argparse

from torch.distributions.constraints import Constraint
from torch.distributions import constraints


class CustomSimplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """

    event_dim = 1

    def check(self, value):

        return th.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-4)


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
    def __init__(self, env: AS2GymnasiumEnv, custom_callback: CustomCallback):
        self.env = env
        self.custom_callback = custom_callback

    # def mask_fn(env: AS2GymnasiumEnv) -> np.ndarray:
    #     # Do whatever you'd like in this function to return the action mask
    #     # for the current env. In this example, we assume the env has a
    #     # helpful method we can rely on.
    #     return env.valid_action_mask()

    def train(self, n_steps: int = 128, batch_size: int = 32, n_epochs: int = 5, learning_rate: float = 0.00005, pi_net_arch: list = [64, 64], vf_net_arch: list = [64, 64]):
        print(
            f"Training with n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, learning_rate={learning_rate}, pi_net_arch={pi_net_arch}, vf_net_arch={vf_net_arch}")
        model = MaskablePPO(
            MaskableActorCriticCnnPolicy,
            self.env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=0.97,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            policy_kwargs=dict(
                activation_fn=th.nn.ReLU,
                net_arch=dict(pi=pi_net_arch, vf=vf_net_arch),
                features_extractor_class=NatureCNN_Mod,
                share_features_extractor=True
            )
        )
        model.learn(
            total_timesteps=8,
            callback=self.custom_callback,
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

        model.save("ppo_as2_gymnasium")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform training of the model")
    parser.add_argument("--n_steps", type=int, default=8,
                        help="Number of steps in the environment")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--pi_net_arch", type=list,
                        default=[64, 64], help="Policy network architecture")
    parser.add_argument("--vf_net_arch", type=list,
                        default=[64, 64], help="Value function network architecture")
    args = parser.parse_args()

    rclpy.init()
    env = AS2GymnasiumEnv(world_name="world2", world_size=10.0,
                          grid_size=200, min_distance=1.0, num_envs=1, policy_type="CnnPolicy")
    #
    env = VecMonitor(env)
    # env = ActionMasker(env.venv, action_mask_fn=Training.mask_fn)
    print("Start mission")
    custom_callback = CustomCallback()
    training = Training(env, custom_callback)

    th.distributions.Categorical.arg_constraints = {
        "probs": CustomSimplex(), "logits": constraints.real_vector}  # Modify simplex constrain to be less restrictive

    print("Training the model...")
    training.train(n_steps=args.n_steps, batch_size=args.batch_size,
                   n_epochs=args.n_epochs, learning_rate=args.learning_rate,
                   pi_net_arch=args.pi_net_arch, vf_net_arch=args.vf_net_arch)
    rclpy.shutdown()
