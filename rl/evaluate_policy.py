import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from algorithms.policies.custom_policy_attention import ActorCriticCnnPolicy, ActorCriticPolicy
from environments.as2_gymnasium_env_discrete_single_agent import AS2GymnasiumEnv
from stable_baselines3.common.utils import obs_as_tensor
import pandas as pd
import matplotlib.pyplot as plt


def fill_data(matrix):
    '''Grab a list of lists and make all the lists the same length, given the longest list, by filling with 1.0'''
    max_len = max(len(lst) for lst in matrix)
    for lst in matrix:
        while len(lst) < max_len:
            lst.append(1.0)


def get_std_dev(matrix):
    '''Grab a list of lists and return the standard deviation of the elements of each index in a new list'''
    std_vector = []
    for i in range(len(matrix[0])):
        column = [lst[i] for lst in matrix]
        std_vector.append(np.std(column))

    return std_vector


def get_mean(matrix):
    '''Grab a list of lists and return the mean of the elements of each index in a new list'''
    mean_vector = []
    for i in range(len(matrix[0])):
        column = [lst[i] for lst in matrix]
        mean_vector.append(np.mean(column))

    return mean_vector


def plot_path(obstacles, paths):
    GRID_SIZE = 20
    HALF = GRID_SIZE / 2

    fig, ax = plt.subplots(figsize=(6, 6))

    # 1) major ticks only at -10, 0, +10
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])

    # 2) minor ticks at every integer
    ax.set_xticks(range(-int(HALF), int(HALF) + 1), minor=True)
    ax.set_yticks(range(-int(HALF), int(HALF) + 1), minor=True)

    # 3) draw minor-grid (dashed) and major-grid (solid)
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.grid(which='major', linestyle='-', linewidth=1)

    # 4) hide all spines (no painted axis lines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 5) label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 6) plot obstacles (smaller circles)
    xo, yo = zip(*obstacles)
    ax.scatter(xo, yo,
               s=50,
               color='black',
               zorder=3)

    # 7) plot paths: blue, thicker, darker overlaps
    for path in paths:
        xs, ys = zip(*path)
        ax.plot(xs, ys,
                color='blue',
                linewidth=5,
                alpha=0.3,
                zorder=1)

    # 8) enforce aspect & limits
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)

    plt.tight_layout()
    plt.show()


def evaluate_policy(
    model: ActorCriticPolicy,
    env: Union[gym.Env, VecEnv, AS2GymnasiumEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    episodes = list(range(1, n_eval_episodes + 1))
    ###### matrix initialization #######
    area_explored_matrix = []
    cum_path_length_matrix = []

    step_path_length = []
    area_explored = []
    path_to_plot = []
    path_length_per_episode = []
    episode_reward = 0.0
    step_path_length.append(0)
    area_explored.append(env.area_explored)

    while (episode_counts < episode_count_targets).any():
        last_frontier_features = env.frontier_features()
        model.action_space = env.action_space
        features_list = []
        for frontier_feature in last_frontier_features:
            features_list.append(obs_as_tensor(
                frontier_feature, "cuda").unsqueeze(0))  # Frontier features
        obs_tensor = obs_as_tensor(observations, "cuda")  # Image
        actions = model._predict(
            obs_tensor,  # type: ignore[arg-type]
            features_list,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        step_path_length.append(env.path_length)
        area_explored.append(env.area_explored)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    area_explored_matrix.append(area_explored)
                    accumulated_path = np.cumsum(step_path_length)
                    cum_path_length_matrix.append(accumulated_path)
                    episode_path_length = np.sum(step_path_length)
                    path_length_per_episode.append(episode_path_length)
                    path_to_plot = env.episode_path
                    env.reset()
                    step_path_length = []
                    area_explored = []
                    step_path_length.append(0)
                    area_explored.append(env.area_explored)
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    # plot_path(env.obstacles, path_to_plot)

    # accumulated_path = np.cumsum(env.cum_path_length)
    # episodes = list(range(1, n_eval_episodes + 1))
    # accumulated_path = np.cumsum(step_path_length)
    # print(len(accumulated_path))
    # print(len(area_explored))
    # Save to CSV
    fill_data(area_explored_matrix)
    mean_area_explored = get_mean(area_explored_matrix)
    std_area_explored = get_std_dev(area_explored_matrix)
    # just return the biggest length list in the matrix cum_path_length_matrix
    distance = max(cum_path_length_matrix, key=len)
    df = pd.DataFrame({
        'distance': distance,
        'mean_area_explored': mean_area_explored,
        'std_area_explored': std_area_explored
    })
    df2 = pd.DataFrame({
        'episode': episodes,
        'path_length': path_length_per_episode
    })

    # df.to_csv('csv/time_graphics_10_episodes/ours.csv', index=False)
    # df2.to_csv('csv/bars_graphic_cum_mean_path_length/enormous_density/ours.csv', index=False)

    # # Optional: plot the data
    fig, ax = plt.subplots()

    # Plot mean (darkest color)
    ax.plot(df['distance'], df['mean_area_explored'],
            color='blue', alpha=1.0, label='Mean Area Explored')

    # Plot standard deviation as transparent band
    ax.fill_between(
        df['distance'],
        df['mean_area_explored'] - df['std_area_explored'],
        df['mean_area_explored'] + df['std_area_explored'],
        color='blue', alpha=0.3, label='Std Dev'
    )

    ax.set_xlabel('Distance')
    ax.set_ylabel('Area Explored')
    ax.set_title('Mean Area Explored with Standard Deviation')
    ax.legend()

    plt.show()

    env.drone_interface_list[0].shutdown()
    # Save to CSV
    # df = pd.DataFrame({
    #     'episode': episodes,
    #     'path_length': env.cum_path_length,
    #     'accumulated_path_length': accumulated_path
    # })
    # df.to_csv('csv/rl_data.csv', index=False)

    # # Optional: plot the data
    # plt.figure(figsize=(8, 5))
    # plt.plot(df['episode'], df['accumulated_path_length'], marker='o')
    # plt.xlabel('Episode')
    # plt.ylabel('Accumulated Path Length')
    # plt.title('Accumulated Path Length per Episode')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
