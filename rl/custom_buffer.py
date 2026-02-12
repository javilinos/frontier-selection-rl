import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    frontier_features: list[list[th.Tensor]]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


import numpy as np
import torch as th
from gymnasium import spaces
from typing import Generator, Optional, Union

from stable_baselines3.common.vec_env import VecNormalize

# Uses your existing RolloutBufferSamples NamedTuple and BaseBuffer


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms (PPO/A2C), modified for event-driven/SMDP PPO.

    Key features:
    - Stores per-transition 'distance' (or any duration proxy).
    - Computes TD/GAE using discount = gamma ** (distance / distance_unit).
    - Can act as:
        * individual per-agent buffer (collect transitions for one agent)
        * shared buffer that merges transitions from many individual buffers

    Recommended usage:
    - Individual buffers: n_envs=1, buffer_size = "max events you might collect before flush"
    - Shared buffer:      n_envs=1, buffer_size = N_total_events_for_update
    - When global event count reaches N_total_events_for_update:
        1) compute_returns_and_advantage for each individual buffer (with its own last_values)
        2) shared.reset()
        3) shared.add_from(individual_buffer_i) for each agent until shared.full
        4) train PPO on shared.get()
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        frontier_features_space: spaces.Space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        ),
        distance_unit: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        if n_envs != 1:
            # You *can* relax this, but your async/event-driven design usually wants n_envs=1 per agent buffer.
            raise ValueError(
                "This event-driven RolloutBuffer is intended for n_envs=1. "
                "Use one buffer per agent (and per env if you later vectorize)."
            )

        if distance_unit <= 0:
            raise ValueError("distance_unit must be > 0")

        self.gae_lambda = float(gae_lambda)
        self.gamma = float(gamma)
        self.frontier_features_space = frontier_features_space
        self.distance_unit = float(distance_unit)
        self.eps = float(eps)

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.frontier_features = []  # list of frontier-feature-lists, one per step
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # NEW: duration proxy per transition (distance traveled, meters)
        self.distances = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.generator_ready = False
        super().reset()

    # -------------------------
    # SMDP discounting helpers
    # -------------------------
    def _discount_from_distance(self, distance: np.ndarray) -> np.ndarray:
        """
        distance: shape (n_envs,) == (1,)
        discount = gamma ** (distance / distance_unit)
        """
        dist = np.maximum(distance.astype(np.float32), 0.0)
        units = dist / max(self.distance_unit, self.eps)
        # float64 internal for stability, then float32
        return np.power(self.gamma, units, dtype=np.float64).astype(np.float32)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Compute GAE(lambda) and returns using variable discount per step.

        last_values: shape [n_envs] == [1]
        dones: shape [n_envs] == [1] indicating terminal at the last transition boundary.
        """
        # Only compute over the actually filled part if buffer isn't full (individual buffers often are not full).
        n_steps = self.buffer_size if self.full else self.pos
        if n_steps == 0:
            return

        last_values_np = last_values.clone().cpu().numpy().flatten().astype(np.float32)

        last_gae_lam = np.zeros((self.n_envs,), dtype=np.float32)

        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32)
                next_values = self.values[step + 1]

            discount = self._discount_from_distance(self.distances[step])

            delta = self.rewards[step] + discount * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + discount * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns[:n_steps] = self.advantages[:n_steps] + self.values[:n_steps]

    # -------------------------
    # Normal add (collection)
    # -------------------------
    def add(
        self,
        obs: np.ndarray,
        frontier_features: list[np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        distance: Union[np.ndarray, float],
    ) -> None:
        """
        Add one event-driven transition.

        distance: traveled distance (meters) for this option/transition (or another duration proxy).
                  For n_envs=1 you can pass float.
        """
        if self.full:
            print("Warning: trying to add to a full buffer. Ignoring.")

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        # Normalize to shape (n_envs,) = (1,)
        if np.isscalar(distance):
            distance_arr = np.full((self.n_envs,), float(distance), dtype=np.float32)
        else:
            distance_arr = np.array(distance, dtype=np.float32).reshape((self.n_envs,))

        self.observations[self.pos] = np.array(obs, dtype=np.float32)
        self.frontier_features.append(frontier_features)
        self.actions[self.pos] = np.array(action, dtype=np.float32)
        self.rewards[self.pos] = np.array(reward, dtype=np.float32).reshape((self.n_envs,))
        self.episode_starts[self.pos] = np.array(episode_start, dtype=np.float32).reshape((self.n_envs,))

        self.values[self.pos] = value.detach().cpu().numpy().flatten().astype(np.float32)
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy().reshape((self.n_envs,))

        self.distances[self.pos] = distance_arr

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    # -------------------------
    # Merge helpers (individual -> shared)
    # -------------------------
    def export(self) -> dict:
        """Return a picklable snapshot (only filled part)."""
        n = self.buffer_size if self.full else self.pos
        return {
            "n": n,
            "observations": self.observations[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "episode_starts": self.episode_starts[:n].copy(),
            "values": self.values[:n].copy(),
            "log_probs": self.log_probs[:n].copy(),
            "advantages": self.advantages[:n].copy(),
            "returns": self.returns[:n].copy(),
            "distances": self.distances[:n].copy(),
            "frontier_features": list(self.frontier_features[:n]),
        }

    def add_from_export(self, data: dict) -> int:
        """Append exported transitions into this buffer."""
        if self.full:
            return 0
        n = int(data["n"])
        if n <= 0:
            return 0

        n = min(n, self.buffer_size - self.pos)
        dst = slice(self.pos, self.pos + n)
        src = slice(0, n)

        self.observations[dst] = data["observations"][src]
        self.actions[dst] = data["actions"][src]
        self.rewards[dst] = data["rewards"][src]
        self.episode_starts[dst] = data["episode_starts"][src]
        self.values[dst] = data["values"][src]
        self.log_probs[dst] = data["log_probs"][src]
        self.advantages[dst] = data["advantages"][src]
        self.returns[dst] = data["returns"][src]
        self.distances[dst] = data["distances"][src]

        self.frontier_features.extend(data["frontier_features"][:n])

        self.pos += n
        if self.pos == self.buffer_size:
            self.full = True

        self.generator_ready = False
        return n

    @staticmethod
    def finalize_individual_buffers_and_build_shared(
        shared_buffer: "RolloutBuffer",
        individual_buffers: list["RolloutBuffer"],
        last_values_per_agent: list[th.Tensor],
        dones: np.ndarray,
    ) -> None:
        """
        Convenience helper for your workflow:

        1) For each individual buffer i:
            buffer_i.compute_returns_and_advantage(last_values_per_agent[i], dones)
        2) shared_buffer.reset()
        3) shared_buffer.add_from(buffer_i) in order until full

        Requirements:
        - shared_buffer.buffer_size should equal your desired N_total_events for PPO update
        - individual buffers are n_envs=1
        - dones is shape (1,) (global done) or compatible
        """
        if len(individual_buffers) != len(last_values_per_agent):
            raise ValueError("last_values_per_agent must have one entry per individual buffer")

        # 1) compute per-agent adv/returns
        for buf, lv in zip(individual_buffers, last_values_per_agent):
            buf.compute_returns_and_advantage(lv, dones)

        # 2) fill shared buffer
        shared_buffer.reset()
        for buf in individual_buffers:
            if shared_buffer.full:
                break
            shared_buffer.add_from(buf)

        if not shared_buffer.full:
            raise RuntimeError(
                f"Shared buffer not full after merge: pos={shared_buffer.pos}, "
                f"buffer_size={shared_buffer.buffer_size}. "
                f"Either increase per-agent collection or lower shared buffer_size."
            )

    # -------------------------
    # Sampling (training)
    # -------------------------
    def get(self, batch_size: Optional[int] = None) -> Generator["RolloutBufferSamples", None, None]:
        assert self.full, "Shared buffer must be full before calling get()"

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            _tensor_names = ["observations", "actions", "values", "log_probs", "advantages", "returns"]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> "RolloutBufferSamples":
        frontier_features = [self.frontier_features[i] for i in batch_inds]

        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        data_torch = tuple(map(self.to_torch, data))
        frontier_features_ret = [[self.to_torch(f).unsqueeze(0) for f in ff] for ff in frontier_features]

        return RolloutBufferSamples(
            observations=data_torch[0],
            frontier_features=frontier_features_ret,
            actions=data_torch[1],
            old_values=data_torch[2],
            old_log_prob=data_torch[3],
            advantages=data_torch[4],
            returns=data_torch[5],
        )

