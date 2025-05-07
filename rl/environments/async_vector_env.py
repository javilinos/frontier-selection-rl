"""An async vector environment."""
import torch.multiprocessing as mp
import sys
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.spaces import Discrete
from gymnasium import logger
from gymnasium.core import Env, ObsType
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    clear_mpi_env_vars,
    concatenate,
    create_empty_array,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gymnasium.vector.vector_env import VectorEnv


__all__ = ["AsyncPPO"]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_CALL = "call"


class AsyncPPO():

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        shared_memory: bool = True,
        copy: bool = True,
        context: Optional[str] = None,
        daemon: bool = True,
        worker: Optional[Callable] = None,
    ):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through
                shared variables. This can improve the efficiency if the observations are large (e.g. images).
            copy: If ``True``, then the :meth:`~AsyncVectorEnv.reset` and :meth:`~AsyncVectorEnv.step` methods
                return a copy of the observations.
            context: Context for `multiprocessing`_. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if
                the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children,
                so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one.
                Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.

        Warnings:
            worker is an advanced mode option. It provides a high degree of flexibility and a high chance
            to shoot yourself in the foot; thus, if you are writing your own worker, it is recommended to start
            from the code for ``_worker`` (or ``_worker_shared_memory``) method, and add changes.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
            ValueError: If observation_space is a custom space (i.e. not a default space in Gym,
                such as gymnasium.spaces.Box, gymnasium.spaces.Discrete, or gymnasium.spaces.Dict) and shared_memory is True.
        """
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        # dummy_env = env_fns[0]()
        self.metadata = gym.Env.metadata

        # if (observation_space is None) or (action_space is None):
        #     observation_space = observation_space or dummy_env.observation_space
        #     action_space = action_space or dummy_env.action_space
        # dummy_env.close()
        # del dummy_env

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker
        target = worker or target
        self.num_envs = len(self.env_fns)
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT

    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `step_wait` times out.
                If `None` (default), the call to `step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_wait` without any prior call to `call_async`.
            TimeoutError: The call to `call_wait` has timed out after timeout second(s).
        """
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling `set_attr` while waiting for a pending call to complete.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def close_extras(
        self, timeout: Optional[Union[int, float]] = None, terminate: bool = False
    ):
        """Close the environments & clean up the extra resources (processes and pipes).

        Args:
            timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated.
            terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated.

        Raises:
            TimeoutError: If :meth:`close` timed out.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    f"Calling `close` while waiting for a pending call to `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def action_masks(self) -> np.ndarray:
        """Returns the action masks for the current environments.

        Returns:
            np.ndarray: The action mask for the current environments.
        """
        self.call_async("action_masks")
        return np.array(self.call_wait())

    def _poll(self, timeout=None):
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                f"Received the following error from Worker-{index}: {exctype.__name__}: {value}"
            )
            logger.error(f"Shutting down Worker-{index}.")
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error("Raising the last exception back to the main process.")
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)


def _worker(index, env_fn, pipe, parent_pipe, error_queue):
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, error_queue):
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
