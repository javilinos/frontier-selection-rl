from torch import nn
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
import torch as th
import math
import numpy as np
import collections
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import os
import sys

sys.path.append(os.path.abspath(
    '/home/javilinos/Desktop/as2_projects/project_rl/rl/algorithms/policies/features_extractors'))
from custom_cnn import NatureCNN_Mod
from attention_network import AttentionExtractor


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor_cnn = self.make_features_extractor_cnn()
        self.features_extractor_flatten = self.make_features_extractor_flatten()
        self.features_dim_cnn = self.features_extractor_cnn.features_dim
        self.features_dim_flatten = self.features_extractor_flatten.features_dim
        # if self.share_features_extractor:
        self.pi_features_extractor_cnn = self.features_extractor_cnn
        self.vf_features_extractor_cnn = self.features_extractor_cnn
        self.pi_features_extractor_flatten = self.features_extractor_flatten
        self.vf_features_extractor_flatten = self.features_extractor_flatten
        # else:
        #     self.pi_features_extractor = self.features_extractor
        #     self.vf_features_extractor = self.make_features_extractor_cnn()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
            squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(
            lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim_flatten,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build_attention_extractor(self) -> None:

        self.attention_extractor = AttentionExtractor(
            self.features_dim_flatten,
            self.features_dim_cnn,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_attention_extractor()

        # latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
        #     self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        # else:
        #     raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.attention_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor_cnn: np.sqrt(2),
                self.features_extractor_flatten: np.sqrt(2),
                # self.attention_extractor.encoder: np.sqrt(2),
                # self.attention_extractor.actor_mlp: np.sqrt(2),
                # self.attention_extractor.critic_encoder: np.sqrt(2),
                # self.attention_extractor.critic_mlp: np.sqrt(2),
                # self.attention_extractor.actor_score: 0.01,  # Lower gain for final output layer
                # self.action_net: 0.01,
                self.value_net: 1,
            }
            # for name, module in self.attention_extractor.named_modules():
            #     if isinstance(module, nn.Linear):
            #         # Final output layer: actor_score
            #         if "actor_score" in name:
            #             gain = 0.01
            #         # Check if the module is part of a Transformer encoder (self-attention)
            #         elif "transformer_actor" in name or "transformer_critic" in name:
            #             if "in_proj" in name or "out_proj" in name:
            #                 gain = 1.0
            #             else:
            #                 gain = math.sqrt(2)
            #         else:
            #             gain = math.sqrt(2)

            #         nn.init.orthogonal_(module.weight, gain=gain)
            #         if module.bias is not None:
            #             nn.init.constant_(module.bias, 0)

            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor_cnn] = np.sqrt(2)
                module_gains[self.vf_features_extractor_cnn] = np.sqrt(2)
                module_gains[self.pi_features_extractor_flatten] = np.sqrt(2)
                module_gains[self.vf_features_extractor_flatten] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(
            1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def make_features_extractor_cnn(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return NatureCNN_Mod(self.observation_space, **self.features_extractor_kwargs)

    def make_features_extractor_flatten(self) -> BaseFeaturesExtractor:
        features_type = spaces.Box(low=0.0, high=1.0, shape=(
            5,), dtype=np.float32)
        return FlattenExtractor(features_type, **self.features_extractor_kwargs)

    def forward(self, obs: th.Tensor, frontier_features: List[th.Tensor], deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        frontier_features_list = []
        cnn_features = self.extract_features(
            self.observation_space, obs, self.features_extractor_cnn)
        for frontier_feature in frontier_features:
            flatenned_features = self.extract_features(spaces.Box(low=0.0, high=1.0, shape=(
                5,), dtype=np.float32),
                frontier_feature, self.features_extractor_flatten)
            # concat_features = th.cat((cnn_features, flatenned_features), dim=1)
            frontier_features_list.append(flatenned_features)

        # Shape (batch_size, num_candidates, n_features_dim)
        features = th.stack(frontier_features_list, dim=1)

        if self.share_features_extractor:
            logits_pi, latent_vf = self.attention_extractor([features], [cnn_features])

        print(f"shape of logits: {logits_pi.shape}")
        print(f"shape of latent_vf: {latent_vf.shape}")
        # else:
        #     pi_features, vf_features = features
        #     latent_pi = self.mlp_extractor.forward_actor(pi_features)
        #     latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(logits_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def extract_features(  # type: ignore[override]
        self, observation_space, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return self.extract_features_super(observation_space, obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = self.extract_features_super(
                observation_space, obs, self.pi_features_extractor)
            vf_features = self.extract_features_super(
                observation_space, obs, self.vf_features_extractor)
            return pi_features, vf_features

    def extract_features_super(self, observation_space, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use.
        :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(
            obs, observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def _get_action_dist_from_latent(self, logits_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        # mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(logits_pi, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=logits_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=logits_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=logits_pi)
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     return self.action_dist.proba_distribution(logits_pi, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def predict(self, observation: PyTorchObs, frontier_features: List[th.Tensor], deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        print(observation.shape)
        print(frontier_features)
        return self.get_distribution(observation, frontier_features).get_actions(deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, frontier_features: List[th.Tensor], deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.predict(observation, frontier_features, deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, frontier_features_batch: List[List[th.Tensor]], actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        # Preprocess the observation if needed
        frontier_features_batched = []
        cnn_features_batched = []
        log_prob_batched = []
        entropy_batched = []
        frontier_features_list = []
        cnn_features = self.extract_features(
            self.observation_space, obs, self.features_extractor_cnn)
        for i, frontier_features in enumerate(frontier_features_batch):
            for frontier_feature in frontier_features:
                flatenned_features = self.extract_features(spaces.Box(low=0.0, high=1.0, shape=(
                    5,), dtype=np.float32),
                    frontier_feature, self.features_extractor_flatten)
                # concat_features = th.cat((cnn_features[i].unsqueeze(0), flatenned_features), dim=1)
                frontier_features_list.append(flatenned_features)
            # Shape (batch_size, num_candidates, n_features_dim)
            features = th.stack(frontier_features_list, dim=1)
            frontier_features_batched.append(features)
            frontier_features_list = []

            logits_pi = self.attention_extractor.forward_actor(
                [features], [cnn_features[i].unsqueeze(0)])

            cnn_features_batched.append(cnn_features[i].unsqueeze(0))

            distribution = self._get_action_dist_from_latent(logits_pi)
            log_prob = distribution.log_prob(actions[i])
            log_prob_batched.append(log_prob)
            entropy = distribution.entropy()
            entropy_batched.append(entropy)

        latent_vf = self.attention_extractor.forward_critic(
            frontier_features_batched, cnn_features_batched)
        values = self.value_net(latent_vf)

        return values, th.cat(log_prob_batched, dim=0), th.cat(entropy_batched, dim=0)

    def get_distribution(self, obs: PyTorchObs, frontier_features: List[th.Tensor]) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        # Preprocess the observation if needed
        frontier_features_list = []
        cnn_features = self.extract_features(
            self.observation_space, obs, self.features_extractor_cnn)
        for frontier_feature in frontier_features:
            flatenned_features = self.extract_features(spaces.Box(low=0.0, high=1.0, shape=(
                5,), dtype=np.float32),
                frontier_feature, self.features_extractor_flatten)
            # concat_features = th.cat((cnn_features, flatenned_features), dim=1)
            frontier_features_list.append(flatenned_features)

        # Shape (batch_size, num_candidates, n_features_dim)
        features = th.stack(frontier_features_list, dim=1)

        logits_pi = self.attention_extractor.forward_actor([features], [cnn_features])

        return self._get_action_dist_from_latent(logits_pi)

    def predict_values(self, obs: PyTorchObs, frontier_features: List[th.Tensor]) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        frontier_features_list = []
        cnn_features = self.extract_features_super(
            self.observation_space, obs, self.features_extractor_cnn)

        for frontier_feature in frontier_features:
            flatenned_features = self.extract_features(spaces.Box(low=0.0, high=1.0, shape=(
                5,), dtype=np.float32),
                frontier_feature, self.features_extractor_flatten)
            # concat_features = th.cat((cnn_features, flatenned_features), dim=1)
            frontier_features_list.append(flatenned_features)

        # Shape (batch_size, num_candidates, n_features_dim)
        features = th.stack(frontier_features_list, dim=1)

        latent_vf = self.attention_extractor.forward_critic([features], [cnn_features])

        return self.value_net(latent_vf)


class ActorCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputActorCriticPolicy(ActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
