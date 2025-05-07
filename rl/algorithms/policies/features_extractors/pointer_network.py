import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Dict, Tuple, Union


class PointerExtractor(nn.Module):
    """
    Pointer network architecture that processes a variable number of candidate features
    (e.g., frontier point features) and outputs:
      - pointer_logits: unnormalized scores for each candidate (to be turned into a probability distribution)
      - latent_value: a latent representation for the critic (value function)

    This is intended as a drop-in replacement for an MLP extractor, but where the actor 
    network “points” to one of the candidate actions.

    :param candidate_feature_dim: Dimension of each candidate's feature vector.
    :param net_arch: Specification of the actor and value networks.
        Either a dict with keys "pi" and "vf" (lists of hidden layer sizes) or a list (shared architecture).
    :param activation_fn: The activation function (e.g. nn.ReLU).
    :param device: PyTorch device.
    """

    def __init__(
        self,
        candidate_feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super(PointerExtractor, self).__init__()
        device = th.device(device)
        self.device = device

        # Parse the network architecture
        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # ---------------------------
        # Actor network (pointer)
        # ---------------------------
        # First, project each candidate's features into an embedding space.
        pointer_hidden_dim = pi_layers_dims[0] if len(
            pi_layers_dims) > 0 else candidate_feature_dim
        self.encoder = nn.Linear(candidate_feature_dim, pointer_hidden_dim)

        # Build additional actor layers (if any)
        actor_layers = []
        last_dim = pointer_hidden_dim
        for layer_dim in pi_layers_dims[1:]:
            actor_layers.append(nn.Linear(last_dim, layer_dim))
            actor_layers.append(activation_fn())
            last_dim = layer_dim
        self.actor_mlp = nn.Sequential(*actor_layers) if actor_layers else nn.Identity()

        # A final linear layer produces a scalar score for each candidate.
        self.actor_score = nn.Linear(last_dim, 1)

        # ---------------------------
        # Critic network
        # ---------------------------
        # For the critic, we process each candidate and then pool over them.
        critic_hidden_dim = vf_layers_dims[0] if len(vf_layers_dims) > 0 else candidate_feature_dim
        self.critic_encoder = nn.Linear(candidate_feature_dim, critic_hidden_dim)
        critic_layers = []
        last_critic_dim = critic_hidden_dim
        for layer_dim in vf_layers_dims[1:]:
            critic_layers.append(nn.Linear(last_critic_dim, layer_dim))
            critic_layers.append(activation_fn())
            last_critic_dim = layer_dim
        self.critic_mlp = nn.Sequential(*critic_layers) if critic_layers else nn.Identity()

        # Save the latent dimension for the critic output (for use in a value head later)
        self.latent_dim_vf = last_critic_dim
        # For the pointer network the actor's latent "dim" is not fixed (depends on the number of candidates),
        # so we leave self.latent_dim_pi undefined.

    def forward(self, candidate_features: List[th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        :param candidate_features: A tensor of shape 
            (batch_size, num_candidates, candidate_feature_dim)
        :return: A tuple (pointer_logits, latent_value) where:
            - pointer_logits: Tensor of shape (batch_size, num_candidates) with unnormalized scores.
              (You can apply a softmax later to obtain probabilities.)
            - latent_value: Tensor of shape (batch_size, latent_dim_vf) for the critic.
        """
        # batch_size, num_candidates, _ = candidate_features.shape

        # --- Actor (Pointer) Processing ---
        # Encode each candidate feature

        # --- Critic Processing ---
        # Process candidate features and then pool (using mean pooling here; you can choose max or other)
        # (B, num_candidates, critic_hidden_dim)

        return self.forward_actor(candidate_features=candidate_features), self.forward_critic(candidate_features=candidate_features)

    def forward_actor(self, candidate_features: List[th.Tensor]):
        pointer_logits_list = []

        for candidate_feature in candidate_features:
            actor_emb = self.encoder(candidate_feature)  # (B, num_candidates, pointer_hidden_dim)
            actor_emb = F.relu(actor_emb)
            actor_emb = self.actor_mlp(actor_emb)         # (B, num_candidates, last_dim)
            # Compute a score for each candidate
            scores = self.actor_score(actor_emb)            # (B, num_candidates, 1)
            pointer_logits = scores.squeeze(-1)             # (B, num_candidates)

            pointer_logits_list.append(pointer_logits)

        pointer_logits = th.cat(pointer_logits_list, dim=0)  # (B, num_candidates)

        return pointer_logits

    def forward_critic(self, candidate_features: List[th.Tensor]):
        latent_values_list = []

        for candidate_feature in candidate_features:
            critic_emb = self.critic_encoder(candidate_feature)
            critic_emb = F.relu(critic_emb)
            pooled = critic_emb.mean(dim=1)                       # (B, critic_hidden_dim)
            latent_value = self.critic_mlp(pooled)                # (B, latent_dim_vf)

            latent_values_list.append(latent_value)

        latent_value = th.cat(latent_values_list, dim=0)     # (B, latent_dim_vf)

        return latent_value
