# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
import torchvision
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState, Memory

class ResNetEncoder(nn.Module):
    """Pretrained ResNet encoder for RGB observations."""
    
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet
        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.imagenet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.shape[-1] == 3 or x.shape[-1] == 1:
        #     x = x.permute(0, 3, 1, 2)
        # if x.max() > 1.0:
        #     x = x / 255.0
        if x.ndim > 4:
            B = x.shape[0]
            T = x.shape[1]
            x = x.reshape(-1, 3, 128, 128)
            x = self.imagenet_transform(x)
            features = self.backbone(x)
            features = features.reshape(B, T, 512)
            # return features
        else:
            x = self.imagenet_transform(x)
            features = self.backbone(x)
        return features

class ActorCriticRecurrent(nn.Module):
    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        create_obs_encoder = False
        self.low_dim_keys = []
        self.rgb_keys = []
        for obs_group in obs_groups["policy"]:
            # assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
            for key in obs[obs_group].keys():
                if len(obs[obs_group][key].shape) == 4:
                    create_obs_encoder = True
                    obs_dim = 512
                else:
                    obs_dim = obs[obs_group][key].shape[-1]
                num_actor_obs += obs_dim
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        if create_obs_encoder:
            self.obs_encoder = ResNetEncoder()
            print(f"Created ResNet encoder for actor observations. New actor obs dim: {num_actor_obs}")

        # Actor
        self.memory_a = Memory(num_actor_obs, rnn_hidden_dim, rnn_num_layers, rnn_type)
        if self.state_dependent_std:
            self.actor = MLP(rnn_hidden_dim, [2, num_actions], actor_hidden_dims, activation)
            self.auxillary_actor = MLP(rnn_hidden_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
            self.auxillary_actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor RNN: {self.memory_a}")
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = False #actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.memory_c = Memory(num_critic_obs, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, activation)
        print(f"Critic RNN: {self.memory_c}")
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
                torch.nn.init.constant_(self.auxillary_actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
                torch.nn.init.constant_(
                    self.auxillary_actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                self.auxillary_std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                self.auxillary_log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev
    
    @property
    def auxillary_action_mean(self) -> torch.Tensor:
        return self.auxillary_distribution.mean

    @property
    def auxillary_action_std(self) -> torch.Tensor:
        return self.auxillary_distribution.stddev


    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _update_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        std = std.clamp(1e-6, 1e4)
        self.distribution = Normal(mean, std)
    
    def _update_auxillary_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.auxillary_actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.auxillary_actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.auxillary_std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.auxillary_log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.auxillary_distribution = Normal(mean, std)

    def act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs, masks, hidden_state).squeeze(0)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def auxillary_act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs, masks, hidden_state).squeeze(0)
        self._update_auxillary_distribution(out_mem)
        return self.auxillary_distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs).squeeze(0)
        if self.state_dependent_std:
            return self.actor(out_mem)[..., 0, :]
        else:
            return self.actor(out_mem)

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        out_mem = self.memory_c(obs, masks, hidden_state).squeeze(0)
        return self.critic(out_mem)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        features = [self.obs_encoder(obs[key]) if len(obs[key].shape) > 3 else obs[key] for key in obs.keys()]
        return torch.cat(features, dim=-1)


    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_auxillary_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.auxillary_distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
