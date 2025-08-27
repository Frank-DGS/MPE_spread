import math
from typing import Tuple

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_sizes: Tuple[int, int], activation: str) -> nn.Sequential:
    if activation == "relu":
        act = nn.ReLU
    elif activation == "tanh":
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(act())
        last_dim = hidden
    return nn.Sequential(*layers)


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, int], activation: str):
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_sizes, activation)
        self.action_head = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        x = self.backbone(obs)
        logits = self.action_head(x)
        return torch.distributions.Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    """
    集中式价值网络。可接受拼接后的全局观测。
    对于MPE spread（同构智能体），我们在训练时拼接所有agent观测形成central obs。
    """

    def __init__(self, central_obs_dim: int, hidden_sizes: Tuple[int, int], activation: str):
        super().__init__()
        self.backbone = build_mlp(central_obs_dim, hidden_sizes, activation)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, central_obs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(central_obs)
        return self.value_head(x)


