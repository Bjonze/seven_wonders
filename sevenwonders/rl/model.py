"""Policy/value networks over the encoded observation, with masked action logits.

arch "mlp"    : the original 3-layer MLP (models v1-v3).
arch "resmlp" : an input layer followed by `layers` pre-norm residual blocks.
"""

from __future__ import annotations

import torch
from torch import nn

MASKED_LOGIT = -1e9


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.relu(self.fc1(self.norm(x))))


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 512, arch: str = "mlp",
                 layers: int = 3, encoding: dict | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.arch = arch
        self.layers = layers
        self.encoding = dict(encoding) if encoding else None
        if arch == "mlp":
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )
            head_in = hidden // 2
        elif arch == "resmlp":
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                *[ResidualBlock(hidden) for _ in range(layers)],
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
            head_in = hidden
        else:
            raise ValueError(f"unknown arch {arch}")
        self.policy_head = nn.Linear(head_in, num_actions)
        self.value_head = nn.Linear(head_in, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.policy_head(h).masked_fill(~mask, MASKED_LOGIT)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def config(self) -> dict:
        cfg = {"obs_dim": self.obs_dim, "num_actions": self.num_actions, "hidden": self.hidden,
               "arch": self.arch, "layers": self.layers}
        if self.encoding:
            cfg["encoding"] = self.encoding
        return cfg


def model_from_checkpoint(checkpoint: dict, device: str = "cpu") -> PolicyValueNet:
    model = PolicyValueNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_model(path: str, device: str = "cpu") -> PolicyValueNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return model_from_checkpoint(checkpoint, device)
