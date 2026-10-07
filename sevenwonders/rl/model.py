"""Policy/value network: an MLP over the encoded observation with masked action logits."""

from __future__ import annotations

import torch
from torch import nn

MASKED_LOGIT = -1e9


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden // 2, num_actions)
        self.value_head = nn.Linear(hidden // 2, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.policy_head(h).masked_fill(~mask, MASKED_LOGIT)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def config(self) -> dict:
        return {"obs_dim": self.obs_dim, "num_actions": self.num_actions, "hidden": self.hidden}


def load_model(path: str, device: str = "cpu") -> PolicyValueNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PolicyValueNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
