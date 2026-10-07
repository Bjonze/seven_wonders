"""PPO update on a batch of self-play data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import PolicyValueNet


@dataclass
class PPOConfig:
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 4096
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    target_kl: float = 0.03


def ppo_update(model: PolicyValueNet, optimizer: torch.optim.Optimizer,
               data: dict[str, np.ndarray], cfg: PPOConfig, device: torch.device) -> dict[str, float]:
    model.train()
    obs = torch.from_numpy(data["obs"]).to(device)
    mask = torch.from_numpy(data["mask"]).to(device)
    act = torch.from_numpy(data["action"]).to(device)
    old_logp = torch.from_numpy(data["logp"]).to(device)
    adv_all = torch.from_numpy(data["adv"]).to(device)
    ret = torch.from_numpy(data["ret"]).to(device)
    n = obs.shape[0]

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    stop = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start:start + cfg.minibatch]
            logits, value = model(obs[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[idx])
            log_ratio = logp - old_logp[idx]
            ratio = log_ratio.exp()
            adv = adv_all[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            v_loss = 0.5 * (value - ret[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
            stats["policy_loss"].append(pg_loss.item())
            stats["value_loss"].append(v_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)
            if cfg.target_kl and approx_kl > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["updates"] = len(stats["policy_loss"])
    return out
