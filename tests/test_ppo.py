import numpy as np
import torch

from sevenwonders.rl.model import PolicyValueNet
from sevenwonders.rl.ppo import PPOConfig, all_finite, ppo_update


def fake_batch(n: int, obs_dim: int = 12, actions: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, actions)) < 0.6
    mask[np.arange(n), 0] = True
    action = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
    return {
        "obs": rng.normal(size=(n, obs_dim)).astype(np.float32),
        "mask": mask,
        "action": action.astype(np.int64),
        "logp": np.full(n, -1.0, dtype=np.float32),
        "adv": rng.normal(size=n).astype(np.float32),
        "ret": rng.normal(size=n).astype(np.float32),
    }


def test_trailing_single_sample_minibatch_does_not_break_the_model():
    torch.manual_seed(0)
    model = PolicyValueNet(12, 5, hidden=16, arch="resmlp", layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = PPOConfig(minibatch=8, epochs=3, target_kl=0)
    for n in (17, 9, 8, 3):  # remainders of 1 sample, and batches smaller than a minibatch
        stats = ppo_update(model, opt, fake_batch(n), cfg, torch.device("cpu"))
        assert all_finite(model), n
        assert stats["skipped_updates"] == 0


def test_non_finite_loss_is_skipped():
    torch.manual_seed(0)
    model = PolicyValueNet(12, 5, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = fake_batch(16)
    batch["ret"][3] = np.inf
    stats = ppo_update(model, opt, batch, PPOConfig(minibatch=8, epochs=1, target_kl=0), torch.device("cpu"))
    assert all_finite(model)
    assert stats["skipped_updates"] >= 1
