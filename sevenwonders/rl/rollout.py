"""Self-play data collection.

Many games run side by side in one process. Each tick, every seat that has to act (across
all unfinished games) is encoded, the batch goes through the network once, and actions are
sampled. When a game ends each seat's trajectory gets its terminal reward and GAE advantages.
"""

from __future__ import annotations

import numpy as np
import torch

from ..game import Game
from .agent import evaluate_games
from .encoding import SCORE_KEYS, encode
from .model import PolicyValueNet
from .priority import lower_priority


def terminal_rewards(game: Game, mode: str) -> list[float]:
    """Reward per seat at game end.

    win:  1 for the winner (shared on ties), 0 otherwise.
    rank: +1 for 1st down to -1 for last, linear in rank (ties share the average rank).
    """
    n = game.n
    if mode == "win":
        winners = game.winners()
        return [1.0 / len(winners) if s in winners else 0.0 for s in range(n)]
    if mode == "rank":
        return [1.0 - 2.0 * (r - 1) / (n - 1) for r in game.ranks()]
    raise ValueError(f"unknown reward mode {mode}")


def gae(values: np.ndarray, reward: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for one trajectory with only a terminal reward."""
    T = len(values)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        r = reward if t == T - 1 else 0.0
        delta = r + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


@torch.no_grad()
def collect(
    model: PolicyValueNet,
    num_games: int,
    num_players: int,
    seed: int,
    reward_mode: str = "win",
    gamma: float = 1.0,
    lam: float = 0.95,
) -> dict[str, np.ndarray]:
    model.eval()
    games = [Game(num_players=num_players, seed=seed + i) for i in range(num_games)]
    # trajectories[game][seat] = list of step indices into the flat buffers
    trajectories = [[[] for _ in range(num_players)] for _ in games]
    obs_buf, mask_buf, act_buf, logp_buf, val_buf = [], [], [], [], []

    while True:
        pending = [(gi, seat) for gi, g in enumerate(games) if not g.over for seat in g.active]
        if not pending:
            break
        encoded = [encode(games[gi], seat) for gi, seat in pending]
        obs = torch.from_numpy(np.stack([e[0] for e in encoded]))
        mask = torch.from_numpy(np.stack([e[1] for e in encoded]))
        logits, values = model(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logps = dist.log_prob(actions)

        step_actions: dict[int, dict] = {}
        for i, (gi, seat) in enumerate(pending):
            a = int(actions[i])
            trajectories[gi][seat].append(len(act_buf))
            obs_buf.append(encoded[i][0])
            mask_buf.append(encoded[i][1])
            act_buf.append(a)
            logp_buf.append(float(logps[i]))
            val_buf.append(float(values[i]))
            step_actions.setdefault(gi, {})[seat] = encoded[i][2][a]
        for gi, acts in step_actions.items():
            games[gi].step(acts)

    values = np.asarray(val_buf, dtype=np.float32)
    adv = np.zeros_like(values)
    ret = np.zeros_like(values)
    breakdowns = []
    for gi, g in enumerate(games):
        rewards = terminal_rewards(g, reward_mode)
        for seat in range(num_players):
            b = g.score_breakdown(seat)
            breakdowns.append([b[k] for k in SCORE_KEYS])
            idx = np.asarray(trajectories[gi][seat], dtype=np.int64)
            a, r = gae(values[idx], rewards[seat], gamma, lam)
            adv[idx] = a
            ret[idx] = r

    return {
        "obs": np.stack(obs_buf),
        "mask": np.stack(mask_buf),
        "action": np.asarray(act_buf, dtype=np.int64),
        "logp": np.asarray(logp_buf, dtype=np.float32),
        "value": values,
        "adv": adv,
        "ret": ret,
        # one row per (game, seat), games in order: reshape to (games, players, keys)
        "breakdowns": np.asarray(breakdowns, dtype=np.float32),
        "games": np.asarray([num_games]),
    }


def _load(model: PolicyValueNet, weights: dict[str, np.ndarray]) -> None:
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})


def worker_main(conn, num_players: int, model_config: dict, reward_mode: str,
                gamma: float, lam: float, priority: str | None = None) -> None:
    """Worker process: receives weights, plays self-play or evaluation games, sends results."""
    if priority:
        lower_priority(priority)
    torch.set_num_threads(1)
    model = PolicyValueNet(**model_config)
    while True:
        cmd, payload = conn.recv()
        if cmd == "close":
            break
        if cmd == "collect":
            weights, seed, num_games = payload
            _load(model, weights)
            conn.send(collect(model, num_games, num_players, seed, reward_mode, gamma, lam))
        elif cmd == "evaluate":
            weights, opponent, opponent_ckpt, game_ids, seed = payload
            _load(model, weights)
            opponent_model = None
            if opponent_ckpt is not None:
                opponent_config, opponent_weights = opponent_ckpt
                opponent_model = PolicyValueNet(**opponent_config)
                _load(opponent_model, opponent_weights)
            conn.send(evaluate_games(model, opponent, game_ids, num_players, seed, opponent_model))


__all__ = ["collect", "worker_main", "terminal_rewards", "gae"]
