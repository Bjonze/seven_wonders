"""Self-play data collection.

Many games run side by side in one process. Each tick, every seat that has to act (across
all unfinished games) is encoded, each network involved runs once on its batch, and actions
are sampled. When a game ends each learner seat's trajectory gets its terminal reward and
GAE advantages.

Opponent pool: with `opponents` and `pool_frac`, that share of the games hands 1-3 random
seats to one randomly chosen frozen opponent (an older version). Only the learner's seats
produce training data.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ..game import SIDE, Game
from .agent import evaluate_games
from .encoding import SCORE_KEYS, encode, model_encoding, random_side
from .model import PolicyValueNet
from .priority import lower_priority

LEARNER = -1


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


def _assign_controllers(num_games: int, num_players: int, num_opponents: int, pool_frac: float,
                        rng: random.Random, weights: list[float] | None = None) -> list[list[int]]:
    """controllers[game][seat] = LEARNER or an index into the opponent list (chosen with
    probability proportional to `weights`, uniform if None)."""
    controllers = []
    for _ in range(num_games):
        ctrl = [LEARNER] * num_players
        if num_opponents and rng.random() < pool_frac:
            opponent = rng.choices(range(num_opponents), weights=weights)[0]
            for seat in rng.sample(range(num_players), rng.randint(1, num_players - 1)):
                ctrl[seat] = opponent
        controllers.append(ctrl)
    return controllers


@torch.no_grad()
def collect(
    model: PolicyValueNet,
    num_games: int,
    num_players: int,
    seed: int,
    reward_mode: str = "win",
    gamma: float = 1.0,
    lam: float = 0.95,
    opponents: list[PolicyValueNet] | None = None,
    pool_frac: float = 0.0,
    opponent_weights: list[float] | None = None,
    choose_sides: bool = False,
    repeat_frac: float = 0.0,
) -> dict[str, np.ndarray]:
    """choose_sides: players pick wonder sides (networks that can't, and pool opponents of
    older versions, pick at random). repeat_frac: share of games dealt with repeated wonders
    (a training variant; statistics below only use normal games)."""
    model.eval()
    rng = random.Random(seed)
    opponents = opponents or []
    nets = {LEARNER: model, **dict(enumerate(opponents))}
    configs = {k: model_encoding(m) for k, m in nets.items()}
    repeated = [rng.random() < repeat_frac for _ in range(num_games)]
    games = [Game(num_players=num_players, seed=seed + i, choose_sides=choose_sides, repeat_wonders=repeated[i])
             for i in range(num_games)]
    controllers = _assign_controllers(num_games, num_players, len(opponents), pool_frac, rng,
                                      opponent_weights)
    # trajectories[game][seat] = list of step indices into the flat buffers (learner seats)
    trajectories = [[[] for _ in range(num_players)] for _ in games]
    obs_buf, mask_buf, act_buf, logp_buf, val_buf = [], [], [], [], []

    while True:
        pending = [(gi, seat) for gi, g in enumerate(games) if not g.over for seat in g.active]
        if not pending:
            break
        groups: dict[int, list[tuple[int, int]]] = {}
        for gi, seat in pending:
            groups.setdefault(controllers[gi][seat], []).append((gi, seat))
        step_actions: dict[int, dict] = {}
        for ctrl, members in groups.items():
            if not configs[ctrl].side_choice:
                for gi, seat in [m for m in members if games[m[0]].phase == SIDE]:
                    step_actions.setdefault(gi, {})[seat] = random_side(rng)
                members = [m for m in members if games[m[0]].phase != SIDE]
                if not members:
                    continue
            encoded = [encode(games[gi], seat, configs[ctrl]) for gi, seat in members]
            obs = torch.from_numpy(np.stack([e[0] for e in encoded]))
            mask = torch.from_numpy(np.stack([e[1] for e in encoded]))
            logits, values = nets[ctrl](obs, mask)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logps = dist.log_prob(actions) if ctrl == LEARNER else None
            for i, (gi, seat) in enumerate(members):
                a = int(actions[i])
                if ctrl == LEARNER:
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
    # Statistics: score breakdowns etc. from pure self-play games; learner results vs. the pool
    breakdowns, trade_paid, wonder_ids, seat_wins = [], [], [], []
    pool_opponent, pool_win = [], []
    for gi, g in enumerate(games):
        rewards = terminal_rewards(g, reward_mode)
        wins = terminal_rewards(g, "win")
        selfplay = all(c == LEARNER for c in controllers[gi]) and not repeated[gi]
        for seat in range(num_players):
            if controllers[gi][seat] != LEARNER:
                continue
            idx = np.asarray(trajectories[gi][seat], dtype=np.int64)
            a, r = gae(values[idx], rewards[seat], gamma, lam)
            adv[idx] = a
            ret[idx] = r
            if selfplay:
                b = g.score_breakdown(seat)
                breakdowns.append([b[k] for k in SCORE_KEYS])
                trade_paid.append(g.players[seat].trade_paid)
                wonder_ids.append(g.players[seat].wonder.id)
                seat_wins.append(wins[seat])
            elif any(c != LEARNER for c in controllers[gi]):
                pool_opponent.append(max(controllers[gi]))
                pool_win.append(wins[seat])

    return {
        "obs": np.stack(obs_buf),
        "mask": np.stack(mask_buf),
        "action": np.asarray(act_buf, dtype=np.int64),
        "logp": np.asarray(logp_buf, dtype=np.float32),
        "value": values,
        "adv": adv,
        "ret": ret,
        # one row per (self-play game, seat), games in order: reshape to (games, players, keys)
        "breakdowns": np.asarray(breakdowns, dtype=np.float32).reshape(-1, len(SCORE_KEYS)),
        "trade_paid": np.asarray(trade_paid, dtype=np.float32),
        "wonder": np.asarray(wonder_ids, dtype=np.int64),
        "win": np.asarray(seat_wins, dtype=np.float32),
        "pool_opponent": np.asarray(pool_opponent, dtype=np.int64),
        "pool_win": np.asarray(pool_win, dtype=np.float32),
        "repeated_games": np.asarray([sum(repeated)]),
        "games": np.asarray([num_games]),
    }


def _load(model: PolicyValueNet, weights: dict[str, np.ndarray]) -> None:
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})


def worker_main(conn, num_players: int, model_config: dict, reward_mode: str,
                gamma: float, lam: float, priority: str | None = None) -> None:
    """Worker process: receives weights, plays self-play or evaluation games, sends results.

    Opponent-pool models are cached by id; the trainer sends an opponent's weights only the
    first time this worker needs it."""
    if priority:
        lower_priority(priority)
    torch.set_num_threads(1)
    model = PolicyValueNet(**model_config)
    cache: dict[str, PolicyValueNet] = {}

    def opponent(entry) -> PolicyValueNet:
        key, config, weights = entry
        if weights is not None:
            cache[key] = PolicyValueNet(**config)
            _load(cache[key], weights)
        return cache[key]

    while True:
        cmd, payload = conn.recv()
        if cmd == "close":
            break
        if cmd == "collect":
            weights, seed, num_games, pool, options = payload
            _load(model, weights)
            opponents = [opponent(entry) for entry in pool]
            for key in set(cache) - {entry[0] for entry in pool}:
                del cache[key]
            conn.send(collect(model, num_games, num_players, seed, reward_mode, gamma, lam,
                              opponents, **options))
        elif cmd == "evaluate":
            weights, opponent_kind, opponent_ckpt, game_ids, seed, choose_sides = payload
            _load(model, weights)
            opponent_model = None
            if opponent_ckpt is not None:
                opponent_config, opponent_weights = opponent_ckpt
                opponent_model = PolicyValueNet(**opponent_config)
                _load(opponent_model, opponent_weights)
            conn.send(evaluate_games(model, opponent_kind, game_ids, num_players, seed, opponent_model,
                                     choose_sides))


__all__ = ["collect", "worker_main", "terminal_rewards", "gae", "LEARNER"]
