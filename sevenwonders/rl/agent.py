"""Use a trained network as a bot, and evaluate it against baseline bots or older policies."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from ..bots import Bot, GreedyBot, RandomBot
from ..runner import play_game
from .encoding import SCORE_KEYS, encode
from .model import PolicyValueNet


class PolicyBot(Bot):
    name = "policy"

    def __init__(self, model: PolicyValueNet, greedy: bool = True, seed: int | None = None):
        self.model = model
        self.greedy = greedy
        self.generator = torch.Generator().manual_seed(seed if seed is not None else 0)

    @torch.no_grad()
    def act(self, game, seat, legal):
        obs, mask, actions = encode(game, seat)
        logits, _ = self.model(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
        if self.greedy:
            index = int(logits[0].argmax())
        else:
            probs = torch.softmax(logits[0], dim=-1)
            index = int(torch.multinomial(probs, 1, generator=self.generator))
        return actions[index]


OPPONENTS = {"greedy": GreedyBot, "random": RandomBot}


def evaluate_games(model: PolicyValueNet, opponent: str, game_ids: Iterable[int],
                   num_players: int = 4, seed: int = 10_000,
                   opponent_model: PolicyValueNet | None = None) -> dict[str, np.ndarray]:
    """Play the policy in one seat (rotating with the game id) against `opponent` bots.

    `opponent` is "greedy", "random", or "checkpoint" (then `opponent_model` plays the other
    seats). Returns per-game arrays so results from several workers can be concatenated.
    """
    model.eval()
    wins, margins, breakdowns = [], [], []
    for g in game_ids:
        seat = g % num_players
        if opponent == "checkpoint":
            bots = [PolicyBot(opponent_model) for _ in range(num_players)]
        else:
            bots = [OPPONENTS[opponent](seed=seed + g * num_players + i) for i in range(num_players)]
        bots[seat] = PolicyBot(model)
        result = play_game(bots, num_players=num_players, seed=seed + g)
        wins.append(1.0 / len(result.winners) if seat in result.winners else 0.0)
        others = [s for i, s in enumerate(result.scores) if i != seat]
        margins.append(result.scores[seat] - max(others))
        breakdowns.append([result.breakdowns[seat][k] for k in SCORE_KEYS])
    return {
        "win": np.asarray(wins, dtype=np.float32),
        "margin": np.asarray(margins, dtype=np.float32),
        "breakdown": np.asarray(breakdowns, dtype=np.float32).reshape(-1, len(SCORE_KEYS)),
    }


def summarize(results: dict[str, np.ndarray]) -> dict[str, float]:
    breakdown = results["breakdown"]
    summary = {
        "win_rate": float(results["win"].mean()),
        "mean_margin": float(results["margin"].mean()),
        "mean_score": float(breakdown[:, SCORE_KEYS.index("total")].mean()),
    }
    for i, key in enumerate(SCORE_KEYS):
        if key != "total":
            summary[f"points_{key}"] = float(breakdown[:, i].mean())
    return summary


def evaluate(model: PolicyValueNet, opponent: str = "greedy", games: int = 200,
             num_players: int = 4, seed: int = 10_000,
             opponent_model: PolicyValueNet | None = None) -> dict[str, float]:
    return summarize(evaluate_games(model, opponent, range(games), num_players, seed, opponent_model))
