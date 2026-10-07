"""Use a trained network as a bot, and evaluate it against baseline bots."""

from __future__ import annotations

import numpy as np
import torch

from ..bots import Bot, GreedyBot, RandomBot
from ..runner import play_game
from .encoding import encode
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


def evaluate(model: PolicyValueNet, opponent: str = "greedy", games: int = 200,
             num_players: int = 4, seed: int = 10_000, greedy: bool = True) -> dict[str, float]:
    """One policy seat against (num_players - 1) baseline bots; the policy's seat rotates."""
    model.eval()
    wins = 0.0
    margins, scores = [], []
    for g in range(games):
        seat = g % num_players
        bots = [OPPONENTS[opponent](seed=seed + g * num_players + i) for i in range(num_players)]
        bots[seat] = PolicyBot(model, greedy=greedy, seed=seed + g)
        result = play_game(bots, num_players=num_players, seed=seed + g)
        if seat in result.winners:
            wins += 1.0 / len(result.winners)
        others = [s for i, s in enumerate(result.scores) if i != seat]
        scores.append(result.scores[seat])
        margins.append(result.scores[seat] - max(others))
    return {
        "win_rate": wins / games,
        "mean_score": float(np.mean(scores)),
        "mean_margin": float(np.mean(margins)),
    }
