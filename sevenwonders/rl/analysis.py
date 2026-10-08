"""Measure how good each card is, using a trained policy.

Two kinds of games, both with the policy in every seat and many games played side by side
so the network runs on batches:

observe():  the policy plays normally (sampling its moves, as in training). At every
    decision, each legal action is tried on a copy of the game (the other players' moves
    for that turn stay the same) and the value network rates the position that follows,
    i.e. the player's chance of winning. One row is recorded per card in the hand.

play_forced():  the policy plays its best move everywhere, except that the first time one
    seat could build card X it is made to build it (or, in a second run on the same deals,
    to sell it). The difference in that seat's win rate is X's played-out value.
"""

from __future__ import annotations

import numpy as np
import torch

from ..game import BABYLON, BUILD, CHEAPEST, PLAY, SELL, Action, Game
from .encoding import encode, model_encoding
from .model import PolicyValueNet

OBS_FIELDS = ("card", "age", "turn", "seat", "game", "buildable", "built", "q_build", "q_sell",
              "q_best_other", "win")
SEAT_FIELDS = ("game", "seat", "wonder", "score", "win")


def win_share(game: Game, seat: int) -> float:
    winners = game.winners()
    return 1.0 / len(winners) if seat in winners else 0.0


def _forward(model: PolicyValueNet, encoded: list) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.from_numpy(np.stack([e[0] for e in encoded]))
    mask = torch.from_numpy(np.stack([e[1] for e in encoded]))
    return model(obs, mask)


@torch.no_grad()
def observe(model: PolicyValueNet, seeds: list[int], num_players: int = 4,
            sample: bool = True, torch_seed: int = 0) -> dict[str, np.ndarray]:
    model.eval()
    pc = model_encoding(model)
    generator = torch.Generator().manual_seed(torch_seed)
    # Models that can choose their wonder side play the real setup (they pick sides)
    games = [Game(num_players=num_players, seed=s, choose_sides=pc.side_choice) for s in seeds]
    rows: list[list[float]] = []
    owner: list[tuple[int, int]] = []  # (game, seat) of each row, to fill in the outcome

    while True:
        pending = [(gi, seat) for gi, g in enumerate(games) if not g.over for seat in g.active]
        if not pending:
            break
        encoded = [encode(games[gi], seat, pc) for gi, seat in pending]
        logits, _ = _forward(model, encoded)
        if sample:
            choice = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)[:, 0]
        else:
            choice = logits.argmax(-1)
        joint: dict[int, dict[int, Action]] = {}
        for i, (gi, seat) in enumerate(pending):
            joint.setdefault(gi, {})[seat] = encoded[i][2][int(choice[i])]

        # Try every legal action of every deciding seat on a copy of its game.
        candidates = []  # (pending index, action, game after the turn)
        for i, (gi, seat) in enumerate(pending):
            g = games[gi]
            if g.phase not in (PLAY, BABYLON):
                continue
            for action in encoded[i][2].values():
                after = g.clone()
                after.step({**joint[gi], seat: action})
                candidates.append((i, action, after))
        q = np.zeros(len(candidates), dtype=np.float32)
        live = [k for k, c in enumerate(candidates) if not c[2].over]
        for k, c in enumerate(candidates):
            if c[2].over:
                q[k] = win_share(c[2], pending[c[0]][1])
        if live:
            enc_next = [encode(candidates[k][2], pending[candidates[k][0]][1], pc) for k in live]
            _, values = _forward(model, enc_next)
            q[live] = values.numpy()

        by_decision: dict[int, list[tuple[Action, float]]] = {}
        for k, (i, action, _) in enumerate(candidates):
            by_decision.setdefault(i, []).append((action, float(q[k])))
        for i, options in by_decision.items():
            gi, seat = pending[i]
            g = games[gi]
            chosen = joint[gi][seat]
            seen = set()
            for card in g.hands[seat]:
                if card.name in seen:
                    continue
                seen.add(card.name)
                builds = [v for a, v in options if a.kind == BUILD and a.card.name == card.name]
                sells = [v for a, v in options if a.kind == SELL and a.card.name == card.name]
                others = [v for a, v in options if not (a.kind == BUILD and a.card.name == card.name)]
                built = chosen.kind == BUILD and chosen.card.name == card.name
                rows.append([card.id, card.age, g.turn, seat, gi, float(bool(builds)), float(built),
                             max(builds) if builds else np.nan, sells[0] if sells else np.nan,
                             max(others) if others else np.nan, np.nan])
                owner.append((gi, seat))

        for gi, acts in joint.items():
            games[gi].step(acts)

    data = np.asarray(rows, dtype=np.float32).reshape(-1, len(OBS_FIELDS))
    wins = {(gi, s): win_share(g, s) for gi, g in enumerate(games) for s in range(num_players)}
    data[:, OBS_FIELDS.index("win")] = [wins[o] for o in owner]
    data[:, OBS_FIELDS.index("game")] = [seeds[gi] for gi, _ in owner]
    seats = np.asarray([[seeds[gi], s, g.players[s].wonder.id, g.scores()[s], wins[(gi, s)]]
                        for gi, g in enumerate(games) for s in range(num_players)], dtype=np.float32)
    return {"rows": data, "seats": seats}


@torch.no_grad()
def play_forced(model: PolicyValueNet, seeds: list[int], card_name: str, card_age: int,
                mode: str, num_players: int = 4) -> dict[str, np.ndarray]:
    """Deterministic (argmax) games. In game `seed`, seat `seed % num_players` plays normally
    until the first time it could build the card (`card_name`, Age `card_age`); at that
    decision it builds it (mode="build") or sells it (mode="sell"), then plays normally again.

    Everything before that decision is identical in both modes, so build-minus-sell on the
    same seeds is a paired, played-out measure of the card's value. Returns the seat's win
    share and whether the opportunity came up."""
    model.eval()
    pc = model_encoding(model)
    games = [Game(num_players=num_players, seed=s, choose_sides=pc.side_choice) for s in seeds]
    forced_seat = [s % num_players for s in seeds]
    opportunity = np.zeros(len(games), dtype=np.float32)
    opportunity_turn = np.zeros(len(games), dtype=np.int64)  # turn of the split (0 = none)
    kind = {"build": BUILD, "sell": SELL}[mode]
    while True:
        pending = [(gi, seat) for gi, g in enumerate(games) if not g.over for seat in g.active]
        if not pending:
            break
        encoded = [encode(games[gi], seat, pc) for gi, seat in pending]
        logits, _ = _forward(model, encoded)
        choice = logits.argmax(-1)
        joint: dict[int, dict[int, Action]] = {}
        for i, (gi, seat) in enumerate(pending):
            action = encoded[i][2][int(choice[i])]
            g = games[gi]
            if not opportunity[gi] and seat == forced_seat[gi] and g.phase in (PLAY, BABYLON):
                target = next((c for c in g.hands[seat] if c.name == card_name and c.age == card_age), None)
                if target is not None and g.build_payment(seat, target) is not None:
                    action = Action(kind, target, CHEAPEST)
                    opportunity[gi] = 1.0
                    opportunity_turn[gi] = 7 if g.phase == BABYLON else g.turn
            joint.setdefault(gi, {})[seat] = action
        for gi, acts in joint.items():
            games[gi].step(acts)
    win = np.asarray([win_share(g, forced_seat[gi]) for gi, g in enumerate(games)], dtype=np.float32)
    return {"seed": np.asarray(seeds), "win": win, "opportunity": opportunity,
            "opportunity_turn": opportunity_turn}
