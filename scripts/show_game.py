"""Print one game played by a trained policy, decision by decision.

    python scripts/show_game.py checkpoints/v1/latest.pt --opponents greedy --seed 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sevenwonders.game import ACTION_KIND_NAMES, HALIKARNASSOS, Game  # noqa: E402
from sevenwonders.rl.agent import OPPONENTS, PolicyBot  # noqa: E402
from sevenwonders.rl.model import load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--opponents", default="policy", choices=["policy", *OPPONENTS])
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seat", type=int, default=0, help="seat whose hands are shown")
    parser.add_argument("--choose-sides", action="store_true", help="players pick their wonder sides")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    game = Game(num_players=args.players, seed=args.seed, choose_sides=args.choose_sides)
    bots = [PolicyBot(model) if args.opponents == "policy" or s == args.seat
            else OPPONENTS[args.opponents](seed=s) for s in range(args.players)]
    for s, p in enumerate(game.players):
        print(f"seat {s}: {p.wonder.label:<22} ({bots[s].name})")

    while not game.over:
        actions = {}
        for seat in game.active:
            action = bots[seat].act(game, seat, game.legal_actions(seat))
            actions[seat] = action
            if seat == args.seat and game.phase == "side":
                print(f"\nseat {seat} chooses {action}")
            elif seat == args.seat:
                offered = game.discard if game.phase == HALIKARNASSOS else game.hands[seat]
                hand = ", ".join(sorted(c.name for c in offered))
                print(f"\nAge {game.age} turn {game.turn} [{game.phase}] coins {game.players[seat].coins}")
                print(f"  hand: {hand}")
                print(f"  -> {ACTION_KIND_NAMES[action.kind]} {action.card.name}")
        game.step(actions)

    print("\nFinal scores:")
    winners = game.winners()
    for s in range(args.players):
        b = game.score_breakdown(s)
        parts = " ".join(f"{k} {v}" for k, v in b.items() if k != "total")
        mark = " WINNER" if s in winners else ""
        print(f"  seat {s} {game.players[s].wonder.label:<22} {b['total']:3d}  ({parts}){mark}")


if __name__ == "__main__":
    main()
