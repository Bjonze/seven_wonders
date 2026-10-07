"""Play full games between bots and collect results / per-decision logs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .game import HALIKARNASSOS, SIDE, Game


@dataclass
class Decision:
    seat: int
    age: int
    turn: int
    phase: str
    hand: tuple[int, ...]  # card ids offered (with duplicates)
    action: int  # flat action index (card id * 3 + kind)


@dataclass
class GameResult:
    wonders: list[str]
    scores: list[int]
    breakdowns: list[dict[str, int]]
    winners: list[int]
    ranks: list[float]
    decisions: list[Decision] = field(default_factory=list)


def play_game(bots, num_players: int = 4, seed: int | None = None, record: bool = False,
              wonders=None, choose_sides: bool = False) -> GameResult:
    game = Game(num_players=num_players, seed=seed, wonders=wonders, choose_sides=choose_sides)
    decisions: list[Decision] = []
    while not game.over:
        actions = {}
        for seat in game.active:
            legal = game.legal_actions(seat)
            action = bots[seat].act(game, seat, legal)
            actions[seat] = action
            if record and game.phase != SIDE:
                offered = game.hands[seat] if game.phase != HALIKARNASSOS else game.discard
                decisions.append(Decision(seat, game.age, game.turn, game.phase,
                                          tuple(c.id for c in offered), action.index))
        game.step(actions)
    return GameResult(
        wonders=[p.wonder.label for p in game.players],
        scores=game.scores(),
        breakdowns=[game.score_breakdown(s) for s in range(num_players)],
        winners=game.winners(),
        ranks=game.ranks(),
        decisions=decisions,
    )


__all__ = ["play_game", "GameResult", "Decision"]
