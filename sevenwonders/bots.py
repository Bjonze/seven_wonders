"""Baseline (non-learning) players."""

from __future__ import annotations

import random

from .cards import BROWN, CARD_BY_ID, GREEN, GREY, PURPLE, RED, STAGE, WILD, YELLOW, Card
from .game import BUILD, SELL, WONDER, Action, Game, science_points
from .wonders import (
    BROWN_CHOICE, DISCARD_BUILD, FREE_FIRST_COLOR, FREE_FIRST_OF_AGE, FREE_LAST_OF_AGE,
    GREY_CHOICE, PLAY_LAST_CARD, SCIENCE_WILD,
)

# Cards that chain from each card name (used to value future free builds)
_CHAINS_INTO: dict[str, int] = {}
for _card in CARD_BY_ID:
    for _src in _card.chain_from:
        _CHAINS_INTO[_src] = _CHAINS_INTO.get(_src, 0) + 1


class Bot:
    name = "bot"

    def act(self, game: Game, seat: int, legal: list[Action]) -> Action:
        raise NotImplementedError


class RandomBot(Bot):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game, seat, legal):
        return self.rng.choice(legal)


class GreedyBot(Bot):
    """Picks the action with the best hand-written one-step value estimate."""

    name = "greedy"

    def __init__(self, seed: int | None = None, noise: float = 0.3):
        self.rng = random.Random(seed)
        self.noise = noise

    def act(self, game, seat, legal):
        return max(legal, key=lambda a: self.value(game, seat, a) + self.rng.random() * self.noise)

    # ------------------------------------------------------------------------------------
    def value(self, game: Game, seat: int, action: Action) -> float:
        if action.kind == SELL:
            return 1.0
        if action.kind == WONDER:
            pay = game.wonder_payment(seat)
            return self._stage_value(game, seat) - sum(pay) / 3.0
        pay = game.build_payment(seat, action.card)
        if pay is None:  # Halikarnassos free build
            pay = (0, 0, 0)
        return self._card_value(game, seat, action.card) - sum(pay) / 3.0

    def _stage_value(self, game: Game, seat: int) -> float:
        p = game.players[seat]
        stage = p.next_stage()
        v = stage.vp + stage.coins / 3.0 + self._military_value(game, seat, stage.shields)
        effect = stage.effect
        if effect in (BROWN_CHOICE, GREY_CHOICE):
            v += 2.0 * (3 - game.age) + 1.0
        elif effect == SCIENCE_WILD:
            v += science_points(p.science, p.science_wild + 1) - science_points(p.science, p.science_wild) + 2
        elif effect == DISCARD_BUILD:
            v += 4.0
        elif effect == PLAY_LAST_CARD:
            v += 2.0 * (4 - game.age)
        elif effect in (FREE_FIRST_COLOR, FREE_FIRST_OF_AGE, FREE_LAST_OF_AGE):
            v += 2.0 * (4 - game.age)
        return v + 0.5  # small bias: stages cost nothing card-wise

    def _military_value(self, game: Game, seat: int, shields: int) -> float:
        if shields == 0:
            return 0.0
        p = game.players[seat]
        v = 0.0
        for nb in (game.left(seat), game.right(seat)):
            theirs = game.players[nb].shields
            before = (p.shields > theirs) - (p.shields < theirs)
            after = (p.shields + shields > theirs) - (p.shields + shields < theirs)
            if after > before:
                v += {1: 1.0, 2: 2.0, 3: 3.0}[game.age] * (after - before)
        return v + 0.3 * shields

    def _card_value(self, game: Game, seat: int, card: Card) -> float:
        p = game.players[seat]
        age = game.age
        v = float(card.vp) + card.coins / 3.0
        if card.color in (BROWN, GREY):
            units = sum(card.produces) + (1 if card.choice else 0)
            new_types = [r for r in range(7) if card.produces[r] and p.fixed[r] == 0]
            v += units * 1.2 * (3 - age) + 1.5 * len(new_types) * (3 - age)
        if card.color == YELLOW:
            if card.coins_per is not None:
                targets, amount, scope = card.coins_per
                v += amount * (game._count(seat, targets, scope) + (targets != STAGE and card.color in targets)) / 3.0
            if card.vp_per is not None:
                targets, amount, scope = card.vp_per
                v += amount * (game._count(seat, targets, scope) + (targets != STAGE and card.color in targets))
            if card.discount_left_brown or card.discount_right_brown or card.discount_grey:
                v += 2.0 * (3 - age)
            if card.choice:
                v += 1.5 * (3 - age) + 1.0
        if card.color == RED:
            v += self._military_value(game, seat, card.shields)
        if card.color == GREEN:
            if card.science == WILD:
                v += science_points(p.science, p.science_wild + 1) - science_points(p.science, p.science_wild)
            else:
                after = list(p.science)
                after[card.science] += 1
                v += science_points(after, p.science_wild) - science_points(p.science, p.science_wild)
                v += 1.0 * (3 - age)  # future synergy
        if card.color == PURPLE:
            if card.vp_per is not None:
                targets, amount, scope = card.vp_per
                v += amount * (game._count(seat, targets, scope) + (targets != STAGE and card.color in targets))
            if card.vp_if_wonder_complete:
                remaining = len(p.wonder.stages) - p.stages_built
                v += card.vp_if_wonder_complete * (1.0 if remaining == 0 else 0.5 if remaining == 1 else 0.1)
            if card.science == WILD:
                v += science_points(p.science, p.science_wild + 1) - science_points(p.science, p.science_wild)
        if age < 3:
            v += 0.8 * _CHAINS_INTO.get(card.name, 0)
        return v
