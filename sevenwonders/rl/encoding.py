"""Turn a game state into a fixed-size observation vector and a legal-action mask.

Everything is from the point of view of one seat. Players are ordered relative to that seat:
position 0 is the seat itself, 1 is its right neighbour, ..., n-1 is its left neighbour.
Only public information plus the seat's own hand is used.

Actions are indexed by card *name* and kind: index = card_id * 3 + kind
(kind 0 = build, 1 = wonder stage, 2 = sell). During the Halikarnassos phase the "hand"
is the discard pile and only build actions are legal.
"""

from __future__ import annotations

import numpy as np

from ..cards import NUM_CARD_IDS, NUM_COLORS
from ..game import BABYLON, HALIKARNASSOS, NUM_ACTION_KINDS, PLAY, Action, Game
from ..resources import NUM_RESOURCES
from ..wonders import MAX_STAGES, NUM_WONDER_SIDES

NUM_ACTIONS = NUM_CARD_IDS * NUM_ACTION_KINDS
PHASES = (PLAY, BABYLON, HALIKARNASSOS)
SCORE_KEYS = ("military", "treasure", "wonder", "civil", "commerce", "guilds", "science", "total")

PLAYER_FEATURES = (
    NUM_CARD_IDS          # city (multi-hot)
    + NUM_WONDER_SIDES    # wonder side (one-hot)
    + MAX_STAGES + 1      # stages built (one-hot)
    + 1                   # stages remaining
    + NUM_COLORS          # cards per colour
    + NUM_RESOURCES * 2   # fixed production, tradeable fixed production
    + 4                   # science symbols + wildcards
    + 3                   # trade discounts
    + 2                   # own choice producers, tradeable choice producers
    + 3                   # coins, shields, military points
    + len(SCORE_KEYS)     # current score breakdown
)
GLOBAL_FEATURES = (
    3 + 7 + len(PHASES)   # age, turn (1-6, 7 = Babylon's extra card), phase
    + NUM_CARD_IDS        # hand counts
    + NUM_CARD_IDS * 3    # per hand card: buildable, free, coins needed
    + 2                   # wonder stage: payable, coins needed
    + NUM_CARD_IDS        # discard pile counts
)


def obs_dim(num_players: int) -> int:
    return PLAYER_FEATURES * num_players + GLOBAL_FEATURES


def _scores(game: Game, seat: int) -> dict[str, int]:
    key = ("score", seat)
    cached = game._cache.get(key)
    if cached is None:
        cached = game.score_breakdown(seat)
        game._cache[key] = cached
    return cached


def encode(game: Game, seat: int) -> tuple[np.ndarray, np.ndarray, dict[int, Action]]:
    """Return (observation, legal-action mask, {action index: Action})."""
    n = game.n
    obs = np.zeros(obs_dim(n), dtype=np.float32)
    off = 0
    for rel in range(n):
        p = game.players[(seat + rel) % n]
        for card in p.city:
            obs[off + card.id] = 1.0
        off += NUM_CARD_IDS
        obs[off + p.wonder.id] = 1.0
        off += NUM_WONDER_SIDES
        obs[off + p.stages_built] = 1.0
        off += MAX_STAGES + 1
        obs[off] = (len(p.wonder.stages) - p.stages_built) / MAX_STAGES
        off += 1
        for c in range(NUM_COLORS):
            obs[off + c] = p.color_counts[c] / 5.0
        off += NUM_COLORS
        for r in range(NUM_RESOURCES):
            obs[off + r] = p.fixed[r] / 3.0
            obs[off + NUM_RESOURCES + r] = p.tradeable_fixed[r] / 3.0
        off += NUM_RESOURCES * 2
        obs[off:off + 3] = p.science
        obs[off:off + 3] /= 3.0
        obs[off + 3] = p.science_wild
        off += 4
        obs[off] = p.discount_left_brown
        obs[off + 1] = p.discount_right_brown
        obs[off + 2] = p.discount_grey
        off += 3
        obs[off] = len(p.choices) / 2.0
        obs[off + 1] = len(p.tradeable_choices) / 2.0
        off += 2
        obs[off] = p.coins / 10.0
        obs[off + 1] = p.shields / 5.0
        obs[off + 2] = sum(p.military) / 10.0
        off += 3
        scores = _scores(game, p.seat)
        for i, key in enumerate(SCORE_KEYS):
            obs[off + i] = scores[key] / (50.0 if key == "total" else 15.0)
        off += len(SCORE_KEYS)

    obs[off + game.age - 1] = 1.0
    off += 3
    turn = 7 if game.phase == BABYLON else game.turn
    obs[off + turn - 1] = 1.0
    off += 7
    obs[off + PHASES.index(game.phase)] = 1.0
    off += len(PHASES)

    legal = game.legal_actions(seat)
    offered = game.discard if game.phase == HALIKARNASSOS else game.hands[seat]
    for card in offered:
        obs[off + card.id] += 1.0
    off += NUM_CARD_IDS
    if game.phase != HALIKARNASSOS:
        for card in game.hands[seat]:
            pay = game.build_payment(seat, card)
            if pay is not None:
                obs[off + card.id] = 1.0
                if sum(pay) == 0 and (card.coin_cost or any(card.cost)):
                    obs[off + NUM_CARD_IDS + card.id] = 1.0
                obs[off + 2 * NUM_CARD_IDS + card.id] = sum(pay) / 5.0
    off += NUM_CARD_IDS * 3
    if game.phase != HALIKARNASSOS:
        wpay = game.wonder_payment(seat)
        if wpay is not None:
            obs[off] = 1.0
            obs[off + 1] = sum(wpay) / 5.0
    off += 2
    for card in game.discard:
        obs[off + card.id] += 0.5
    off += NUM_CARD_IDS
    assert off == obs.shape[0]

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    actions: dict[int, Action] = {}
    for action in legal:
        mask[action.index] = True
        actions[action.index] = action
    return obs, mask, actions


__all__ = ["encode", "obs_dim", "NUM_ACTIONS"]
