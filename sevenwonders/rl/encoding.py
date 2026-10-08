"""Turn a game state into a fixed-size observation vector and a legal-action mask.

Everything is from the point of view of one seat. Players are ordered relative to that seat:
position 0 is the seat itself, 1 is its right neighbour, ..., n-1 is its left neighbour.
Only public information plus the seat's own hand is used.

Actions are indexed by card *name*. There are two layouts:
  * basic (v1/v2): index = card_id * 3 + kind (0 = build, 1 = wonder stage, 2 = sell);
    resources are always bought the cheapest way.
  * payment choice: index = card_id * 7 + slot, with slots 0-2 as above (cheapest payment)
    plus 3/4 = build paying left/right, 5/6 = wonder stage paying left/right
    (see PAY_LEFT / PAY_RIGHT in game.py).
During the Halikarnassos phase the "hand" is the discard pile and only builds are legal.

What a model sees is described by an EncodingConfig, stored with the model:
  * payment_choice: the 7-slot action layout above, plus features describing each payment
  * hidden_discard: the discard pile is face down, as in the real game. A seat sees only the
    cards it discarded itself and the size of the pile; Halikarnassos sees the whole pile
    while it picks a card from it. Without this flag (models v1-v3) every seat saw the
    whole pile all the time.
  * side_choice: the model also picks its wonder side in the SIDE phase. Adds two actions
    (Day, Night) after all others and one feature (the "choosing sides" flag) at the end of
    the observation, so an older model's weights carry over unchanged. While sides are
    being chosen, each player's wonder shows as half Day, half Night.
Models without side_choice get a random side (see `random_side`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..cards import NUM_CARD_IDS, NUM_COLORS
from ..game import (
    BABYLON, BUILD, CHEAPEST, CHOOSE_SIDE, HALIKARNASSOS, NUM_ACTION_KINDS, PAY_LEFT, PAY_RIGHT,
    PLAY, SIDE, SIDES, WONDER, Action, Game, SideChoice,
)
from ..resources import NUM_RESOURCES
from ..wonders import MAX_STAGES, NUM_WONDER_SIDES, WONDERS

# wonder name -> (day side id, night side id)
_SIDE_IDS = {name: tuple(next(w.id for w in WONDERS if w.name == name and w.side == s) for s in SIDES)
             for name in {w.name for w in WONDERS}}

NUM_ACTIONS = NUM_CARD_IDS * NUM_ACTION_KINDS
SLOTS_PAY = 7
NUM_ACTIONS_PAY = NUM_CARD_IDS * SLOTS_PAY
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
# Per hand card (one block of NUM_CARD_IDS per feature): buildable, free, coins needed; with
# payment choice also cheapest split (to left, to right) and, for the pay-left and pay-right
# options, (available, to left, to right).
HAND_FEATURES = {False: 3, True: 11}
# Wonder stage: payable, coins needed; with payment choice the same extra 8 as above.
WONDER_FEATURES = {False: 2, True: 10}


@dataclass(frozen=True)
class EncodingConfig:
    payment_choice: bool = False
    hidden_discard: bool = False
    side_choice: bool = False
    # Not a training setting: feed a model trained with the visible discard pile (v1-v3) only
    # the legal view (own discards), keeping its input layout. Used for fair comparisons.
    censor_discard: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def random_side(rng) -> SideChoice:
    """Side pick for bots that can't choose (old models, baseline bots)."""
    return SideChoice(rng.choice(SIDES))


def as_config(cfg: EncodingConfig | bool) -> EncodingConfig:
    """Accept a config, or a bool meaning payment_choice (older call style)."""
    return cfg if isinstance(cfg, EncodingConfig) else EncodingConfig(payment_choice=bool(cfg))


def model_encoding(model) -> EncodingConfig:
    """Encoding a model was trained with (older models only imply it by their output size)."""
    enc = getattr(model, "encoding", None)
    if enc:
        return EncodingConfig(**enc)
    return EncodingConfig(payment_choice=model.num_actions == NUM_ACTIONS_PAY)


def global_features(cfg: EncodingConfig | bool = False) -> int:
    cfg = as_config(cfg)
    return (
        3 + 7 + len(PHASES)   # age, turn (1-6, 7 = Babylon's extra card), phase
        + NUM_CARD_IDS        # hand counts
        + NUM_CARD_IDS * HAND_FEATURES[cfg.payment_choice]
        + WONDER_FEATURES[cfg.payment_choice]
        + NUM_CARD_IDS        # discard pile counts (hidden_discard: own discards only)
        + (1 if cfg.hidden_discard else 0)  # discard pile size
        + (1 if cfg.side_choice else 0)     # choosing sides
    )


def obs_dim(num_players: int, cfg: EncodingConfig | bool = False) -> int:
    return PLAYER_FEATURES * num_players + global_features(cfg)


def _card_actions(cfg: EncodingConfig) -> int:
    return NUM_ACTIONS_PAY if cfg.payment_choice else NUM_ACTIONS


def num_actions(cfg: EncodingConfig | bool) -> int:
    cfg = as_config(cfg)
    return _card_actions(cfg) + (len(SIDES) if cfg.side_choice else 0)


def uses_payment_choice(model_num_actions: int) -> bool:
    return model_num_actions == NUM_ACTIONS_PAY


def action_index(action: Action | SideChoice, cfg: EncodingConfig | bool) -> int:
    cfg = as_config(cfg)
    if isinstance(action, SideChoice):
        return _card_actions(cfg) + SIDES.index(action.side)
    if not cfg.payment_choice:
        return action.index
    if action.pay == CHEAPEST:
        slot = action.kind
    else:
        slot = 3 + 2 * action.kind + (action.pay - PAY_LEFT)
    return action.card.id * SLOTS_PAY + slot


def action_kind_and_pay(indices: np.ndarray, cfg: EncodingConfig | bool) -> tuple[np.ndarray, np.ndarray]:
    """Decode flat action indices into (kind, payment option) arrays (kind CHOOSE_SIDE for
    side picks)."""
    cfg = as_config(cfg)
    side = indices >= _card_actions(cfg)
    if not cfg.payment_choice:
        kind, pay = indices % NUM_ACTION_KINDS, np.zeros_like(indices)
    else:
        slot = indices % SLOTS_PAY
        basic = slot < 3
        kind = np.where(basic, slot, (slot - 3) // 2)
        pay = np.where(basic, CHEAPEST, PAY_LEFT + (slot - 3) % 2)
    return np.where(side, CHOOSE_SIDE, kind), np.where(side, CHEAPEST, pay)


def _scores(game: Game, seat: int) -> dict[str, int]:
    key = ("score", seat)
    cached = game._cache.get(key)
    if cached is None:
        cached = game.score_breakdown(seat)
        game._cache[key] = cached
    return cached


def _payment_features(obs: np.ndarray, off: int, stride: int, cheapest, alternatives) -> None:
    """Write cheapest split + (available, to left, to right) for pay-left and pay-right.

    Feature f goes to obs[off + f * stride]; f = 0 is the first of the 8 features."""
    obs[off] = cheapest[1] / 5.0
    obs[off + stride] = cheapest[2] / 5.0
    for i, pay in enumerate(alternatives):
        if pay is not None:
            base = off + (2 + 3 * i) * stride
            obs[base] = 1.0
            obs[base + stride] = pay[1] / 5.0
            obs[base + 2 * stride] = pay[2] / 5.0


def encode(game: Game, seat: int, cfg: EncodingConfig | bool = False
           ) -> tuple[np.ndarray, np.ndarray, dict[int, Action]]:
    """Return (observation, legal-action mask, {action index: Action})."""
    cfg = as_config(cfg)
    payment_choice = cfg.payment_choice
    choosing = game.phase == SIDE
    if choosing and not cfg.side_choice:
        raise ValueError("this encoding cannot choose sides; use random_side()")
    n = game.n
    obs = np.zeros(obs_dim(n, cfg), dtype=np.float32)
    off = 0
    for rel in range(n):
        p = game.players[(seat + rel) % n]
        for card in p.city:
            obs[off + card.id] = 1.0
        off += NUM_CARD_IDS
        if choosing:  # side not known yet
            for side_id in _SIDE_IDS[p.wonder.name]:
                obs[off + side_id] = 0.5
        else:
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

    if not choosing:
        obs[off + game.age - 1] = 1.0
        turn = 7 if game.phase == BABYLON else game.turn
        obs[off + 3 + turn - 1] = 1.0
        obs[off + 10 + PHASES.index(game.phase)] = 1.0
    off += 3 + 7 + len(PHASES)

    legal = game.legal_actions(seat, payment_choice)
    legal_keys = {(a.card.id, a.kind, a.pay) for a in legal if not choosing}

    def alternatives(card_id: int, kind: int, payment_fn) -> list:
        return [payment_fn(pay) if (card_id, kind, pay) in legal_keys else None
                for pay in (PAY_LEFT, PAY_RIGHT)]

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
                if payment_choice:
                    alts = alternatives(card.id, BUILD,
                                        lambda p, c=card: game.build_payment(seat, c, p))
                    _payment_features(obs, off + 3 * NUM_CARD_IDS + card.id, NUM_CARD_IDS, pay, alts)
    off += NUM_CARD_IDS * HAND_FEATURES[payment_choice]
    if game.phase not in (HALIKARNASSOS, SIDE):
        wpay = game.wonder_payment(seat)
        if wpay is not None:
            obs[off] = 1.0
            obs[off + 1] = sum(wpay) / 5.0
            if payment_choice and game.hands[seat]:
                alts = alternatives(game.hands[seat][0].id, WONDER,
                                    lambda p: game.wonder_payment(seat, p))
                _payment_features(obs, off + 2, 1, wpay, alts)
    off += WONDER_FEATURES[payment_choice]
    picking = game.phase == HALIKARNASSOS and seat in game.active
    if (cfg.hidden_discard or cfg.censor_discard) and not picking:
        for card, origin in zip(game.discard, game.discard_origin):
            if origin[0] == seat:
                obs[off + card.id] += 0.5
    else:
        for card in game.discard:
            obs[off + card.id] += 0.5
    off += NUM_CARD_IDS
    if cfg.hidden_discard:
        obs[off] = len(game.discard) / 20.0
        off += 1
    if cfg.side_choice:
        obs[off] = float(choosing)
        off += 1
    assert off == obs.shape[0]

    mask = np.zeros(num_actions(cfg), dtype=bool)
    actions: dict[int, Action | SideChoice] = {}
    for action in legal:
        index = action_index(action, cfg)
        mask[index] = True
        actions[index] = action
    return obs, mask, actions


__all__ = ["encode", "obs_dim", "num_actions", "uses_payment_choice", "action_index",
           "action_kind_and_pay", "EncodingConfig", "as_config", "model_encoding", "random_side",
           "NUM_ACTIONS", "NUM_ACTIONS_PAY"]
