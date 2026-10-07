import numpy as np
import torch

from sevenwonders.bots import RandomBot
from sevenwonders.game import HALIKARNASSOS, SELL, Action, Game
from sevenwonders.rl.encoding import (
    NUM_ACTIONS, NUM_ACTIONS_PAY, EncodingConfig, encode, model_encoding, obs_dim,
)
from sevenwonders.rl.model import PolicyValueNet, model_from_checkpoint
from sevenwonders.rl.rollout import collect

from .helpers import give, make_game

HIDDEN = EncodingConfig(payment_choice=True, hidden_discard=True)


def play_until_discards(seed=3):
    g = Game(seed=seed)
    bots = [RandomBot(i) for i in range(4)]
    while len(g.discard) < 6:
        g.step({s: bots[s].act(g, s, g.legal_actions(s)) for s in g.active})
    return g


def test_other_players_discards_are_hidden():
    g = play_until_discards()
    seat = g.active[0]
    obs, _, _ = encode(g, seat, HIDDEN)
    # Swap a card another player discarded for a different card: this seat must not notice.
    other = next(i for i, o in enumerate(g.discard_origin) if o[0] != seat)
    replacement = next(c for c in g.hands[seat] if c.name != g.discard[other].name)
    g.discard[other] = replacement
    g._cache.clear()
    obs2, _, _ = encode(g, seat, HIDDEN)
    assert np.array_equal(obs, obs2)
    # ...but the old encoding (models v1-v3) did see it
    assert not np.array_equal(encode(g, seat, True)[0], obs)


def test_own_discards_and_pile_size_are_visible():
    g = Game(seed=4)
    before, _, _ = encode(g, 0, HIDDEN)
    acts = {s: Action(SELL, g.hands[s][0]) for s in g.active}
    g.step(acts)
    after, _, _ = encode(g, 0, HIDDEN)
    assert after[-1] > before[-1]  # pile size grew
    pile = after[-1 - 77:-1]
    assert pile[acts[0].card.id] == 0.5 and pile.sum() == 0.5  # only my own sold card


def test_halikarnassos_sees_whole_pile_while_picking():
    g = make_game(("Halikarnassos", "night"), ("Gizah", "day"), ("Rhodos", "day"), ("Ephesos", "day"))
    give(g, 0, "Clay Pool", "Clay Pit")
    g.step({0: Action(1, g.hands[0][0]), **{s: Action(SELL, g.hands[s][0]) for s in (1, 2, 3)}})
    assert g.phase == HALIKARNASSOS
    obs, mask, actions = encode(g, 0, HIDDEN)
    pile = obs[-1 - 77:-1]
    assert pile.sum() == 0.5 * len(g.discard)
    assert len(actions) == len(g.legal_actions(0))


def test_old_checkpoints_still_load_and_play():
    old = PolicyValueNet(obs_dim(4, True), NUM_ACTIONS_PAY, hidden=64)
    ckpt = {"model": old.state_dict(), "model_config": {"obs_dim": old.obs_dim, "num_actions": old.num_actions,
                                                        "hidden": 64}}
    loaded = model_from_checkpoint(ckpt)
    assert loaded.arch == "mlp" and model_encoding(loaded) == EncodingConfig(payment_choice=True)
    legacy = PolicyValueNet(obs_dim(4), NUM_ACTIONS, hidden=64)
    assert model_encoding(legacy) == EncodingConfig()


def test_resmlp_with_pool_opponents():
    torch.manual_seed(0)
    learner = PolicyValueNet(obs_dim(4, HIDDEN), NUM_ACTIONS_PAY, 64, "resmlp", 2, HIDDEN.to_dict())
    assert model_encoding(learner) == HIDDEN
    reloaded = model_from_checkpoint({"model": learner.state_dict(), "model_config": learner.config()})
    assert reloaded.config() == learner.config()
    old = PolicyValueNet(obs_dim(4), NUM_ACTIONS, hidden=64)  # v2-style opponent
    data = collect(learner, num_games=6, num_players=4, seed=0, opponents=[old], pool_frac=1.0)
    n = len(data["action"])
    assert data["obs"].shape == (n, obs_dim(4, HIDDEN))
    assert data["mask"][np.arange(n), data["action"]].all()
    assert len(data["pool_win"]) > 0 and len(data["breakdowns"]) == 0  # every game had pool seats
    # learner seats only: fewer than 4 trajectories per game
    assert n < 6 * 4 * 18
