import numpy as np
import pytest
import torch

from sevenwonders.bots import GreedyBot, RandomBot
from sevenwonders.game import PLAY, SIDE, SIDES, SideChoice, Game
from sevenwonders.rl.agent import evaluate_games
from sevenwonders.rl.encoding import (
    NUM_ACTIONS_PAY, EncodingConfig, action_index, action_kind_and_pay, encode, num_actions, obs_dim,
)
from sevenwonders.rl.model import PolicyValueNet, warm_start
from sevenwonders.rl.rollout import collect
from sevenwonders.game import CHOOSE_SIDE

V4 = EncodingConfig(payment_choice=True, hidden_discard=True)
V5 = EncodingConfig(payment_choice=True, hidden_discard=True, side_choice=True)


def test_side_phase_comes_first_and_sets_wonders():
    g = Game(seed=3, choose_sides=True)
    assert g.phase == SIDE and g.active == [0, 1, 2, 3] and all(not h for h in g.hands)
    assert set(g.legal_actions(0)) == {SideChoice("day"), SideChoice("night")}
    names = [p.wonder.name for p in g.players]
    g.step({0: SideChoice("night"), 1: SideChoice("day"), 2: SideChoice("night"), 3: SideChoice("day")})
    assert g.phase == PLAY and g.age == 1 and all(len(h) == 7 for h in g.hands)
    assert [p.wonder.name for p in g.players] == names
    assert [p.wonder.side for p in g.players] == ["night", "day", "night", "day"]


def test_normal_deals_are_unchanged_by_the_new_options():
    a, b = Game(seed=11), Game(seed=11, choose_sides=False, repeat_wonders=False)
    assert [p.wonder.id for p in a.players] == [p.wonder.id for p in b.players]
    assert a.hands == b.hands


def test_repeated_wonders_always_repeat():
    for seed in range(50):
        names = [p.wonder.name for p in Game(seed=seed, repeat_wonders=True).players]
        assert len(set(names)) < 4


@pytest.mark.parametrize("repeat", [False, True])
def test_full_games_with_side_choice(repeat):
    for seed in range(10):
        g = Game(seed=seed, choose_sides=True, repeat_wonders=repeat)
        bots = [RandomBot(seed + i) for i in range(4)]
        while not g.over:
            g.step({s: bots[s].act(g, s, g.legal_actions(s)) for s in g.active})
        assert g.total_cards() == 84


def test_encoding_during_side_choice():
    g = Game(seed=5, choose_sides=True)
    obs, mask, actions = encode(g, 0, V5)
    assert obs.shape == (obs_dim(4, V5),) and obs[-1] == 1.0
    assert set(np.flatnonzero(mask)) == {NUM_ACTIONS_PAY, NUM_ACTIONS_PAY + 1}
    assert {a.side for a in actions.values()} == set(SIDES)
    kind, _ = action_kind_and_pay(np.flatnonzero(mask), V5)
    assert (kind == CHOOSE_SIDE).all()
    with pytest.raises(ValueError):
        encode(g, 0, V4)  # older encodings cannot choose
    g.step({s: SideChoice("day") for s in range(4)})
    obs, mask, actions = encode(g, 0, V5)
    assert obs[-1] == 0.0 and mask[:NUM_ACTIONS_PAY].any() and not mask[NUM_ACTIONS_PAY:].any()
    assert all(action_index(a, V5) == i for i, a in actions.items())


def test_warm_start_keeps_the_old_model_behaviour():
    torch.manual_seed(0)
    old = PolicyValueNet(obs_dim(4, V4), num_actions(V4), 64, "resmlp", 2, V4.to_dict())
    new = PolicyValueNet(obs_dim(4, V5), num_actions(V5), 64, "resmlp", 2, V5.to_dict())
    warm_start(new, old.state_dict())
    g = Game(seed=8)  # a normal game in the PLAY phase
    o4, m4, _ = encode(g, 1, V4)
    o5, m5, _ = encode(g, 1, V5)
    assert np.array_equal(o5[:len(o4)], o4) and np.array_equal(m5[:len(m4)], m4)
    with torch.no_grad():
        l4, v4 = old(torch.from_numpy(o4)[None], torch.from_numpy(m4)[None])
        l5, v5 = new(torch.from_numpy(o5)[None], torch.from_numpy(m5)[None])
    assert torch.allclose(l4, l5[:, :l4.shape[1]], atol=1e-5) and torch.allclose(v4, v5, atol=1e-5)


def test_collect_and_evaluate_with_sides_repeats_and_old_opponents():
    torch.manual_seed(0)
    learner = PolicyValueNet(obs_dim(4, V5), num_actions(V5), 64, "resmlp", 1, V5.to_dict())
    old = PolicyValueNet(obs_dim(4, V4), num_actions(V4), 64, "resmlp", 1, V4.to_dict())
    data = collect(learner, num_games=8, num_players=4, seed=1, opponents=[old], pool_frac=0.5,
                   opponent_weights=[1.0], choose_sides=True, repeat_frac=0.5)
    kind, _ = action_kind_and_pay(data["action"], V5)
    assert (kind == CHOOSE_SIDE).sum() > 0  # the learner's side picks are training data
    assert data["mask"][np.arange(len(data["action"])), data["action"]].all()
    res = evaluate_games(learner, "checkpoint", [0, 1], opponent_model=old, choose_sides=True)
    assert res["win"].shape == (2,)
    res = evaluate_games(learner, "greedy", [2], choose_sides=True)
    assert res["win"].shape == (1,)
    assert GreedyBot(0).act(Game(seed=1, choose_sides=True), 0, [SideChoice("day")]) == SideChoice("day")
