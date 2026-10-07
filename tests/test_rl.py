import numpy as np
import pytest
import torch

from sevenwonders.bots import RandomBot
from sevenwonders.game import Game
from sevenwonders.rl.agent import evaluate_games, summarize
from sevenwonders.rl.encoding import (
    NUM_ACTIONS, NUM_ACTIONS_PAY, action_index, action_kind_and_pay, encode, obs_dim,
)
from sevenwonders.rl.model import PolicyValueNet
from sevenwonders.rl.rollout import collect, gae, terminal_rewards


@pytest.mark.parametrize("payment_choice", [False, True])
@pytest.mark.parametrize("players", [3, 4, 7])
def test_encoding_matches_legal_actions_through_a_game(players, payment_choice):
    g = Game(num_players=players, seed=1)
    bots = [RandomBot(i) for i in range(players)]
    n_actions = NUM_ACTIONS_PAY if payment_choice else NUM_ACTIONS
    variants_seen = 0
    while not g.over:
        for seat in g.active:
            obs, mask, actions = encode(g, seat, payment_choice)
            assert obs.shape == (obs_dim(players, payment_choice),) and np.isfinite(obs).all()
            assert mask.shape == (n_actions,)
            legal = g.legal_actions(seat, payment_choice)
            indices = [action_index(a, payment_choice) for a in legal]
            assert len(set(indices)) == len(indices)
            assert set(np.flatnonzero(mask)) == set(indices)
            assert all(actions[i] == a for i, a in zip(indices, legal))
            kinds, pays = action_kind_and_pay(np.asarray(indices), payment_choice)
            assert list(kinds) == [a.kind for a in legal] and list(pays) == [a.pay for a in legal]
            variants_seen += sum(a.pay != 0 for a in legal)
        g.step({s: bots[s].act(g, s, g.legal_actions(s, payment_choice)) for s in g.active})
    assert (variants_seen > 0) == payment_choice


def test_gae_terminal_reward():
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    adv, ret = gae(values, 1.0, gamma=1.0, lam=1.0)
    assert np.allclose(ret, 1.0)  # Monte Carlo return with lam = 1
    assert np.allclose(adv, 0.5)


def test_rewards():
    g = Game(seed=0)
    while not g.over:
        g.step({s: g.legal_actions(s)[0] for s in g.active})
    win = terminal_rewards(g, "win")
    assert abs(sum(win) - 1.0) < 1e-6
    rank = terminal_rewards(g, "rank")
    assert max(rank) <= 1.0 and min(rank) >= -1.0


def test_collect_shapes():
    torch.manual_seed(0)
    model = PolicyValueNet(obs_dim(4), NUM_ACTIONS, hidden=64)
    data = collect(model, num_games=3, num_players=4, seed=0)
    n = len(data["action"])
    assert n >= 3 * 4 * 18  # at least 18 decisions per seat
    for key in ("obs", "mask", "logp", "value", "adv", "ret"):
        assert len(data[key]) == n
    assert data["mask"][np.arange(n), data["action"]].all()  # only legal actions sampled
    assert data["breakdowns"].shape == (3 * 4, 8)
    assert data["trade_paid"].shape == (3 * 4,)


def test_collect_and_play_with_payment_choice():
    torch.manual_seed(0)
    model = PolicyValueNet(obs_dim(4, True), NUM_ACTIONS_PAY, hidden=64)
    data = collect(model, num_games=2, num_players=4, seed=0)
    n = len(data["action"])
    assert data["obs"].shape[1] == obs_dim(4, True)
    assert data["mask"][np.arange(n), data["action"]].all()
    old = PolicyValueNet(obs_dim(4), NUM_ACTIONS, hidden=64)  # a v2-style opponent
    res = evaluate_games(model, "checkpoint", [0, 1], num_players=4, opponent_model=old)
    assert res["win"].shape == (2,)


def test_evaluate_games_against_bots_and_policy():
    torch.manual_seed(0)
    model = PolicyValueNet(obs_dim(4), NUM_ACTIONS, hidden=64)
    for opponent in ("greedy", "random"):
        res = evaluate_games(model, opponent, [0, 1, 5], num_players=4)
        assert res["win"].shape == (3,) and res["breakdown"].shape == (3, 8)
    res = evaluate_games(model, "checkpoint", [2, 3], num_players=4, opponent_model=model)
    summary = summarize(res)
    assert 0.0 <= summary["win_rate"] <= 1.0 and "points_science" in summary
