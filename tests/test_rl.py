import numpy as np
import pytest
import torch

from sevenwonders.bots import RandomBot
from sevenwonders.game import Game
from sevenwonders.rl.agent import evaluate_games, summarize
from sevenwonders.rl.encoding import NUM_ACTIONS, encode, obs_dim
from sevenwonders.rl.model import PolicyValueNet
from sevenwonders.rl.rollout import collect, gae, terminal_rewards


@pytest.mark.parametrize("players", [3, 4, 7])
def test_encoding_matches_legal_actions_through_a_game(players):
    g = Game(num_players=players, seed=1)
    bots = [RandomBot(i) for i in range(players)]
    while not g.over:
        for seat in g.active:
            obs, mask, actions = encode(g, seat)
            assert obs.shape == (obs_dim(players),) and np.isfinite(obs).all()
            assert mask.shape == (NUM_ACTIONS,)
            legal = g.legal_actions(seat)
            assert set(np.flatnonzero(mask)) == {a.index for a in legal}
            assert all(actions[a.index] == a for a in legal)
        g.step({s: bots[s].act(g, s, g.legal_actions(s)) for s in g.active})


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


def test_evaluate_games_against_bots_and_policy():
    torch.manual_seed(0)
    model = PolicyValueNet(obs_dim(4), NUM_ACTIONS, hidden=64)
    for opponent in ("greedy", "random"):
        res = evaluate_games(model, opponent, [0, 1, 5], num_players=4)
        assert res["win"].shape == (3,) and res["breakdown"].shape == (3, 8)
    res = evaluate_games(model, "checkpoint", [2, 3], num_players=4, opponent_model=model)
    summary = summarize(res)
    assert 0.0 <= summary["win_rate"] <= 1.0 and "points_science" in summary
