import numpy as np
import torch

from sevenwonders.bots import RandomBot
from sevenwonders.game import Game
from sevenwonders.rl.analysis import OBS_FIELDS, SEAT_FIELDS, observe, play_forced
from sevenwonders.rl.encoding import NUM_ACTIONS_PAY, obs_dim
from sevenwonders.rl.model import PolicyValueNet


def tiny_model():
    torch.manual_seed(0)
    return PolicyValueNet(obs_dim(4, True), NUM_ACTIONS_PAY, hidden=32)


def test_clone_is_independent_and_replays_identically():
    g = Game(seed=5)
    bots = [RandomBot(i) for i in range(4)]
    while not g.over:
        c = g.clone()
        acts = {s: bots[s].act(g, s, g.legal_actions(s)) for s in g.active}
        g.step(acts)
        c.step(acts)
        assert [p.coins for p in g.players] == [p.coins for p in c.players]
        assert g.hands == c.hands and g.discard == c.discard and g.phase == c.phase
        assert [p.names for p in g.players] == [p.names for p in c.players]
    # Changing the clone must not touch the original
    g2 = Game(seed=6)
    c2 = g2.clone()
    c2.step({s: c2.legal_actions(s)[-1] for s in c2.active})
    assert g2.turn == 1 and all(len(h) == 7 for h in g2.hands)


def test_observe_records_every_card_in_hand():
    data = observe(tiny_model(), [11, 12])
    rows, seats = data["rows"], data["seats"]
    col = {k: i for i, k in enumerate(OBS_FIELDS)}
    assert rows.shape[1] == len(OBS_FIELDS) and seats.shape == (8, len(SEAT_FIELDS))
    buildable = rows[:, col["buildable"]] == 1
    assert not np.isnan(rows[buildable, col["q_build"]]).any()
    assert np.isnan(rows[~buildable, col["q_build"]]).all()
    assert not np.isnan(rows[:, col["q_sell"]]).any()
    wins = rows[:, col["win"]]
    assert ((wins >= 0) & (wins <= 1)).all()
    # one row per distinct card name in hand: 7 distinct-or-fewer at the first decision
    first = rows[(rows[:, col["turn"]] == 1) & (rows[:, col["age"]] == 1)]
    assert len(first) <= 2 * 4 * 7


def test_build_and_sell_runs_split_at_the_same_decision():
    model = tiny_model()
    seeds = list(range(20, 36))
    build = play_forced(model, seeds, "Lumber Yard", 1, "build")
    sell = play_forced(model, seeds, "Lumber Yard", 1, "sell")
    assert np.array_equal(build["opportunity"], sell["opportunity"])
    assert build["opportunity"].sum() > 0
    # Without the opportunity the two runs are the same game
    same = build["opportunity"] == 0
    assert np.array_equal(build["win"][same], sell["win"][same])
