import numpy as np
import torch

from sevenwonders.game import SELL, Action, Game
from sevenwonders.rl.agent import EnsembleBot, PolicyBot
from sevenwonders.rl.encoding import EncodingConfig, encode, num_actions, obs_dim
from sevenwonders.rl.model import PolicyValueNet
from sevenwonders.runner import play_game

V2 = EncodingConfig()
V3 = EncodingConfig(payment_choice=True)
V5 = EncodingConfig(payment_choice=True, hidden_discard=True, side_choice=True)


def net(cfg: EncodingConfig, seed: int) -> PolicyValueNet:
    torch.manual_seed(seed)
    enc = cfg.to_dict() if cfg != V2 and cfg != V3 else None  # old models stored no encoding
    return PolicyValueNet(obs_dim(4, cfg), num_actions(cfg), 32, encoding=enc)


def test_single_member_ensemble_plays_like_the_model():
    model = net(V5, 1)
    g1, g2 = Game(seed=4, choose_sides=True), Game(seed=4, choose_sides=True)
    solo, ens = PolicyBot(model), EnsembleBot([model])
    while not g1.over:
        a1 = {s: solo.act(g1, s, g1.legal_actions(s)) for s in g1.active}
        a2 = {s: ens.act(g2, s, g2.legal_actions(s)) for s in g2.active}
        assert a1 == a2
        g1.step(a1)
        g2.step(a2)


def test_mixed_layouts_play_full_games():
    members = [net(V2, 2), net(V3, 3), net(V5, 4)]
    for seed in range(3):
        bots = [EnsembleBot(members, seed=seed)] + [PolicyBot(net(V5, 5)) for _ in range(3)]
        result = play_game(bots, seed=seed, choose_sides=True)
        assert len(result.scores) == 4


def test_old_models_get_the_face_down_discard_view():
    ens = EnsembleBot([net(V3, 3)])
    cfg = ens.members[0][1]
    assert cfg.censor_discard and not cfg.hidden_discard
    g = Game(seed=6)
    g.step({s: Action(SELL, g.hands[s][0]) for s in g.active})
    full, censored = encode(g, 0, V3)[0], encode(g, 0, cfg)[0]
    assert len(full) == len(censored) and not np.array_equal(full, censored)
    assert censored[-77:].sum() == 0.5  # only seat 0's own sold card
