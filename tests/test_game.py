import pytest

from sevenwonders.bots import GreedyBot, RandomBot
from sevenwonders.game import BABYLON, BUILD, HALIKARNASSOS, OVER, PLAY, SELL, WONDER, Action, Game
from sevenwonders.runner import play_game

from .helpers import DEFAULT_WONDERS, card, give, make_game


def everyone_sells(game, **overrides):
    actions = {s: Action(SELL, game.hands[s][0]) for s in game.active}
    actions.update(overrides)
    return actions


@pytest.mark.parametrize("players", [3, 4, 5, 6, 7])
def test_random_games_finish_with_consistent_state(players):
    for seed in range(40):
        bots = [RandomBot(seed * 10 + i) for i in range(players)]
        g = Game(num_players=players, seed=seed)
        while not g.over:
            assert all(p.coins >= 0 for p in g.players)
            legal = {s: g.legal_actions(s) for s in g.active}
            assert all(legal.values())
            g.step({s: bots[s].act(g, s, legal[s]) for s in g.active})
        assert g.total_cards() == 3 * 7 * players
        for p in g.players:
            assert len(p.names) == len(p.city)  # no duplicate buildings
        assert len(g.winners()) >= 1


def test_greedy_beats_random():
    wins = 0
    for seed in range(60):
        bots = [GreedyBot(seed)] + [RandomBot(seed * 10 + i) for i in range(3)]
        wins += 0 in play_game(bots, seed=seed).winners
    assert wins > 45


def test_hands_pass_left_then_right():
    g = make_game(*DEFAULT_WONDERS)
    hands = [list(h) for h in g.hands]
    actions = everyone_sells(g)
    g.step(actions)
    # Age I passes left: seat s now holds what its right neighbour had.
    for s in range(4):
        expected = list(hands[g.right(s)])
        expected.remove(actions[g.right(s)].card)
        assert g.hands[s] == expected
    while g.age == 1:
        g.step(everyone_sells(g))
    hands = [list(h) for h in g.hands]
    actions = everyone_sells(g)
    g.step(actions)
    for s in range(4):
        expected = list(hands[g.left(s)])
        expected.remove(actions[g.left(s)].card)
        assert g.hands[s] == expected


def test_age_ends_with_military_and_discards_last_cards():
    g = make_game(*DEFAULT_WONDERS)
    g.players[0].shields = 1
    for _ in range(6):
        g.step(everyone_sells(g))
    assert g.age == 2 and g.turn == 1
    assert g.players[0].military == [1, 1]
    assert len(g.discard) == 28  # 24 sold + 4 leftover cards


def test_chain_is_free_and_duplicates_are_illegal():
    g = make_game(*DEFAULT_WONDERS)
    give(g, 1, "Scriptorium")
    assert g.build_payment(1, card("Courthouse")) == (0, 0, 0)
    assert g.build_payment(1, card("Scriptorium")) is None
    give(g, 1, "Loom")
    assert g.build_payment(1, card("Loom", age=2)) is None


def test_ephesos_coins_and_wonder_stage():
    g = make_game(*DEFAULT_WONDERS)
    give(g, 2, "Clay Pool", "Clay Pit")
    g.step(everyone_sells(g) | {2: Action(WONDER, g.hands[2][0])})
    assert g.players[2].stages_built == 1
    give(g, 2, "Lumber Yard", "Sawmill")
    before = g.players[2].coins
    g.step(everyone_sells(g) | {2: Action(WONDER, g.hands[2][0])})
    assert g.players[2].coins == before + 9


def test_olympia_day_first_colour_free():
    g = make_game(("Olympia", "day"), *DEFAULT_WONDERS[1:])
    g.players[0].add_stage(card("Altar"))
    g.players[0].add_stage(card("Altar"))
    g.players[0].coins = 0
    assert g.build_payment(0, card("Palace")) == (0, 0, 0)
    give(g, 0, "Theater")
    assert g.build_payment(0, card("Palace")) is None


def test_olympia_night_first_and_last_card_of_age():
    g = make_game(("Olympia", "night"), *DEFAULT_WONDERS[1:])
    g.players[0].add_stage(card("Altar"))
    g.players[0].coins = 0
    assert g.turn == 1 and g.build_payment(0, card("Palace")) == (0, 0, 0)
    g.step(everyone_sells(g))
    g.players[0].coins = 0
    assert g.build_payment(0, card("Palace")) is None
    g.players[0].add_stage(card("Altar"))
    while g.turn < 6:
        g.step(everyone_sells(g))
    g.players[0].coins = 0
    g._cache.clear()
    assert g.build_payment(0, card("Palace")) == (0, 0, 0)


def test_babylon_night_plays_seventh_card():
    g = make_game(("Babylon", "night"), *DEFAULT_WONDERS[1:])
    g.players[0].add_stage(card("Altar"))
    for _ in range(6):
        g.step(everyone_sells(g))
    assert g.phase == BABYLON and g.active == [0]
    last = g.hands[0][0]
    g.step({0: Action(SELL, last)})
    assert g.phase == PLAY and g.age == 2
    assert len(g.discard) == 24 + 1 + 3  # 24 sold, Babylon's 7th sold, 3 leftovers


def test_halikarnassos_builds_from_discard():
    g = make_game(("Halikarnassos", "night"), *DEFAULT_WONDERS[1:])
    give(g, 0, "Clay Pool", "Clay Pit")
    g.step(everyone_sells(g) | {0: Action(WONDER, g.hands[0][0])})
    assert g.phase == HALIKARNASSOS and g.active == [0]
    options = g.legal_actions(0)
    assert options and all(a.kind == BUILD for a in options)
    choice = options[0]
    discard_before = len(g.discard)
    g.step({0: choice})
    assert choice.card.name in g.players[0].names
    assert len(g.discard) == discard_before - 1
    assert g.phase == PLAY and g.turn == 2


def test_game_over_scores():
    r = play_game([GreedyBot(i) for i in range(4)], seed=3)
    assert len(r.scores) == 4 and all(s == b["total"] for s, b in zip(r.scores, r.breakdowns))
    g = Game(seed=1)
    while not g.over:
        g.step({s: g.legal_actions(s)[0] for s in g.active})
    assert g.phase == OVER
