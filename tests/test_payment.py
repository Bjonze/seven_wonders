from sevenwonders.game import BUILD, Action
from sevenwonders.resources import ZERO, min_purchase_cost, res, res_options

from .helpers import DEFAULT_WONDERS, card, give, make_game

FULL = (2,) * 7


def solve(cost, own=ZERO, own_choices=(), left=ZERO, left_choices=(), right=ZERO,
          right_choices=(), lp=FULL, rp=FULL):
    return min_purchase_cost(res(cost), list(own), [res_options(c) for c in own_choices],
                             list(left), [res_options(c) for c in left_choices],
                             list(right), [res_options(c) for c in right_choices], lp, rp)


def test_own_production_is_free():
    assert solve("WW", own=res("WW")) == (0, 0)
    assert solve("WS", own=res("W"), own_choices=["WS"]) == (0, 0)


def test_choice_card_only_covers_one_unit():
    assert solve("WS", own_choices=["WS"]) is None
    assert solve("WS", own_choices=["WS"], left=res("S")) == (2, 0)


def test_cheaper_neighbour_is_used_first():
    cheap_right = (2, 2, 2, 2, 2, 2, 2), (1, 1, 1, 1, 2, 2, 2)
    assert solve("O", left=res("O"), right=res("O"), lp=cheap_right[0], rp=cheap_right[1]) == (0, 1)


def test_buy_from_neighbour_choice_card():
    assert solve("CO", left_choices=["CO"], right=res("O")) == (2, 2)
    assert solve("CC", left_choices=["CO"]) is None


def test_own_choice_assignment_is_optimised():
    # Own W/S choice should cover S (no one sells S), wood is bought.
    assert solve("WS", own_choices=["WS"], left=res("W")) == (2, 0)


def test_game_trade_pays_neighbours():
    g = make_game(*DEFAULT_WONDERS)
    # Seat 0 (Gizah, stone) wants Baths (cost S): own production.
    assert g.build_payment(0, card("Baths")) == (0, 0, 0)
    # Seat 1 (Rhodos, ore) wants Baths: buys stone from left neighbour Gizah (seat 0).
    assert g.left(1) == 0
    assert g.build_payment(1, card("Baths")) == (0, 2, 0)
    g.hands[1] = [card("Baths")] + g.hands[1][1:]
    actions = {s: Action(2, g.hands[s][0]) for s in range(4)}  # everyone sells
    actions[1] = Action(BUILD, card("Baths"))
    before = [p.coins for p in g.players]
    g.step(actions)
    assert g.players[1].coins == before[1] - 2
    assert g.players[0].coins == before[0] + 3 + 2  # sold a card + trade income


def test_yellow_choice_is_not_sellable():
    g = make_game(*DEFAULT_WONDERS)
    give(g, 0, "Caravansery")
    # Seat 1 can only buy stone from Gizah's starting resource (1 unit).
    assert g._trade_cost(1, res("SS")) is None
    give(g, 0, "Timber Yard")
    assert g._trade_cost(1, res("SS")) == (4, 0)


def test_discounts():
    g = make_game(*DEFAULT_WONDERS)
    give(g, 1, "West Trading Post")  # left neighbour = seat 0
    assert g._trade_cost(1, res("S")) == (1, 0)
    give(g, 1, "Marketplace")
    give(g, 2, "Loom")
    assert g._trade_cost(1, res("L")) == (0, 1)


def test_cannot_afford():
    g = make_game(*DEFAULT_WONDERS)
    g.players[1].coins = 1
    assert g.build_payment(1, card("Baths")) is None
