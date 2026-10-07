from sevenwonders.game import BUILD, PAY_RIGHT, Action
from sevenwonders.resources import ZERO, min_purchase_cost, res, res_options

from .helpers import DEFAULT_WONDERS, card, give, make_game

FULL = (2,) * 7


def solve(cost, own=ZERO, own_choices=(), left=ZERO, left_choices=(), right=ZERO,
          right_choices=(), lp=FULL, rp=FULL, wl=1, wr=1):
    return min_purchase_cost(res(cost), list(own), [res_options(c) for c in own_choices],
                             list(left), [res_options(c) for c in left_choices],
                             list(right), [res_options(c) for c in right_choices], lp, rp, wl, wr)


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


def test_weighted_payment_avoids_one_neighbour():
    cheap_right = (1, 1, 1, 1, 2, 2, 2)
    # Cheapest buys the ore from the right (1 coin); avoiding the right costs 2.
    assert solve("O", left=res("O"), right=res("O"), rp=cheap_right) == (0, 1)
    assert solve("O", left=res("O"), right=res("O"), rp=cheap_right, wr=1001) == (2, 0)
    # Avoiding a neighbour who is the only seller is impossible: still buys there.
    assert solve("O", right=res("O"), wr=1001) == (0, 2)
    # Mixed: two ore needed, left has one, right has two.
    assert solve("OO", left=res("O"), right=res("OO"), wr=1001) == (2, 2)
    assert solve("OO", left=res("O"), right=res("OO"), wl=1001) == (0, 4)


def test_payment_variants_in_legal_actions():
    g = make_game(*DEFAULT_WONDERS)
    # Seat 1 (Rhodos) needs stone for Baths; give both neighbours stone.
    give(g, 2, "Stone Pit")
    g.hands[1] = [card("Baths")] + g.hands[1][1:]
    g._cache.clear()
    plain = [a for a in g.legal_actions(1) if a.card.name == "Baths" and a.kind == BUILD]
    assert len(plain) == 1  # without payment choice: only the cheapest
    variants = [a for a in g.legal_actions(1, payment_choice=True)
                if a.card.name == "Baths" and a.kind == BUILD]
    assert {a.pay for a in variants} == {0, PAY_RIGHT}  # cheapest buys left; pay-left is the same
    assert g.payment(1, Action(BUILD, card("Baths"), PAY_RIGHT)) == (0, 0, 2)
    before = [p.coins for p in g.players]
    actions = {s: Action(2, g.hands[s][-1]) for s in range(4)}
    actions[1] = Action(BUILD, card("Baths"), PAY_RIGHT)
    g.step(actions)
    assert g.players[2].coins == before[2] + 3 + 2  # right neighbour sold a card and got paid
    assert g.players[0].coins == before[0] + 3
    assert g.players[1].trade_paid == 2 and g.players[2].trade_received == 2


def test_variants_only_when_something_is_bought():
    g = make_game(*DEFAULT_WONDERS)
    for seat in range(4):
        for a in g.legal_actions(seat, payment_choice=True):
            if a.pay != 0:
                cheapest = g.payment(seat, Action(a.kind, a.card))
                assert cheapest[1] + cheapest[2] > 0
                assert g.payment(seat, a) != cheapest