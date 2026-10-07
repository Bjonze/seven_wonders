from sevenwonders.game import science_points

from .helpers import DEFAULT_WONDERS, give, make_game


def test_science_points():
    assert science_points([1, 1, 1]) == 10
    assert science_points([2, 1, 0]) == 5
    assert science_points([3, 3, 3]) == 27 + 21
    assert science_points([4, 0, 0]) == 16


def test_science_wildcards_are_optimal():
    assert science_points([2, 2, 1], wild=1) == 26  # complete the third set
    assert science_points([4, 0, 0], wild=1) == 25  # pile on
    assert science_points([0, 0, 0], wild=2) == 4


def test_military():
    g = make_game(*DEFAULT_WONDERS)
    g.players[0].shields = 3
    g.players[1].shields = 1
    g.players[3].shields = 3
    g.age = 2
    g._resolve_military()
    assert g.players[0].military == [3]  # beats seat 1, ties seat 3
    assert g.players[1].military == [-1, 3]  # loses to 0, beats 2
    assert g.players[2].military == [-1, -1]


def test_breakdown_and_guilds():
    g = make_game(*DEFAULT_WONDERS)
    give(g, 0, "Altar", "Workers Guild", "Shipowners Guild", "Lumber Yard", "Loom")
    give(g, 1, "Stone Pit", "Clay Pool")  # left neighbour of 0 is seat 3, right is seat 1
    give(g, 3, "Ore Vein")
    g.players[0].coins = 7
    b = g.score_breakdown(0)
    assert b["civil"] == 3
    assert b["treasure"] == 2
    assert b["guilds"] == 3 + 4  # Workers: 3 neighbour browns; Shipowners: 1+1+2
    assert b["total"] == 3 + 2 + 7


def test_commerce_and_builders():
    g = make_game(*DEFAULT_WONDERS)
    give(g, 0, "Lighthouse", "Tavern", "Builders Guild", "Decorators Guild")
    g.players[0].stages_built = 3
    g.players[1].stages_built = 1
    b = g.score_breakdown(0)
    assert b["commerce"] == 2  # two yellow cards
    assert b["guilds"] == 4 + 7  # 4 stages around, wonder complete
    assert b["wonder"] == 3 + 5 + 7  # Gizah day


def test_ties_broken_by_coins():
    g = make_game(*DEFAULT_WONDERS)
    for p in g.players:
        p.coins = 0
    g.players[2].coins = 2  # same 0 VP from coins, but more coins
    assert g.winners() == [2]
