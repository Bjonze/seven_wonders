from sevenwonders.cards import ALL_CARDS
from sevenwonders.game import Game
from sevenwonders.wonders import wonder


def card(name, age=None):
    for c in ALL_CARDS:
        if c.name == name and (age is None or c.age == age):
            return c
    raise KeyError(name)


def make_game(*wonder_specs, seed=0):
    """Game with fixed wonders, e.g. make_game(("Gizah", "day"), ("Rhodos", "night"), ...)."""
    return Game(num_players=len(wonder_specs), seed=seed,
                wonders=[wonder(name, side) for name, side in wonder_specs])


DEFAULT_WONDERS = (("Gizah", "day"), ("Rhodos", "day"), ("Ephesos", "day"), ("Alexandria", "day"))


def give(game, seat, *names):
    """Put cards straight into a player's city (no payment)."""
    for name in names:
        game.players[seat].add_card(card(name))
    game._cache.clear()
