"""Age card data for the 2020 (2nd) edition base game.

Source: 7 Wonders fandom wiki card tables (2nd edition: Well, Ludus, Castrum, Decorators
Guild) cross-checked against the per-player-count deck sizes (7 cards per player per Age).
See DATA_NOTES.md for the few entries that sources disagreed on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .resources import ZERO, res, res_options

# Colours
BROWN, GREY, BLUE, YELLOW, RED, GREEN, PURPLE = range(7)
NUM_COLORS = 7
COLOR_NAMES = ("brown", "grey", "blue", "yellow", "red", "green", "purple")

# Science symbols (WILD = choose at game end)
COMPASS, TABLET, GEAR = range(3)
WILD = 3
SCIENCE_NAMES = ("compass", "tablet", "gear")

# Special "per X" targets for coins/VP counting
STAGE = "stage"  # wonder stages built
# scopes
SELF = "self"
NEIGHBORS = "neighbors"
ALL3 = "self+neighbors"


@dataclass(frozen=True, eq=False)
class Card:
    name: str
    age: int
    color: int
    players: tuple[int, ...]  # one entry per copy: min player count for that copy
    cost: tuple[int, ...] = ZERO
    coin_cost: int = 0
    chain_from: tuple[str, ...] = ()
    produces: tuple[int, ...] = ZERO  # fixed production
    choice: tuple[int, ...] = ()  # produces one of these per turn
    vp: int = 0
    shields: int = 0
    science: int = -1
    coins: int = 0  # immediate coins
    # immediate coins per counted thing: (targets, amount, scope)
    coins_per: tuple | None = None
    # end-game VP per counted thing: (targets, amount, scope)
    vp_per: tuple | None = None
    # trade discounts
    discount_left_brown: bool = False
    discount_right_brown: bool = False
    discount_grey: bool = False
    vp_if_wonder_complete: int = 0
    id: int = field(default=-1, compare=False)

    @property
    def tradeable(self) -> bool:
        """Neighbours may buy resources from brown and grey cards only."""
        return self.color in (BROWN, GREY)

    def __repr__(self) -> str:
        return f"Card({self.name}, age {self.age})"


def _c(name, age, color, players, **kw):
    if "cost" in kw:
        kw["cost"] = res(kw["cost"])
    if "produces" in kw:
        kw["produces"] = res(kw["produces"])
    if "choice" in kw:
        kw["choice"] = res_options(kw["choice"])
    if isinstance(kw.get("chain_from"), str):
        kw["chain_from"] = (kw["chain_from"],)
    return Card(name=name, age=age, color=color, players=tuple(players), **kw)


_AGE1 = [
    # Brown
    _c("Lumber Yard", 1, BROWN, (3, 4), produces="W"),
    _c("Stone Pit", 1, BROWN, (3, 5), produces="S"),
    _c("Clay Pool", 1, BROWN, (3, 5), produces="C"),
    _c("Ore Vein", 1, BROWN, (3, 4), produces="O"),
    _c("Tree Farm", 1, BROWN, (6,), coin_cost=1, choice="WC"),
    _c("Excavation", 1, BROWN, (4,), coin_cost=1, choice="SC"),
    _c("Clay Pit", 1, BROWN, (3,), coin_cost=1, choice="CO"),
    _c("Timber Yard", 1, BROWN, (3,), coin_cost=1, choice="WS"),
    _c("Forest Cave", 1, BROWN, (5,), coin_cost=1, choice="WO"),
    _c("Mine", 1, BROWN, (6,), coin_cost=1, choice="SO"),
    # Grey
    _c("Loom", 1, GREY, (3, 6), produces="L"),
    _c("Glassworks", 1, GREY, (3, 6), produces="G"),
    _c("Press", 1, GREY, (3, 6), produces="P"),
    # Blue
    _c("Altar", 1, BLUE, (3, 5), vp=3),
    _c("Theater", 1, BLUE, (3, 6), vp=3),
    _c("Well", 1, BLUE, (4, 7), vp=3),
    _c("Baths", 1, BLUE, (3, 7), cost="S", vp=3),
    # Yellow
    _c("Tavern", 1, YELLOW, (4, 5, 7), coins=5),
    _c("East Trading Post", 1, YELLOW, (3, 7), discount_right_brown=True),
    _c("West Trading Post", 1, YELLOW, (3, 7), discount_left_brown=True),
    _c("Marketplace", 1, YELLOW, (3, 6), discount_grey=True),
    # Red
    _c("Guard Tower", 1, RED, (3, 4), cost="C", shields=1),
    _c("Barracks", 1, RED, (3, 5), cost="O", shields=1),
    _c("Stockade", 1, RED, (3, 7), cost="W", shields=1),
    # Green
    _c("Scriptorium", 1, GREEN, (3, 4), cost="P", science=TABLET),
    _c("Apothecary", 1, GREEN, (3, 5), cost="L", science=COMPASS),
    _c("Workshop", 1, GREEN, (3, 7), cost="G", science=GEAR),
]

_AGE2 = [
    # Brown
    _c("Sawmill", 2, BROWN, (3, 4), coin_cost=1, produces="WW"),
    _c("Quarry", 2, BROWN, (3, 4), coin_cost=1, produces="SS"),
    _c("Brickyard", 2, BROWN, (3, 4), coin_cost=1, produces="CC"),
    _c("Foundry", 2, BROWN, (3, 4), coin_cost=1, produces="OO"),
    # Grey (same names as Age I: you can't own both copies)
    _c("Loom", 2, GREY, (3, 5), produces="L"),
    _c("Glassworks", 2, GREY, (3, 5), produces="G"),
    _c("Press", 2, GREY, (3, 5), produces="P"),
    # Blue
    _c("Courthouse", 2, BLUE, (3, 5), cost="CCL", chain_from="Scriptorium", vp=4),
    _c("Temple", 2, BLUE, (3, 6), cost="WCG", vp=4),
    _c("Statue", 2, BLUE, (3, 7), cost="OOW", chain_from="Well", vp=4),
    _c("Aqueduct", 2, BLUE, (3, 7), cost="SSS", chain_from="Baths", vp=5),
    # Yellow
    _c("Caravansery", 2, YELLOW, (3, 5, 6), cost="WW", chain_from="Marketplace", choice="WSCO"),
    _c("Forum", 2, YELLOW, (3, 6, 7), cost="CC",
       chain_from=("East Trading Post", "West Trading Post"), choice="GLP"),
    _c("Vineyard", 2, YELLOW, (3, 6), coins_per=((BROWN,), 1, ALL3)),
    _c("Bazaar", 2, YELLOW, (4, 7), coins_per=((GREY,), 2, ALL3)),
    # Red
    _c("Stables", 2, RED, (3, 5), cost="WOC", chain_from="Apothecary", shields=2),
    _c("Archery Range", 2, RED, (3, 6), cost="WWO", chain_from="Workshop", shields=2),
    _c("Walls", 2, RED, (3, 7), cost="SSS", shields=2),
    _c("Training Ground", 2, RED, (4, 6, 7), cost="OOW", shields=2),
    # Green
    _c("Dispensary", 2, GREEN, (3, 4), cost="OOG", chain_from="Apothecary", science=COMPASS),
    _c("Laboratory", 2, GREEN, (3, 5), cost="CCP", chain_from="Workshop", science=GEAR),
    _c("Library", 2, GREEN, (3, 6), cost="SSL", chain_from="Scriptorium", science=TABLET),
    _c("School", 2, GREEN, (3, 7), cost="WP", science=TABLET),
]

_AGE3 = [
    # Blue
    _c("Gardens", 3, BLUE, (3, 4), cost="CCW", chain_from="Theater", vp=5),
    _c("Senate", 3, BLUE, (3, 5), cost="WWSO", chain_from="Library", vp=6),
    _c("Town Hall", 3, BLUE, (3, 6), cost="SSSG", vp=6),
    _c("Pantheon", 3, BLUE, (3, 6), cost="CCOGPL", chain_from="Altar", vp=7),
    _c("Palace", 3, BLUE, (3, 7), cost="WSOCGPL", vp=8),
    # Yellow
    _c("Lighthouse", 3, YELLOW, (3, 6), cost="SG", chain_from="Caravansery",
       coins_per=((YELLOW,), 1, SELF), vp_per=((YELLOW,), 1, SELF)),
    _c("Haven", 3, YELLOW, (3, 4), cost="WOL", chain_from="Forum",
       coins_per=((BROWN,), 1, SELF), vp_per=((BROWN,), 1, SELF)),
    _c("Chamber of Commerce", 3, YELLOW, (4, 6), cost="CCP",
       coins_per=((GREY,), 2, SELF), vp_per=((GREY,), 2, SELF)),
    _c("Ludus", 3, YELLOW, (5, 7), cost="SO",
       coins_per=((RED,), 3, SELF), vp_per=((RED,), 1, SELF)),
    _c("Arena", 3, YELLOW, (3, 5), cost="CCO", chain_from="Dispensary",
       coins_per=(STAGE, 3, SELF), vp_per=(STAGE, 1, SELF)),
    # Red
    _c("Arsenal", 3, RED, (3, 5), cost="WWOL", shields=3),
    _c("Siege Workshop", 3, RED, (3, 5), cost="CCCW", chain_from="Laboratory", shields=3),
    _c("Fortifications", 3, RED, (3, 7), cost="OOOC", chain_from="Walls", shields=3),
    _c("Circus", 3, RED, (4, 6), cost="CCCO", chain_from="Training Ground", shields=3),
    _c("Castrum", 3, RED, (4, 7), cost="CCWP", shields=3),
    # Green
    _c("University", 3, GREEN, (3, 4), cost="WWGP", chain_from="Library", science=TABLET),
    _c("Study", 3, GREEN, (3, 5), cost="WPL", chain_from="School", science=GEAR),
    _c("Lodge", 3, GREEN, (3, 6), cost="CCPL", chain_from="Dispensary", science=COMPASS),
    _c("Academy", 3, GREEN, (3, 7), cost="SSSG", chain_from="School", science=COMPASS),
    _c("Observatory", 3, GREEN, (3, 7), cost="OOGL", chain_from="Laboratory", science=GEAR),
]

# Guilds: (players + 2) of these are drawn at random each game.
_GUILDS = [
    _c("Workers Guild", 3, PURPLE, (), cost="OOWSC", vp_per=((BROWN,), 1, NEIGHBORS)),
    _c("Craftsmens Guild", 3, PURPLE, (), cost="OOSS", vp_per=((GREY,), 2, NEIGHBORS)),
    _c("Magistrates Guild", 3, PURPLE, (), cost="WWWSL", vp_per=((BLUE,), 1, NEIGHBORS)),
    _c("Traders Guild", 3, PURPLE, (), cost="GPL", vp_per=((YELLOW,), 1, NEIGHBORS)),
    _c("Spies Guild", 3, PURPLE, (), cost="CCG", vp_per=((RED,), 1, NEIGHBORS)),
    _c("Philosophers Guild", 3, PURPLE, (), cost="CCCPL", vp_per=((GREEN,), 1, NEIGHBORS)),
    _c("Shipowners Guild", 3, PURPLE, (), cost="WWWGP",
       vp_per=((BROWN, GREY, PURPLE), 1, SELF)),
    _c("Scientists Guild", 3, PURPLE, (), cost="WWOOP", science=WILD),
    _c("Decorators Guild", 3, PURPLE, (), cost="OOSL", vp_if_wonder_complete=7),
    _c("Builders Guild", 3, PURPLE, (), cost="SSSCCG", vp_per=(STAGE, 1, ALL3)),
]


def _assign_ids() -> tuple[list[str], dict[str, int]]:
    """Give every distinct card *name* an id. Same-named cards (e.g. Loom I/II) share it."""
    names: list[str] = []
    for card in _AGE1 + _AGE2 + _AGE3 + _GUILDS:
        if card.name not in names:
            names.append(card.name)
    index = {n: i for i, n in enumerate(names)}
    for card in _AGE1 + _AGE2 + _AGE3 + _GUILDS:
        object.__setattr__(card, "id", index[card.name])
    return names, index


CARD_NAMES, CARD_INDEX = _assign_ids()
NUM_CARD_IDS = len(CARD_NAMES)
AGE_CARDS = {1: _AGE1, 2: _AGE2, 3: _AGE3}
GUILDS = _GUILDS
ALL_CARDS = _AGE1 + _AGE2 + _AGE3 + _GUILDS

# One representative Card object per id (for lookups by id)
CARD_BY_ID: list[Card] = [None] * NUM_CARD_IDS  # type: ignore[list-item]
for _card in ALL_CARDS:
    if CARD_BY_ID[_card.id] is None:
        CARD_BY_ID[_card.id] = _card


def build_deck(age: int, num_players: int, rng: random.Random) -> list[Card]:
    """All cards used in `age` for `num_players`, shuffled."""
    deck = [card for card in AGE_CARDS[age] for p in card.players if p <= num_players]
    if age == 3:
        deck += rng.sample(GUILDS, num_players + 2)
    rng.shuffle(deck)
    return deck
