"""Wonder boards for the 2020 (2nd) edition base game, Day and Night sides."""

from __future__ import annotations

from dataclasses import dataclass

from .resources import CLAY, CLOTH, GLASS, ORE, PAPYRUS, STONE, WOOD, res, res_options

# Special stage effects
BROWN_CHOICE = "brown_choice"  # produce one brown resource per turn (not tradeable)
GREY_CHOICE = "grey_choice"  # produce one grey resource per turn (not tradeable)
SCIENCE_WILD = "science_wild"  # one science symbol of your choice at game end
PLAY_LAST_CARD = "play_last_card"  # Babylon night: play the 7th card of each Age
DISCARD_BUILD = "discard_build"  # Halikarnassos: build one discarded card for free
FREE_FIRST_COLOR = "free_first_color"  # Olympia day: first card of each colour is free
FREE_FIRST_OF_AGE = "free_first_of_age"  # Olympia night: first card of each Age is free
FREE_LAST_OF_AGE = "free_last_of_age"  # Olympia night: last card of each Age is free


@dataclass(frozen=True)
class Stage:
    cost: tuple[int, ...]
    vp: int = 0
    coins: int = 0
    shields: int = 0
    effect: str | None = None


@dataclass(frozen=True)
class WonderSide:
    name: str
    side: str  # "day" or "night"
    resource: int
    stages: tuple[Stage, ...]
    id: int = -1

    @property
    def label(self) -> str:
        return f"{self.name} ({self.side})"


def _s(cost: str, **kw) -> Stage:
    return Stage(cost=res(cost), **kw)


_WONDERS = [
    WonderSide("Alexandria", "day", GLASS, (
        _s("SS", vp=3), _s("OO", effect=BROWN_CHOICE), _s("PL", vp=7))),
    WonderSide("Alexandria", "night", GLASS, (
        _s("CC", effect=BROWN_CHOICE), _s("OOO", effect=GREY_CHOICE), _s("WWWW", vp=7))),
    WonderSide("Babylon", "day", WOOD, (
        _s("CC", vp=3), _s("OOL", effect=SCIENCE_WILD), _s("WWWW", vp=7))),
    WonderSide("Babylon", "night", WOOD, (
        _s("SS", effect=PLAY_LAST_CARD), _s("CCCG", effect=SCIENCE_WILD))),
    WonderSide("Ephesos", "day", PAPYRUS, (
        _s("CC", vp=3), _s("WW", coins=9), _s("OOG", vp=7))),
    WonderSide("Ephesos", "night", PAPYRUS, (
        _s("SS", vp=2, coins=4), _s("WW", vp=3, coins=4), _s("OOL", vp=5, coins=4))),
    WonderSide("Gizah", "day", STONE, (
        _s("WW", vp=3), _s("CCL", vp=5), _s("SSSS", vp=7))),
    WonderSide("Gizah", "night", STONE, (
        _s("WW", vp=3), _s("SSS", vp=5), _s("CCC", vp=5), _s("SSSSP", vp=7))),
    WonderSide("Halikarnassos", "day", CLOTH, (
        _s("OO", vp=3), _s("GP", effect=DISCARD_BUILD), _s("SSS", vp=7))),
    WonderSide("Halikarnassos", "night", CLOTH, (
        _s("CC", vp=2, effect=DISCARD_BUILD), _s("GP", vp=1, effect=DISCARD_BUILD),
        _s("WWW", effect=DISCARD_BUILD))),
    WonderSide("Olympia", "day", CLAY, (
        _s("SS", vp=3), _s("WW", effect=FREE_FIRST_COLOR), _s("CCC", vp=7))),
    WonderSide("Olympia", "night", CLAY, (
        _s("OO", vp=2, effect=FREE_FIRST_OF_AGE), _s("CCC", vp=3, effect=FREE_LAST_OF_AGE),
        _s("GPL", vp=5))),
    WonderSide("Rhodos", "day", ORE, (
        _s("WW", vp=3), _s("CCC", shields=2), _s("OOOO", vp=7))),
    WonderSide("Rhodos", "night", ORE, (
        _s("SSS", vp=3, coins=3, shields=1), _s("OOOO", vp=4, coins=4, shields=1))),
]

WONDERS: list[WonderSide] = [
    WonderSide(w.name, w.side, w.resource, w.stages, id=i) for i, w in enumerate(_WONDERS)
]
WONDER_NAMES = sorted({w.name for w in WONDERS})
NUM_WONDER_SIDES = len(WONDERS)
MAX_STAGES = max(len(w.stages) for w in WONDERS)

BROWN_OPTIONS = res_options("WSCO")
GREY_OPTIONS = res_options("GLP")


def wonder(name: str, side: str) -> WonderSide:
    for w in WONDERS:
        if w.name == name and w.side == side:
            return w
    raise KeyError(f"{name} ({side})")


__all__ = [
    "WONDERS", "WONDER_NAMES", "WonderSide", "Stage", "wonder", "MAX_STAGES",
    "NUM_WONDER_SIDES", "BROWN_OPTIONS", "GREY_OPTIONS",
    "BROWN_CHOICE", "GREY_CHOICE", "SCIENCE_WILD", "PLAY_LAST_CARD", "DISCARD_BUILD",
    "FREE_FIRST_COLOR", "FREE_FIRST_OF_AGE", "FREE_LAST_OF_AGE",
]
