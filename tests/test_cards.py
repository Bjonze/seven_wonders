import random

import pytest

from sevenwonders.cards import AGE_CARDS, ALL_CARDS, CARD_INDEX, GUILDS, build_deck
from sevenwonders.wonders import WONDERS


@pytest.mark.parametrize("players", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("age", [1, 2, 3])
def test_deck_has_seven_cards_per_player(age, players):
    deck = build_deck(age, players, random.Random(0))
    assert len(deck) == 7 * players


def test_box_totals_match_rulebook():
    # 148 Age cards: 49 / 49 / 50 (Age III includes the 10 guilds)
    assert sum(len(c.players) for c in AGE_CARDS[1]) == 49
    assert sum(len(c.players) for c in AGE_CARDS[2]) == 49
    assert sum(len(c.players) for c in AGE_CARDS[3]) + len(GUILDS) == 50
    assert len(GUILDS) == 10


def test_chains_point_to_earlier_cards():
    for card in ALL_CARDS:
        for name in card.chain_from:
            assert name in CARD_INDEX, f"{card.name} chains from unknown {name}"
            sources = [c for c in ALL_CARDS if c.name == name]
            assert all(s.age < card.age for s in sources)


def test_no_duplicate_names_within_an_age():
    for age, cards in AGE_CARDS.items():
        names = [c.name for c in cards]
        assert len(names) == len(set(names)), age


def test_wonders_have_both_sides():
    names = {w.name for w in WONDERS}
    assert len(names) == 7
    for name in names:
        assert {w.side for w in WONDERS if w.name == name} == {"day", "night"}
