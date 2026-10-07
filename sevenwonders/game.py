"""7 Wonders (2nd edition) rules engine.

Seats are arranged in a circle. The left neighbour of seat s is (s - 1) % n and the right
neighbour is (s + 1) % n. In Ages I and III hands pass left, in Age II they pass right.

Flow: during the PLAY phase every seat chooses an action simultaneously and `step` resolves
them together. Two wonder powers add single-player phases:
  * BABYLON: Babylon (night) plays the 7th card of an Age after everyone's 6th turn.
  * HALIKARNASSOS: Halikarnassos picks a discarded card to build for free at the end of the
    turn in which it built a stage with that power.

Simplifications (documented in DATA_NOTES.md): payments always use the cheapest legal
combination of trades, science wildcards are assigned optimally at game end, and
Halikarnassos must build a discarded card if any is legal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .cards import (
    ALL3, BLUE, NEIGHBORS, NUM_COLORS, PURPLE, SELF, STAGE, WILD, YELLOW,
    Card, build_deck,
)
from .resources import IS_BROWN, NUM_RESOURCES, min_purchase_cost
from .wonders import (
    BROWN_CHOICE, BROWN_OPTIONS, DISCARD_BUILD, FREE_FIRST_COLOR, FREE_FIRST_OF_AGE,
    FREE_LAST_OF_AGE, GREY_CHOICE, GREY_OPTIONS, PLAY_LAST_CARD, SCIENCE_WILD, WONDERS,
    Stage, WonderSide,
)

BUILD, WONDER, SELL = 0, 1, 2
NUM_ACTION_KINDS = 3
ACTION_KIND_NAMES = ("build", "wonder", "sell")

# How to pay for resources bought from neighbours (build and wonder actions)
CHEAPEST = 0  # lowest total; when both neighbours sell at the same price, buy from the left
PAY_LEFT = 1  # pay the right neighbour as little as possible (even if it costs more)
PAY_RIGHT = 2  # pay the left neighbour as little as possible (even if it costs more)
PAY_NAMES = ("cheapest", "pay-left", "pay-right")
_AVOID = 1000
_PAY_WEIGHTS = {CHEAPEST: (1, 1), PAY_LEFT: (1, 1 + _AVOID), PAY_RIGHT: (1 + _AVOID, 1)}

PLAY, BABYLON, HALIKARNASSOS, OVER = "play", "babylon", "halikarnassos", "over"

MILITARY_WIN = {1: 1, 2: 3, 3: 5}
MILITARY_LOSS = -1
STARTING_COINS = 3
SELL_COINS = 3
CARDS_PER_HAND = 7
TURNS_PER_AGE = 6


@dataclass(frozen=True)
class Action:
    kind: int
    card: Card
    pay: int = CHEAPEST

    @property
    def index(self) -> int:
        """Flat action index ignoring the payment option: card id * 3 + kind."""
        return self.card.id * NUM_ACTION_KINDS + self.kind

    def __repr__(self) -> str:
        suffix = f"({PAY_NAMES[self.pay]})" if self.pay != CHEAPEST else ""
        return f"{ACTION_KIND_NAMES[self.kind]}:{self.card.name}{suffix}"


def science_points(counts: list[int] | tuple[int, ...], wild: int = 0) -> int:
    """Science score: sum of squares + 7 per complete set; wildcards placed optimally."""
    if wild == 0:
        a, b, c = counts
        return a * a + b * b + c * c + 7 * min(a, b, c)
    best = 0
    for i in range(3):
        bumped = list(counts)
        bumped[i] += 1
        best = max(best, science_points(bumped, wild - 1))
    return best


class Player:
    def __init__(self, seat: int, wonder: WonderSide):
        self.seat = seat
        self.wonder = wonder
        self.stages_built = 0
        self.stage_cards: list[Card] = []
        self.city: list[Card] = []
        self.names: set[str] = set()
        self.coins = STARTING_COINS
        self.military: list[int] = []
        self.color_counts = [0] * NUM_COLORS
        self.shields = 0
        self.science = [0, 0, 0]
        self.science_wild = 0
        # Production. "fixed" includes everything single-valued; "tradeable" is the subset
        # neighbours may buy (brown/grey cards and the wonder's starting resource).
        self.fixed = [0] * NUM_RESOURCES
        self.fixed[wonder.resource] += 1
        self.tradeable_fixed = list(self.fixed)
        self.choices: list[tuple[int, ...]] = []
        self.tradeable_choices: list[tuple[int, ...]] = []
        self.discount_left_brown = False
        self.discount_right_brown = False
        self.discount_grey = False
        self.free_first_color = False
        self.free_first_of_age = False
        self.free_last_of_age = False
        self.play_last_card = False
        self.price_left = self.price_right = (2,) * NUM_RESOURCES
        self.trade_paid = 0  # coins paid to neighbours for resources (statistics only)
        self.trade_received = 0

    # -- state changes -------------------------------------------------------------------
    def add_card(self, card: Card) -> None:
        self.city.append(card)
        self.names.add(card.name)
        self.color_counts[card.color] += 1
        self.shields += card.shields
        if card.science == WILD:
            self.science_wild += 1
        elif card.science >= 0:
            self.science[card.science] += 1
        if any(card.produces):
            for r in range(NUM_RESOURCES):
                self.fixed[r] += card.produces[r]
                if card.tradeable:
                    self.tradeable_fixed[r] += card.produces[r]
        if card.choice:
            self.choices.append(card.choice)
            if card.tradeable:
                self.tradeable_choices.append(card.choice)
        if card.discount_left_brown or card.discount_right_brown or card.discount_grey:
            self.discount_left_brown |= card.discount_left_brown
            self.discount_right_brown |= card.discount_right_brown
            self.discount_grey |= card.discount_grey
            self.price_left = self._prices(self.discount_left_brown)
            self.price_right = self._prices(self.discount_right_brown)

    def add_stage(self, card: Card) -> Stage:
        stage = self.wonder.stages[self.stages_built]
        self.stages_built += 1
        self.stage_cards.append(card)
        self.shields += stage.shields
        if stage.effect == BROWN_CHOICE:
            self.choices.append(BROWN_OPTIONS)
        elif stage.effect == GREY_CHOICE:
            self.choices.append(GREY_OPTIONS)
        elif stage.effect == SCIENCE_WILD:
            self.science_wild += 1
        elif stage.effect == PLAY_LAST_CARD:
            self.play_last_card = True
        elif stage.effect == FREE_FIRST_COLOR:
            self.free_first_color = True
        elif stage.effect == FREE_FIRST_OF_AGE:
            self.free_first_of_age = True
        elif stage.effect == FREE_LAST_OF_AGE:
            self.free_last_of_age = True
        return stage

    @property
    def wonder_complete(self) -> bool:
        return self.stages_built == len(self.wonder.stages)

    def next_stage(self) -> Stage | None:
        if self.wonder_complete:
            return None
        return self.wonder.stages[self.stages_built]

    def _prices(self, brown_discount: bool) -> tuple[int, ...]:
        """Coins per unit when buying from a neighbour."""
        return tuple(
            1 if (brown_discount if IS_BROWN[r] else self.discount_grey) else 2
            for r in range(NUM_RESOURCES)
        )

    def clone(self) -> Player:
        p = Player.__new__(Player)
        p.__dict__ = self.__dict__.copy()
        for name in ("stage_cards", "city", "military", "color_counts", "science", "fixed",
                     "tradeable_fixed", "choices", "tradeable_choices"):
            setattr(p, name, list(getattr(self, name)))
        p.names = set(self.names)
        return p


class Game:
    def __init__(
        self,
        num_players: int = 4,
        seed: int | None = None,
        wonders: list[WonderSide] | None = None,
    ):
        if not 3 <= num_players <= 7:
            raise ValueError("2nd edition base game supports 3-7 players")
        self.n = num_players
        self.rng = random.Random(seed)
        if wonders is None:
            names = self.rng.sample(sorted({w.name for w in WONDERS}), num_players)
            sides = [self.rng.choice(("day", "night")) for _ in names]
            wonders = [
                next(w for w in WONDERS if w.name == name and w.side == side)
                for name, side in zip(names, sides)
            ]
        if len(wonders) != num_players:
            raise ValueError("need one wonder per player")
        self.players = [Player(i, w) for i, w in enumerate(wonders)]
        self.discard: list[Card] = []
        # For statistics: (seat, "sold" | "leftover", age, turn) of each discard pile card,
        # and (card, seat, how, age, turn) of the latest Halikarnassos pick
        self.discard_origin: list[tuple[int, str, int, int]] = []
        self.last_free_build: tuple | None = None
        self.hands: list[list[Card]] = [[] for _ in range(num_players)]
        self.age = 0
        self.turn = 0
        self.phase = PLAY
        self.active: list[int] = []
        self._babylon_queue: list[int] = []
        self._hali_queue: list[int] = []
        self._leftovers_done = False
        self._cache: dict = {}
        self._start_age(1)

    def clone(self) -> Game:
        """Independent copy of the game (cards and wonders are shared, they never change).

        The random generator is copied too, so later deals are the same as in the original."""
        g = Game.__new__(Game)
        g.__dict__ = self.__dict__.copy()
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        g.players = [p.clone() for p in self.players]
        g.discard = list(self.discard)
        g.discard_origin = list(self.discard_origin)
        g.hands = [list(h) for h in self.hands]
        g.active = list(self.active)
        g._babylon_queue = list(self._babylon_queue)
        g._hali_queue = list(self._hali_queue)
        g._cache = dict(self._cache)
        return g

    # -- neighbours ----------------------------------------------------------------------
    def left(self, seat: int) -> int:
        return (seat - 1) % self.n

    def right(self, seat: int) -> int:
        return (seat + 1) % self.n

    @property
    def over(self) -> bool:
        return self.phase == OVER

    # -- payments ------------------------------------------------------------------------
    def _trade_cost(self, seat: int, cost: tuple[int, ...], pay: int = CHEAPEST) -> tuple[int, int] | None:
        p = self.players[seat]
        lp = self.players[self.left(seat)]
        rp = self.players[self.right(seat)]
        return min_purchase_cost(
            cost, p.fixed, p.choices,
            lp.tradeable_fixed, lp.tradeable_choices,
            rp.tradeable_fixed, rp.tradeable_choices,
            p.price_left, p.price_right, *_PAY_WEIGHTS[pay],
        )

    def is_free(self, seat: int, card: Card) -> bool:
        """Chain or Olympia power makes the card free to build."""
        p = self.players[seat]
        for name in card.chain_from:
            if name in p.names:
                return True
        if p.free_first_color and p.color_counts[card.color] == 0:
            return True
        if self.phase == PLAY:
            if p.free_first_of_age and self.turn == 1:
                return True
            if p.free_last_of_age and self.turn == TURNS_PER_AGE:
                return True
        return False

    def build_payment(self, seat: int, card: Card, pay: int = CHEAPEST) -> tuple[int, int, int] | None:
        """(coins to bank, coins to left, coins to right) to build `card`, or None."""
        key = ("b", seat, card.id, pay)
        if key in self._cache:
            return self._cache[key]
        result = self._build_payment(seat, card, pay)
        self._cache[key] = result
        return result

    def _build_payment(self, seat: int, card: Card, pay: int) -> tuple[int, int, int] | None:
        p = self.players[seat]
        if card.name in p.names:
            return None
        if self.is_free(seat, card):
            return (0, 0, 0)
        if card.coin_cost > p.coins:
            return None
        trade = self._trade_cost(seat, card.cost, pay)
        if trade is None or card.coin_cost + trade[0] + trade[1] > p.coins:
            return None
        return (card.coin_cost, trade[0], trade[1])

    def wonder_payment(self, seat: int, pay: int = CHEAPEST) -> tuple[int, int, int] | None:
        key = ("w", seat, pay)
        if key in self._cache:
            return self._cache[key]
        p = self.players[seat]
        stage = p.next_stage()
        result = None
        if stage is not None:
            trade = self._trade_cost(seat, stage.cost, pay)
            if trade is not None and trade[0] + trade[1] <= p.coins:
                result = (0, trade[0], trade[1])
        self._cache[key] = result
        return result

    def payment(self, seat: int, action: Action) -> tuple[int, int, int] | None:
        """(coins to bank, coins to left, coins to right) for `action`, or None if illegal."""
        if action.kind == BUILD:
            return self.build_payment(seat, action.card, action.pay)
        if action.kind == WONDER:
            return self.wonder_payment(seat, action.pay)
        return (0, 0, 0)

    # -- actions -------------------------------------------------------------------------
    def legal_actions(self, seat: int, payment_choice: bool = False) -> list[Action]:
        """Legal actions for `seat`. With payment_choice, build and wonder actions that buy
        resources also come in "pay-left" / "pay-right" variants when those give a
        different (affordable) payment than the cheapest one."""
        if seat not in self.active:
            return []
        if self.phase == HALIKARNASSOS:
            return self._hali_options(seat)
        actions = []
        wonder_pay = self.wonder_payment(seat)
        seen = set()
        for card in self.hands[seat]:
            if card.name in seen:
                continue
            seen.add(card.name)
            build_pay = self.build_payment(seat, card)
            if build_pay is not None:
                actions.append(Action(BUILD, card))
                if payment_choice:
                    actions += self._payment_variants(seat, Action(BUILD, card), build_pay)
            if wonder_pay is not None:
                actions.append(Action(WONDER, card))
                if payment_choice:
                    actions += self._payment_variants(seat, Action(WONDER, card), wonder_pay)
            actions.append(Action(SELL, card))
        return actions

    def _payment_variants(self, seat: int, base: Action, cheapest: tuple[int, int, int]) -> list[Action]:
        if cheapest[1] == 0 and cheapest[2] == 0:
            return []  # nothing bought, nothing to choose
        variants, seen = [], {cheapest}
        for pay in (PAY_LEFT, PAY_RIGHT):
            action = Action(base.kind, base.card, pay)
            result = self.payment(seat, action)
            if result is not None and result not in seen:
                seen.add(result)
                variants.append(action)
        return variants

    def _hali_options(self, seat: int) -> list[Action]:
        names = self.players[seat].names
        seen = set()
        actions = []
        for card in self.discard:
            if card.name in names or card.name in seen:
                continue
            seen.add(card.name)
            actions.append(Action(BUILD, card))
        return actions

    def step(self, actions: dict[int, Action]) -> None:
        """Apply one action for every active seat."""
        if self.phase == OVER:
            raise RuntimeError("game is over")
        missing = [s for s in self.active if s not in actions]
        if missing:
            raise ValueError(f"missing actions for seats {missing}")
        if self.phase == PLAY:
            self._resolve_plays({s: actions[s] for s in self.active})
            self._after_play()
        elif self.phase == BABYLON:
            seat = self.active[0]
            self._resolve_plays({seat: actions[seat]})
            for card in self.hands[seat]:  # nothing left in practice
                self._discard(card, seat, "leftover")
            self.hands[seat] = []
            self._babylon_queue.pop(0)
            self._continue()
        elif self.phase == HALIKARNASSOS:
            seat = self.active[0]
            action = actions[seat]
            if action.kind != BUILD or action.card.name in self.players[seat].names:
                raise ValueError(f"illegal Halikarnassos action {action}")
            i = self.discard.index(action.card)
            del self.discard[i]
            self.last_free_build = (action.card, *self.discard_origin.pop(i))
            self.players[seat].add_card(action.card)
            self._card_income(seat, action.card)
            self._hali_queue.pop(0)
            self._cache.clear()
            self._continue()

    def _resolve_plays(self, actions: dict[int, Action]) -> None:
        payments = {}
        for seat, action in actions.items():
            if action.card not in self.hands[seat]:
                raise ValueError(f"seat {seat} does not hold {action.card}")
            pay = self.payment(seat, action)
            if pay is None:
                raise ValueError(f"illegal action {action} for seat {seat}")
            payments[seat] = pay

        transfers = [0] * self.n
        built: list[tuple[int, Card]] = []
        stages: list[tuple[int, Stage]] = []
        for seat, action in actions.items():
            p = self.players[seat]
            self.hands[seat].remove(action.card)
            bank, to_left, to_right = payments[seat]
            p.coins -= bank + to_left + to_right
            p.trade_paid += to_left + to_right
            transfers[self.left(seat)] += to_left
            transfers[self.right(seat)] += to_right
            if action.kind == BUILD:
                p.add_card(action.card)
                built.append((seat, action.card))
            elif action.kind == WONDER:
                stages.append((seat, p.add_stage(action.card)))
            else:
                self._discard(action.card, seat, "sold")
                p.coins += SELL_COINS
        for seat in range(self.n):
            self.players[seat].coins += transfers[seat]
            self.players[seat].trade_received += transfers[seat]
        # Immediate effects happen after every card of the turn is in place.
        for seat, card in built:
            self._card_income(seat, card)
        for seat, stage in stages:
            self.players[seat].coins += stage.coins
            if stage.effect == DISCARD_BUILD:
                self._hali_queue.append(seat)
        self._cache.clear()

    def _discard(self, card: Card, seat: int, how: str) -> None:
        self.discard.append(card)
        self.discard_origin.append((seat, how, self.age, self.turn))

    def _count(self, seat: int, targets, scope: str) -> int:
        seats = {SELF: (seat,), NEIGHBORS: (self.left(seat), self.right(seat)),
                 ALL3: (seat, self.left(seat), self.right(seat))}[scope]
        total = 0
        for s in seats:
            p = self.players[s]
            if targets == STAGE:
                total += p.stages_built
            else:
                total += sum(p.color_counts[c] for c in targets)
        return total

    def _card_income(self, seat: int, card: Card) -> None:
        p = self.players[seat]
        p.coins += card.coins
        if card.coins_per is not None:
            targets, amount, scope = card.coins_per
            p.coins += amount * self._count(seat, targets, scope)

    # -- turn / age flow -----------------------------------------------------------------
    def _start_age(self, age: int) -> None:
        self.age = age
        self.turn = 1
        deck = build_deck(age, self.n, self.rng)
        self.hands = [deck[i * CARDS_PER_HAND:(i + 1) * CARDS_PER_HAND] for i in range(self.n)]
        self._leftovers_done = False
        self.phase = PLAY
        self.active = list(range(self.n))
        self._cache.clear()

    def _pass_hands(self) -> None:
        if self.age == 2:  # pass right: seat s receives from its left neighbour
            self.hands = [self.hands[self.left(s)] for s in range(self.n)]
        else:  # pass left: seat s receives from its right neighbour
            self.hands = [self.hands[self.right(s)] for s in range(self.n)]

    def _after_play(self) -> None:
        if self.turn == TURNS_PER_AGE:
            self._babylon_queue = [
                s for s in range(self.n) if self.players[s].play_last_card and self.hands[s]
            ]
        self._continue()

    def _continue(self) -> None:
        if self.turn == TURNS_PER_AGE and self._babylon_queue:
            self.phase = BABYLON
            self.active = [self._babylon_queue[0]]
            self._cache.clear()
            return
        if self.turn == TURNS_PER_AGE and not self._leftovers_done:
            for s in range(self.n):
                for card in self.hands[s]:
                    self._discard(card, s, "leftover")
                self.hands[s] = []
            self._leftovers_done = True
        while self._hali_queue:
            seat = self._hali_queue[0]
            if self._hali_options(seat):
                self.phase = HALIKARNASSOS
                self.active = [seat]
                self._cache.clear()
                return
            self._hali_queue.pop(0)
        if self.turn == TURNS_PER_AGE:
            self._resolve_military()
            if self.age == 3:
                self.phase = OVER
                self.active = []
                return
            self._start_age(self.age + 1)
        else:
            self._pass_hands()
            self.turn += 1
            self.phase = PLAY
            self.active = list(range(self.n))
            self._cache.clear()

    def _resolve_military(self) -> None:
        win = MILITARY_WIN[self.age]
        for seat, p in enumerate(self.players):
            for nb in (self.left(seat), self.right(seat)):
                other = self.players[nb].shields
                if p.shields > other:
                    p.military.append(win)
                elif p.shields < other:
                    p.military.append(MILITARY_LOSS)

    # -- scoring -------------------------------------------------------------------------
    def score_breakdown(self, seat: int) -> dict[str, int]:
        p = self.players[seat]
        wonder_vp = sum(st.vp for st in p.wonder.stages[:p.stages_built])
        civil = commerce = guilds = 0
        for card in p.city:
            if card.color == BLUE:
                civil += card.vp
            vp = 0
            if card.vp_per is not None:
                targets, amount, scope = card.vp_per
                vp = amount * self._count(seat, targets, scope)
            if card.vp_if_wonder_complete and p.wonder_complete:
                vp += card.vp_if_wonder_complete
            if card.color == YELLOW:
                commerce += vp
            elif card.color == PURPLE:
                guilds += vp
        breakdown = {
            "military": sum(p.military),
            "treasure": p.coins // 3,
            "wonder": wonder_vp,
            "civil": civil,
            "commerce": commerce,
            "guilds": guilds,
            "science": science_points(p.science, p.science_wild),
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def scores(self) -> list[int]:
        return [self.score_breakdown(s)["total"] for s in range(self.n)]

    def winners(self) -> list[int]:
        """Seats with the best score; ties broken by coins, remaining ties are shared."""
        keys = [(self.score_breakdown(s)["total"], self.players[s].coins) for s in range(self.n)]
        best = max(keys)
        return [s for s in range(self.n) if keys[s] == best]

    def ranks(self) -> list[float]:
        """1-based rank per seat (ties share the average rank), using the coin tie-break."""
        keys = [(self.score_breakdown(s)["total"], self.players[s].coins) for s in range(self.n)]
        ranks = []
        for s in range(self.n):
            better = sum(1 for k in keys if k > keys[s])
            equal = sum(1 for k in keys if k == keys[s])
            ranks.append(better + (equal + 1) / 2)
        return ranks

    # -- misc ----------------------------------------------------------------------------
    def total_cards(self) -> int:
        """Cards currently anywhere in the game (for conservation checks)."""
        in_play = sum(len(p.city) + len(p.stage_cards) for p in self.players)
        return in_play + len(self.discard) + sum(len(h) for h in self.hands)
