"""Resource constants and the payment solver.

Resources are indexed 0..6. A cost or a production amount is a 7-tuple of counts.
"""

from __future__ import annotations

WOOD, STONE, CLAY, ORE, GLASS, CLOTH, PAPYRUS = range(7)
NUM_RESOURCES = 7
RESOURCE_NAMES = ("wood", "stone", "clay", "ore", "glass", "cloth", "papyrus")
BROWN_RESOURCES = (WOOD, STONE, CLAY, ORE)
GREY_RESOURCES = (GLASS, CLOTH, PAPYRUS)
IS_BROWN = (True, True, True, True, False, False, False)

# Letters used in the card data: W S C O G L(oom) P
_LETTERS = {"W": WOOD, "S": STONE, "C": CLAY, "O": ORE, "G": GLASS, "L": CLOTH, "P": PAPYRUS}

ZERO = (0,) * NUM_RESOURCES


def res(spec: str) -> tuple[int, ...]:
    """Parse a resource string like "WWSG" into a 7-tuple of counts."""
    counts = [0] * NUM_RESOURCES
    for ch in spec:
        counts[_LETTERS[ch]] += 1
    return tuple(counts)


def res_options(spec: str) -> tuple[int, ...]:
    """Parse a choice like "WS" (wood OR stone) into a tuple of resource indices."""
    return tuple(_LETTERS[ch] for ch in spec)


def min_purchase_cost(
    cost: tuple[int, ...],
    own_fixed: list[int],
    own_choices: list[tuple[int, ...]],
    left_fixed: list[int],
    left_choices: list[tuple[int, ...]],
    right_fixed: list[int],
    right_choices: list[tuple[int, ...]],
    left_price: tuple[int, ...],
    right_price: tuple[int, ...],
) -> tuple[int, int] | None:
    """Cheapest way to cover `cost` with own production plus purchases from neighbours.

    Each production source covers one unit per turn. Fixed production is single-resource,
    choice sources produce one of several resources. Buying from a neighbour costs
    left_price[r] / right_price[r] coins per unit.

    Returns (coins paid to left neighbour, coins paid to right neighbour) for the cheapest
    option, or None if the cost cannot be covered. Coin costs printed on cards are not
    included here.
    """
    need = [max(0, c - f) for c, f in zip(cost, own_fixed)]
    if not any(need):
        return (0, 0)

    # Choice units: (options, side) where side 0 = own (free), 1 = left, 2 = right.
    units: list[tuple[tuple[int, ...], int]] = []
    for opts in own_choices:
        units.append((opts, 0))
    for opts in left_choices:
        units.append((opts, 1))
    for opts in right_choices:
        units.append((opts, 2))

    best: list = [None, 10**9]  # [(left, right), total]

    def finish(need_: list[int], paid_l: int, paid_r: int) -> None:
        # Cover the rest with neighbours' fixed production, cheapest side first.
        pl, pr = paid_l, paid_r
        for r in range(NUM_RESOURCES):
            n = need_[r]
            if n == 0:
                continue
            lp, rp = left_price[r], right_price[r]
            if lp <= rp:
                take = min(n, left_fixed[r])
                pl += take * lp
                n -= take
                take = min(n, right_fixed[r])
                pr += take * rp
                n -= take
            else:
                take = min(n, right_fixed[r])
                pr += take * rp
                n -= take
                take = min(n, left_fixed[r])
                pl += take * lp
                n -= take
            if n > 0:
                return
            if pl + pr >= best[1]:
                return
        if pl + pr < best[1]:
            best[0] = (pl, pr)
            best[1] = pl + pr

    def search(i: int, need_: list[int], paid_l: int, paid_r: int) -> None:
        if paid_l + paid_r >= best[1]:
            return
        if not any(need_):
            best[0] = (paid_l, paid_r)
            best[1] = paid_l + paid_r
            return
        if i == len(units):
            finish(need_, paid_l, paid_r)
            return
        opts, side = units[i]
        useful = [r for r in opts if need_[r] > 0]
        for r in useful:
            need_[r] -= 1
            if side == 0:
                search(i + 1, need_, paid_l, paid_r)
            elif side == 1:
                search(i + 1, need_, paid_l + left_price[r], paid_r)
            else:
                search(i + 1, need_, paid_l, paid_r + right_price[r])
            need_[r] += 1
        # Skipping an own (free) unit is never better than using it on something useful.
        if side != 0 or not useful:
            search(i + 1, need_, paid_l, paid_r)

    search(0, need, 0, 0)
    return best[0]
