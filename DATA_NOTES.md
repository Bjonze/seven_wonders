# Card data and rules notes

## Sources
- Card tables, chains, guilds and wonder boards: the 7 Wonders fandom wiki
  (`List of Age Cards/Overview`, `Chains`, `List of Cards`, and the individual wonder pages).
  It describes the 2nd edition (Well, Ludus, Castrum, Decorators Guild, new wonder costs).
- 1st → 2nd edition change list: assortedmeeples.com, "7 Wonders 1st vs 2nd Edition".
- Raw wiki dumps are kept in `research/` (not needed to run anything).

## Checks the data passes (`tests/test_cards.py`)
- Every Age has exactly 7 cards per player for 3–7 players.
- 148 Age cards in total: 49 / 49 / 50 (Age III includes the 10 guilds).
- Every chain points to a card from an earlier Age.

## Entries where sources disagreed
| Card | Used | Note |
|---|---|---|
| Craftsmens Guild | 2 ore + 2 stone | The wiki lists stone, stone, ore. The 1st edition card is 2 ore + 2 stone, and the 2nd edition change list (which mentions the Builders and Spies Guild cost changes) doesn't mention this card. |
| Ludus | stone + ore; 3 coins and 1 VP per red card | Only one source found. |

## Rule interpretations and simplifications
- **Payment** always uses the cheapest legal combination of trades. Players can't choose to
  pay a specific neighbour more.
- **Trading**: you can buy from a neighbour's brown/grey cards and starting resource, never
  from yellow cards (Caravansery, Forum) or wonder stages (Alexandria). Each source sells one
  unit per turn. Prices: 2 coins, or 1 with East/West Trading Post (brown, one side) or
  Marketplace (grey, both sides).
- **Turn resolution**: all actions resolve together using start-of-turn coins. Then trade
  income is paid, then immediate income (Tavern, Vineyard, Bazaar, Age III yellows, Ephesos,
  Rhodos). So Vineyard/Bazaar also count cards that neighbours built this turn.
- **Halikarnassos** builds from the discard at the end of the turn it built that stage. On an
  Age's last turn this happens after Babylon's 7th card and after the leftover cards are
  discarded, so it can pick those cards. If any discarded card is legal, it must build one.
- **Babylon (night)** can play the 7th card even if it built the stage on the 6th turn (the
  official app does the same).
- **Olympia (night)** free first/last card only applies when constructing a building, not a
  wonder stage or a sale.
- **Science wildcards** (Babylon, Scientists Guild) are assigned to maximise score.
- **Ties**: most coins wins. If still tied, the win is shared.
