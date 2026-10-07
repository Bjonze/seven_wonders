"""Play many games between baseline bots and print score / win-rate summaries.

    python scripts/simulate.py --games 2000 --bots greedy,random,random,random
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sevenwonders.bots import GreedyBot, RandomBot  # noqa: E402
from sevenwonders.runner import play_game  # noqa: E402

BOTS = {"random": RandomBot, "greedy": GreedyBot}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--bots", default="greedy,greedy,greedy,greedy",
                        help="comma-separated bot names, one per seat")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = args.bots.split(",")
    n = len(names)
    wins = [0.0] * n
    scores = [[] for _ in range(n)]
    categories = collections.defaultdict(list)
    wonder_wins = collections.Counter()
    wonder_games = collections.Counter()

    start = time.time()
    for g in range(args.games):
        seed = args.seed + g
        bots = [BOTS[name](seed=seed * 31 + i) for i, name in enumerate(names)]
        result = play_game(bots, num_players=n, seed=seed)
        for s in range(n):
            scores[s].append(result.scores[s])
            for key, value in result.breakdowns[s].items():
                categories[key].append(value)
            wonder_games[result.wonders[s]] += 1
            if s in result.winners:
                share = 1 / len(result.winners)
                wins[s] += share
                wonder_wins[result.wonders[s]] += share
    elapsed = time.time() - start

    print(f"{args.games} games in {elapsed:.1f}s ({args.games / elapsed:.0f} games/s)\n")
    print("seat  bot      win%   mean score")
    for s in range(n):
        print(f"{s:>4}  {names[s]:<7} {100 * wins[s] / args.games:5.1f}   {statistics.mean(scores[s]):6.1f}")
    print("\nmean points per category (all seats):")
    for key, values in categories.items():
        print(f"  {key:<9} {statistics.mean(values):5.1f}")
    print("\nwin% by wonder side (all seats):")
    for label in sorted(wonder_games, key=lambda w: -wonder_wins[w] / wonder_games[w]):
        print(f"  {label:<22} {100 * wonder_wins[label] / wonder_games[label]:5.1f}%  ({wonder_games[label]} games)")


if __name__ == "__main__":
    main()
