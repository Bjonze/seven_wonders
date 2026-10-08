"""Evaluate single models and ensembles against fixed opponents on the same deals.

    python scripts/ensemble_eval.py --configs v5a v4,v5a,v5b v2,v3,v4,v5a,v5b --opponents v5a v4

Each config is a comma-separated list of checkpoints/<name>/latest.pt models played as one
EnsembleBot (a single name is just that model). The config plays one seat (rotating) against
three copies of each opponent, in normal 4-player games where players pick wonder sides.
All configs play the same deals, so differences between configs are paired and the
confidence interval of a difference is narrower than that of each win rate.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MODELS = {}


def _init(names: list[str], low_priority: bool) -> None:
    import torch

    from sevenwonders.rl.model import load_model
    from sevenwonders.rl.priority import lower_priority
    torch.set_num_threads(1)
    if low_priority:
        lower_priority("idle")
    for name in names:
        _MODELS[name] = load_model(str(ROOT / "checkpoints" / name / "latest.pt"))


def _task(args):
    config, opponent, ids, seed = args
    from sevenwonders.rl.agent import EnsembleBot, PolicyBot
    from sevenwonders.runner import play_game
    members = [_MODELS[m] for m in config.split(",")]
    wins, margins = [], []
    for g in ids:
        seat = g % 4
        bots = [PolicyBot(_MODELS[opponent], seed=seed + g * 4 + i) for i in range(4)]
        bots[seat] = EnsembleBot(members, seed=seed + g)
        result = play_game(bots, num_players=4, seed=seed + g, choose_sides=True)
        wins.append(1.0 / len(result.winners) if seat in result.winners else 0.0)
        margins.append(result.scores[seat] - max(s for i, s in enumerate(result.scores) if i != seat))
    return config, opponent, np.asarray(ids), np.asarray(wins), np.asarray(margins)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--opponents", nargs="+", default=["v5a"])
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--seed", type=int, default=600_000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--low-priority", action="store_true")
    args = p.parse_args()

    names = sorted({m for c in args.configs for m in c.split(",")} | set(args.opponents))
    chunks = [list(range(i, args.games, args.workers * 2)) for i in range(args.workers * 2)]
    tasks = [(c, o, ids, args.seed) for o in args.opponents for c in args.configs for ids in chunks]
    wins = {(c, o): np.zeros(args.games) for c in args.configs for o in args.opponents}
    margins = {k: np.zeros(args.games) for k in wins}
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(names, args.low_priority)) as pool:
        for config, opponent, ids, w, m in pool.imap_unordered(_task, tasks):
            wins[(config, opponent)][ids] = w
            margins[(config, opponent)][ids] = m

    base = args.configs[0]
    print(f"{args.games} deals per matchup, 4 players, sides chosen; 25% = equal strength.")
    print(f"'vs first' = paired difference to '{base}' on the same deals (95% CI).\n")
    for o in args.opponents:
        print(f"against 3x {o}:")
        for c in args.configs:
            w = wins[(c, o)]
            ci = 1.96 * w.std(ddof=1) / np.sqrt(len(w))
            line = f"  {c:<22} win {100 * w.mean():5.1f}% ± {100 * ci:.1f}   margin {margins[(c, o)].mean():+5.2f}"
            if c != base:
                d = w - wins[(base, o)]
                line += f"   vs first {100 * d.mean():+5.1f} ± {196 * d.std(ddof=1) / np.sqrt(len(d)):.1f}"
            print(line)
        print()


if __name__ == "__main__":
    main()
