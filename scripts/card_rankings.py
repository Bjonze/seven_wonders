"""Rank every card with a trained policy; writes results/<name>/ (README, CSV, figures).

    python scripts/card_rankings.py --checkpoint checkpoints/v3/latest.pt --name v3 --low-priority
    python scripts/card_rankings.py --name v3 --report-only     # rebuild report from saved data

Three measurements (see sevenwonders/rl/analysis.py):
  * the bot's own picks: pick rate, win rate when built vs. passed
  * value network: win-chance gain from building a card instead of selling it
  * played out: paired games that split at the first chance to build the card (build it
    vs. sell it) and are played to the end
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevenwonders.cards import ALL_CARDS, CARD_NAMES, COLOR_NAMES, PURPLE  # noqa: E402
from sevenwonders.rl.analysis import OBS_FIELDS, SEAT_FIELDS  # noqa: E402
from sevenwonders.wonders import WONDERS  # noqa: E402

NUM_PLAYERS = 4
EARLY_LAST_TURN = 3  # turns 1-3 = first half of an Age, 4-6 (and Babylon's 7th card) = second half
OBS_SEED = 1_000_000
FORCED_SEED = 2_000_000
AGE_LABEL = {1: "I", 2: "II", 3: "III"}

# Chart tokens (dataviz reference palette): one series -> slot 1 for every bar
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", baseline="#c3c2b7", bar="#2a78d6", late="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", baseline="#383835", bar="#3987e5", late="#d95926"),
}


# -- simulation (worker processes) -------------------------------------------------------
_MODEL = None


def _init_worker(checkpoint: str, low_priority: bool) -> None:
    global _MODEL
    import torch

    from sevenwonders.rl.model import load_model
    from sevenwonders.rl.priority import lower_priority

    torch.set_num_threads(1)
    if low_priority:
        lower_priority("idle")
    _MODEL = load_model(checkpoint)


def _observe_task(seeds):
    from sevenwonders.rl.analysis import observe
    return observe(_MODEL, seeds, sample=True, torch_seed=seeds[0])


def _forced_task(task):
    from sevenwonders.rl.analysis import play_forced
    name, age, mode, seeds = task
    return name, age, mode, play_forced(_MODEL, seeds, name, age, mode, num_players=NUM_PLAYERS)


def chunks(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def card_entries(num_players: int) -> list[tuple[str, int]]:
    """(name, Age) of every card used with this many players (guilds always)."""
    entries = []
    for card in ALL_CARDS:
        used = card.color == PURPLE or min(card.players) <= num_players
        if used and (card.name, card.age) not in entries:
            entries.append((card.name, card.age))
    return entries


def simulate(args, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(str(args.checkpoint), args.low_priority)) as pool:
        t0 = time.time()
        seeds = list(range(OBS_SEED, OBS_SEED + args.games))
        rows, seats = [], []
        for i, part in enumerate(pool.imap_unordered(_observe_task, chunks(seeds, 20)), 1):
            rows.append(part["rows"])
            seats.append(part["seats"])
            if i % 50 == 0:
                print(f"  observed {20 * i} games ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(raw_dir / "observe.npz", rows=np.concatenate(rows), seats=np.concatenate(seats))
        print(f"observation games done in {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        entries = card_entries(NUM_PLAYERS)
        fseeds = list(range(FORCED_SEED, FORCED_SEED + args.forced_games))
        tasks = [(name, age, mode, c) for name, age in entries for mode in ("build", "sell")
                 for c in chunks(fseeds, 250)]
        win = {m: np.zeros((len(entries), len(fseeds)), dtype=np.float32) for m in ("build", "sell")}
        opportunity = {m: np.zeros_like(win[m]) for m in win}
        opp_turn = {m: np.zeros(win[m].shape, dtype=np.int64) for m in win}
        index = {e: i for i, e in enumerate(entries)}
        for k, (name, age, mode, out) in enumerate(pool.imap_unordered(_forced_task, tasks), 1):
            cols = out["seed"] - FORCED_SEED
            win[mode][index[(name, age)], cols] = out["win"]
            opportunity[mode][index[(name, age)], cols] = out["opportunity"]
            opp_turn[mode][index[(name, age)], cols] = out["opportunity_turn"]
            if k % 50 == 0:
                print(f"  build/sell tasks {k}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        # Both runs are identical up to the opportunity, so it must come up in the same games
        assert np.array_equal(opportunity["build"], opportunity["sell"])
        assert np.array_equal(opp_turn["build"], opp_turn["sell"])
        np.savez_compressed(raw_dir / "forced.npz", names=np.asarray([n for n, _ in entries]),
                            ages=np.asarray([a for _, a in entries]), win_build=win["build"],
                            win_sell=win["sell"], opportunity=opportunity["build"],
                            opportunity_turn=opp_turn["build"])
        print(f"build/sell games done in {time.time() - t0:.0f}s", flush=True)


# -- aggregation -------------------------------------------------------------------------
def mean_se(x: np.ndarray) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def build_table(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    obs = np.load(raw_dir / "observe.npz")
    rows = pd.DataFrame(obs["rows"], columns=OBS_FIELDS)
    seats = pd.DataFrame(obs["seats"], columns=SEAT_FIELDS)
    forced = np.load(raw_dir / "forced.npz")
    f_index = {(str(n), int(a)): i for i, (n, a) in enumerate(zip(forced["names"], forced["ages"]))}

    has_turns = "opportunity_turn" in forced.files
    color_of = {(c.name, c.age): c.color for c in ALL_CARDS}
    records = []
    for (card_id, age), g in rows.groupby(["card", "age"]):
        name = CARD_NAMES[int(card_id)]
        age = int(age)
        b = g[g.buildable == 1]
        gain, gain_se = mean_se((b.q_build - b.q_sell).to_numpy())
        best, best_se = mean_se((b.q_build - b.q_best_other).to_numpy())
        early_rows, late_rows = b[b.turn <= EARLY_LAST_TURN], b[b.turn > EARLY_LAST_TURN]
        early, early_se = mean_se((early_rows.q_build - early_rows.q_sell).to_numpy())
        late, late_se = mean_se((late_rows.q_build - late_rows.q_sell).to_numpy())
        fi = f_index[(name, age)]
        seen = forced["opportunity"][fi] == 1
        diff = forced["win_build"][fi] - forced["win_sell"][fi]
        played, played_se = mean_se(diff[seen])
        played_early = played_late = (float("nan"), float("nan"))
        if has_turns:
            turn = forced["opportunity_turn"][fi]
            played_early = mean_se(diff[seen & (turn <= EARLY_LAST_TURN)])
            played_late = mean_se(diff[seen & (turn > EARLY_LAST_TURN)])
        color = color_of[(name, age)]
        records.append({
            "card": name,
            "age": age,
            "type": "guild" if color == PURPLE else COLOR_NAMES[color],
            "gain_vs_sell": gain,
            "gain_vs_sell_se": gain_se,
            "gain_early": early,
            "gain_early_se": early_se,
            "gain_late": late,
            "gain_late_se": late_se,
            "buildable_early": len(early_rows),
            "buildable_late": len(late_rows),
            "pick_rate_early": float(early_rows.built.mean()) if len(early_rows) else float("nan"),
            "pick_rate_late": float(late_rows.built.mean()) if len(late_rows) else float("nan"),
            "gain_vs_best": best,
            "gain_vs_best_se": best_se,
            "pick_rate": float(b.built.mean()) if len(b) else float("nan"),
            "win_if_built": float(b[b.built == 1].win.mean()),
            "win_if_passed": float(b[b.built == 0].win.mean()),
            "played_gain": played,
            "played_gain_se": played_se,
            "played_games": int(seen.sum()),
            "played_gain_early": played_early[0],
            "played_gain_early_se": played_early[1],
            "played_gain_late": played_late[0],
            "played_gain_late_se": played_late[1],
            "times_offered": len(g),
            "times_buildable": len(b),
        })
    table = pd.DataFrame(records).sort_values(["age", "gain_vs_sell"], ascending=[True, False])
    table["rank_in_age"] = table.groupby("age").cumcount() + 1

    wonder_label = {w.id: w.label for w in WONDERS}
    seats["wonder_side"] = seats.wonder.astype(int).map(wonder_label)
    wonders = (seats.groupby("wonder_side")
               .agg(win_rate=("win", "mean"), mean_score=("score", "mean"), seat_games=("win", "size"))
               .sort_values("win_rate", ascending=False).reset_index())
    meta = {
        "games": int(seats.game.nunique()),
        "forced_games": int(forced["win_build"].shape[1]),
        "played_split": has_turns,
    }
    return table, wonders, meta


# -- figures -----------------------------------------------------------------------------
def _style(plt, t):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"],
        "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"],
        "ytick.color": t["ink"],
        "axes.edgecolor": t["baseline"],
    })


def figure_ranking(table: pd.DataFrame, meta: dict, path: Path, theme: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    _style(plt, t)
    counts = [int((table.age == a).sum()) for a in (1, 2, 3)]
    row_h = 0.215
    fig_h = 1.55 + sum(counts) * row_h + 3 * 0.3
    fig, axes = plt.subplots(3, 1, figsize=(9.5, fig_h), sharex=True,
                             gridspec_kw={"height_ratios": counts, "hspace": 0.09})
    lo = min(0.0, float((table.gain_vs_sell - 1.96 * table.gain_vs_sell_se).min())) * 100
    hi = float((table.gain_vs_sell + 1.96 * table.gain_vs_sell_se).max()) * 100
    for ax, age in zip(axes, (1, 2, 3)):
        sub = table[table.age == age].sort_values("gain_vs_sell", ascending=True)
        y = np.arange(len(sub))
        vals = sub.gain_vs_sell.to_numpy() * 100
        err = 1.96 * sub.gain_vs_sell_se.to_numpy() * 100
        ax.barh(y, vals, height=0.62, color=t["bar"], zorder=2)
        ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=t["secondary"], elinewidth=0.8, capsize=0, zorder=3)
        ax.set_yticks(y, sub.card.tolist())
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=0)
        for yi, typ in zip(y, sub.type):
            ax.text(-0.36, yi, typ, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=t["muted"], fontsize=8)
        ax.axvline(0, color=t["baseline"], linewidth=1, zorder=1)
        ax.grid(axis="x", color=t["grid"], linewidth=0.6, zorder=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_xlim(lo - 0.5, hi + 2.2)
        ax.text(-0.36, 1.0, f"Age {AGE_LABEL[age]}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, fontweight="bold", color=t["ink"])
        # Selective direct labels: the three highest-value cards of the Age
        for yi, v, e in list(zip(y, vals, err))[-3:]:
            ax.text(v + e + 0.3, yi, f"+{v:.1f}", va="center", ha="left", fontsize=8, color=t["secondary"])
    axes[-1].set_xlabel("Win-chance gain from building the card instead of selling it (percentage points)")
    axes[-1].tick_params(axis="x", labelbottom=True)
    fig.text(0.02, 1 - 0.32 / fig_h, "Which 7 Wonders cards win games?", fontsize=15, fontweight="bold",
             color=t["ink"], ha="left", va="top")
    fig.text(0.02, 1 - 0.68 / fig_h,
             f"Self-play bot v3, 4 players, {meta['games']:,} games. Bars: how much the bot's value network "
             "says building\nthe card raises its chance of winning, compared with selling it for 3 coins "
             "(mean, 95% CI).",
             fontsize=9, color=t["secondary"], ha="left", va="top")
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 1.25 / fig_h, bottom=0.55 / fig_h)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_timing(table: pd.DataFrame, meta: dict, path: Path, theme: str) -> None:
    """Dumbbell chart: gain vs. selling in the first half (turns 1-3) and second half
    (turns 4-6) of the Age, cards in the same order as the main ranking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = THEMES[theme]
    _style(plt, t)
    counts = [int((table.age == a).sum()) for a in (1, 2, 3)]
    row_h = 0.215
    fig_h = 2.05 + sum(counts) * row_h + 3 * 0.3
    fig, axes = plt.subplots(3, 1, figsize=(9.5, fig_h), sharex=True,
                             gridspec_kw={"height_ratios": counts, "hspace": 0.09})
    vals = np.concatenate([table.gain_early.to_numpy(), table.gain_late.to_numpy()]) * 100
    lo, hi = min(0.0, np.nanmin(vals)) - 0.5, np.nanmax(vals) + 2.8
    for ax, age in zip(axes, (1, 2, 3)):
        sub = table[table.age == age].sort_values("gain_vs_sell", ascending=True)
        y = np.arange(len(sub))
        early = sub.gain_early.to_numpy() * 100
        late = sub.gain_late.to_numpy() * 100
        ax.hlines(y, np.minimum(early, late), np.maximum(early, late), color=t["baseline"], linewidth=2, zorder=2)
        # Late dot larger and underneath, so it still shows as a ring when the two coincide
        ax.scatter(late, y, s=50, color=t["late"], edgecolors=t["surface"], linewidths=1.2, zorder=3)
        ax.scatter(early, y, s=22, color=t["bar"], edgecolors=t["surface"], linewidths=1.0, zorder=4)
        ax.set_yticks(y, sub.card.tolist())
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=0)
        for yi, typ in zip(y, sub.type):
            ax.text(-0.36, yi, typ, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=t["muted"], fontsize=8)
        ax.axvline(0, color=t["baseline"], linewidth=1, zorder=1)
        ax.grid(axis="x", color=t["grid"], linewidth=0.6, zorder=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_xlim(lo, hi)
        ax.text(-0.36, 1.0, f"Age {AGE_LABEL[age]}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, fontweight="bold", color=t["ink"])
        # Selective direct labels: the biggest shift towards late and towards early (if any)
        shift = late - early
        picks = []
        if np.nanmax(shift) > 0:
            picks.append(int(np.nanargmax(shift)))
        if np.nanmin(shift) < 0:
            picks.append(int(np.nanargmin(shift)))
        for i in picks:
            ax.text(max(early[i], late[i]) + 0.35, y[i], f"{shift[i]:+.1f} late vs. early", va="center",
                    ha="left", fontsize=8, color=t["secondary"])
    axes[-1].set_xlabel("Win-chance gain from building the card instead of selling it (percentage points)")
    fig.text(0.02, 1 - 0.32 / fig_h, "When is each card strong?", fontsize=15, fontweight="bold",
             color=t["ink"], ha="left", va="top")
    fig.text(0.02, 1 - 0.68 / fig_h,
             f"Same measure as the main ranking, split by when in the Age the card could be built "
             f"(self-play bot, {meta['games']:,} games).\nCards in the same order as the main ranking.",
             fontsize=9, color=t["secondary"], ha="left", va="top")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5.5, color=t["bar"], label="Turns 1–3"),
               Line2D([], [], marker="o", linestyle="", markersize=8, color=t["late"], label="Turns 4–6")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 1 - 1.0 / fig_h), ncol=2,
               frameon=False, fontsize=9, labelcolor=t["ink"], handletextpad=0.3, columnspacing=1.5)
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 1.75 / fig_h, bottom=0.55 / fig_h)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_agreement(table: pd.DataFrame, path: Path, theme: str) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    _style(plt, t)
    sub = table[table.played_games >= 200]
    x = sub.gain_vs_sell.to_numpy() * 100
    y = sub.played_gain.to_numpy() * 100
    r = float(np.corrcoef(x, y)[0, 1])
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    lo, hi = min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1
    ax.plot([lo, hi], [lo, hi], color=t["baseline"], linewidth=1, zorder=1)
    ax.text(hi, hi, "estimate = played out ", ha="right", va="bottom", fontsize=8, color=t["muted"],
            rotation=0)
    ax.axhline(0, color=t["grid"], linewidth=1, zorder=1)
    ax.axvline(0, color=t["grid"], linewidth=1, zorder=1)
    ax.scatter(x, y, s=26, color=t["bar"], edgecolors=t["surface"], linewidths=1.2, zorder=3)
    ax.grid(color=t["grid"], linewidth=0.6, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.tick_params(axis="y", colors=t["muted"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Label the extremes; labels near the right edge go to the left of their dot
    label_idx = set(np.argsort(x)[-3:]) | set(np.argsort(y)[-2:]) | set(np.argsort(y - x)[:2]) \
        | set(np.argsort(y - x)[-2:])
    renderer = fig.canvas.get_renderer()
    taken = []  # window extents of labels already placed
    for i in sorted(label_idx, key=lambda k: -y[k]):
        text = f"{sub.card.iloc[i]} ({AGE_LABEL[int(sub.age.iloc[i])]})"
        right = x[i] > lo + 0.75 * (hi - lo)
        candidates = [(-6, 3, "right"), (-6, -11, "right"), (6, 3, "left"), (6, -11, "left")] if right \
            else [(6, 3, "left"), (6, -11, "left"), (-6, 3, "right"), (-6, -11, "right")]
        for dx, dy, ha in candidates:
            ann = ax.annotate(text, (x[i], y[i]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                              ha=ha, color=t["secondary"])
            box = ann.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(b) for b in taken):
                taken.append(box)
                break
            ann.remove()
    ax.set_xlabel("Value network's estimate (percentage points of win chance)")
    ax.set_ylabel("Played out: win rate if built − if sold (pp)")
    fig.text(0.02, 0.975, "Does the value network's estimate hold up?", fontsize=13, fontweight="bold",
             ha="left", va="top")
    fig.text(0.02, 0.918,
             "Each dot is a card: gain from building it instead of selling it. x: the value network's estimate. "
             "y: paired games\nthat split at the first chance to build the card and are played to the end. "
             f"Correlation r = {r:.2f}.",
             fontsize=8.5, color=t["secondary"], ha="left", va="top")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.10)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return r


# -- report ------------------------------------------------------------------------------
def pp(v: float, se: float | None = None, sign: bool = True) -> str:
    if np.isnan(v):
        return "–"
    s = f"{v * 100:+.1f}" if sign else f"{v * 100:.1f}"
    s = s.replace("-", "−")
    if se is not None and not np.isnan(se):
        ci = 1.96 * se * 100
        s += f" ± {ci:.1f}" if ci >= 0.05 else " ± <0.1"
    return s


def picture(name: str, alt: str) -> str:
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="figures/{name}_dark.png">\n'
            f'  <img alt="{alt}" src="figures/{name}_light.png">\n</picture>')


def write_report(out: Path, table: pd.DataFrame, wonders: pd.DataFrame, meta: dict, r: float,
                 checkpoint: str) -> None:
    lines = [
        f"# Card rankings: bot {out.name}",
        "",
        f"Bot `{checkpoint}` playing itself in 4-player games of 7 Wonders (2nd edition, base game). "
        f"{meta['games']:,} games were analysed decision by decision, and for every card "
        f"{meta['forced_games']:,} deals were played twice (build it vs. sell it at the first chance) "
        "to check the estimates.",
        "",
        "## Top cards per Age",
        "",
    ]
    for age in (1, 2, 3):
        top = table[table.age == age].head(5)
        items = ", ".join(f"**{c}** ({pp(v)})" for c, v in zip(top.card, top.gain_vs_sell))
        lines.append(f"- **Age {AGE_LABEL[age]}:** {items}")
    lines += [
        "",
        "Numbers are percentage points of win chance gained by building the card instead of selling it.",
        "",
        picture("card_value", "Win-chance gain per card, by Age"),
        "",
        "## When is each card strong?",
        "",
        "The same measure, split by when in the Age the card could be built: the first half (turns 1–3) or "
        "the second half (turns 4–6, including Babylon's extra 7th card).",
        "",
        picture("card_timing", "Win-chance gain per card in the first and second half of each Age"),
        "",
    ]
    for age in (1, 2, 3):
        sub = table[(table.age == age) & (table.buildable_early >= 300) & (table.buildable_late >= 300)].copy()
        sub["shift"] = sub.gain_late - sub.gain_early
        later = sub.sort_values("shift", ascending=False).head(3)
        earlier = sub[sub["shift"] < 0].sort_values("shift").head(3)
        fmt = lambda df: ", ".join(f"{c} ({pp(s)})" for c, s in zip(df.card, df["shift"]))  # noqa: E731
        if len(earlier):
            other = f"stronger early: {fmt(earlier)}"
        else:
            other = f"no card is stronger early; least gain from waiting: {fmt(sub.sort_values('shift').head(3))}"
        lines.append(f"- **Age {AGE_LABEL[age]}:** stronger late: {fmt(later)}; {other}")
    lines += [
        "",
        "Numbers in brackets: late minus early, in percentage points. Only cards that could be built at least "
        "300 times in each half are listed.",
        "",
        "Late decisions tend to swing the win chance more in general, because fewer turns remain for anyone to "
        "respond (most visible in Age III, where nearly every card gains late). So compare a card's shift with "
        "the other cards of the same Age rather than with zero.",
    ]
    if meta.get("played_split"):
        lines += ["", "The played-out check is split the same way in `card_rankings.csv` "
                      "(`played_gain_early` / `played_gain_late`)."]
    lines += [
        "",
        "## How to read this",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| **Gain vs. sell** | Main ranking. At every decision where the card could be built, the bot's value network "
        "rates the position after building it and after selling it for 3 coins (other players' moves that turn "
        "unchanged). The difference is the change in the bot's estimated chance of winning, in percentage points, "
        "averaged over all those decisions, ± 95% CI. |",
        "| **Early / Late** | Gain vs. sell, only counting decisions in turns 1–3 / turns 4–6 of the Age. |",
        "| **Gain vs. best other** | Same, but compared with the best other option in that hand (building another "
        "card, a wonder stage, selling). Negative = usually something else in the hand was better. |",
        "| **Pick rate** | How often the bot builds the card when it is in its hand and affordable. |",
        "| **Win% built / passed** | The bot's final win rate when it built the card vs. when it could have but didn't. "
        "Raw correlation, not cause: e.g. players who are already ahead can afford expensive cards. 25% = average. |",
        "| **Played out** | The same question as *gain vs. sell*, answered by playing games instead of asking "
        "the network. Each deal is played twice with the bot in every seat; the first time one seat can build "
        "the card, it builds it in one copy and sells it in the other, then both games are played to the end. "
        "Win-rate difference ± 95% CI, over the deals where that chance came up (n). |",
        "| **Offered** | Number of times the card was in a hand of the analysed games. |",
        "",
    ]
    for age in (1, 2, 3):
        sub = table[table.age == age]
        lines += [
            f"## Age {AGE_LABEL[age]}",
            "",
            "| # | Card | Type | Gain vs. sell | Early | Late | Gain vs. best other | Pick rate | "
            "Win% built / passed | Played out | Offered |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sub.itertuples():
            forced = (f"{pp(row.played_gain, row.played_gain_se)} (n={row.played_games:,})"
                      if row.played_games >= 30 else "–")
            lines.append(
                f"| {row.rank_in_age} | {row.card} | {row.type} | {pp(row.gain_vs_sell, row.gain_vs_sell_se)} | "
                f"{pp(row.gain_early)} | {pp(row.gain_late)} | "
                f"{pp(row.gain_vs_best)} | {row.pick_rate * 100:.0f}% | "
                f"{row.win_if_built * 100:.0f}% / {row.win_if_passed * 100:.0f}% | {forced} | "
                f"{row.times_offered:,} |")
        lines.append("")
    lines += [
        "## Does the value network's estimate hold up?",
        "",
        picture("measures_agreement", "Value-network estimate vs. played-out gain per card"),
        "",
        f"Across cards the estimate and the played-out result correlate with r = {r:.2f}. Dots on the diagonal "
        "mean the network judged that card right; dots below it mean the network overrates the card. The two "
        "measures also average over slightly different situations: the estimate covers every decision where "
        "the card could be built, the played-out games only the *first* such chance in a game.",
        "",
        "## Wonder sides",
        "",
        "Win rate of each wonder side in the same self-play games (25% = average).",
        "",
        "| Wonder side | Win rate | Mean score | Seat-games |",
        "|---|---:|---:|---:|",
    ]
    for row in wonders.itertuples():
        lines.append(f"| {row.wonder_side} | {row.win_rate * 100:.1f}% | {row.mean_score:.1f} | {row.seat_games:,} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- These are the preferences of one self-taught bot playing copies of itself. A different style of play at "
        "your table (e.g. nobody contesting science) changes what is strong.",
        "- The value network is an estimate (it explains roughly a third of the variance in who wins), so treat small "
        "differences between neighbouring cards as ties. Confidence intervals treat decisions as independent, "
        "so they are somewhat too narrow.",
        "- Card values depend on context (wonder, neighbours, Age, what you already own); these are averages over "
        "the situations where the bot could build the card.",
        "",
        "## Files",
        "",
        "- `card_rankings.csv`: all numbers above at full precision",
        "- `wonders.csv`: wonder-side table",
        "- `figures/`: charts (light and dark versions)",
        "",
        f"Reproduce: `python scripts/card_rankings.py --checkpoint {checkpoint} --name {out.name}`",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/v3/latest.pt")
    p.add_argument("--name", default="v3", help="output folder results/<name>")
    p.add_argument("--games", type=int, default=8000, help="observation games")
    p.add_argument("--forced-games", type=int, default=4000, help="paired games per card")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--low-priority", action="store_true")
    p.add_argument("--report-only", action="store_true", help="skip simulation, reuse saved raw data")
    args = p.parse_args()

    out = ROOT / "results" / args.name
    raw_dir = out / "raw"
    if not args.report_only:
        simulate(args, raw_dir)
    table, wonders, meta = build_table(raw_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "card_rankings.csv", index=False, float_format="%.5f")
    wonders.to_csv(out / "wonders.csv", index=False, float_format="%.5f")
    r = 0.0
    for theme in THEMES:
        figure_ranking(table, meta, out / "figures" / f"card_value_{theme}.png", theme)
        figure_timing(table, meta, out / "figures" / f"card_timing_{theme}.png", theme)
        r = figure_agreement(table, out / "figures" / f"measures_agreement_{theme}.png", theme)
    write_report(out, table, wonders, meta, r, Path(args.checkpoint).as_posix())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
