"""PPO self-play training.

    python scripts/train.py --run v4 --iterations 4000 --low-priority --anneal-lr \
        --payment-choice --hidden-discard --arch resmlp --hidden 1024 --layers 3 \
        --pool-init checkpoints/v3/latest.pt --eval-checkpoint checkpoints/v3/latest.pt

Fine-tune from an earlier model; inputs/outputs it lacks (e.g. side choice) start at zero:

    python scripts/train.py --run v5a --init checkpoints/v4/latest.pt --side-choice ...

Continue after an interruption (settings, optimizer, opponent pool, best score and wandb run
all come from the checkpoint; options given on the command line override them):

    python scripts/train.py --resume checkpoints/v4/latest.pt

Rollouts and evaluation games run on CPU worker processes; the PPO update runs on the GPU.
Metrics go to the console, logs/<run>.log, TensorBoard (runs/<run>) and Weights & Biases.

Checkpoints in checkpoints/<run>/ (all written to a temporary file and then renamed, so an
interruption while saving cannot leave a broken file):
    latest.pt           every --save-every iterations, and when training is interrupted
    iter_XXXXX.pt       rolling copies; the newest --keep-rolling are kept
    best.pt             best value of --best-metric so far
    pool/iter_XXXXX.pt  opponent-pool snapshots (weights only)
latest/iter/best include the optimizer, opponent pool, RNG states and everything else
needed to resume.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevenwonders.game import CHOOSE_SIDE, PAY_LEFT, PAY_RIGHT  # noqa: E402
from sevenwonders.rl.agent import evaluate_games, summarize  # noqa: E402
from sevenwonders.rl.encoding import (  # noqa: E402
    SCORE_KEYS, EncodingConfig, action_kind_and_pay, num_actions, obs_dim,
)
from sevenwonders.rl.model import PolicyValueNet, warm_start  # noqa: E402
from sevenwonders.rl.ppo import PPOConfig, all_finite, ppo_update  # noqa: E402
from sevenwonders.rl.priority import keep_awake, lower_priority  # noqa: E402
from sevenwonders.rl.rollout import collect, worker_main  # noqa: E402
from sevenwonders.wonders import WONDERS  # noqa: E402

EVAL_SEED = 10_000  # same evaluation games every time, so evaluations are comparable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="baseline", help="run name (logs and checkpoints)")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="worker processes (default: CPU count - 1); 0 = everything in the main process")
    p.add_argument("--games-per-worker", type=int, default=128)
    p.add_argument("--low-priority", action="store_true",
                   help="run below normal priority so other programs (games) get the CPU first")
    p.add_argument("--keep-awake", action="store_true",
                   help="stop Windows from idle-sleeping while training runs")
    # model and observation
    p.add_argument("--arch", choices=["mlp", "resmlp"], default="mlp")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3, help="residual blocks (resmlp)")
    p.add_argument("--payment-choice", action="store_true",
                   help="let the policy choose whom to pay for resources (cheapest / pay-left / pay-right)")
    p.add_argument("--hidden-discard", action="store_true",
                   help="discard pile is face down: a seat only sees its own discards and the pile size")
    p.add_argument("--side-choice", action="store_true",
                   help="players pick their wonder side (Day/Night) at the start, as in the real game")
    p.add_argument("--repeat-frac", type=float, default=0.0,
                   help="share of training games dealt with repeated wonders (training variant)")
    p.add_argument("--init", default=None,
                   help="start from this checkpoint's weights; new inputs/outputs (e.g. side choice) start at 0")
    # PPO
    p.add_argument("--reward", choices=["win", "rank"], default="win")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action="store_true", help="decay the learning rate linearly to 0")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--ent-coef", type=float, default=0.01)
    # opponent pool
    p.add_argument("--pool-init", action="append", default=[],
                   help="checkpoint that is always in the opponent pool; repeatable")
    p.add_argument("--pool-frac", type=float, default=0.3,
                   help="share of training games where 1-3 seats are played by a pool opponent")
    p.add_argument("--pool-every", type=int, default=50, help="add a snapshot of the learner every N iterations")
    p.add_argument("--pool-size", type=int, default=10, help="max learner snapshots in the pool")
    p.add_argument("--pool-prioritize", type=float, default=0.0,
                   help="0 = pick pool opponents uniformly; 1 = by difficulty only "
                        "(weight (1 - learner win rate vs. it)^2); in between = mix")
    # evaluation and checkpoints
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=400, help="games per evaluation opponent")
    p.add_argument("--eval-bots", default="greedy", help="comma-separated baseline bots to evaluate against")
    p.add_argument("--eval-checkpoint", action="append", default=[],
                   help="also evaluate against 3 copies of this (older) policy; repeatable")
    p.add_argument("--best-metric", default=None,
                   help="metric for best.pt (default: win rate vs. the first --eval-checkpoint, else greedy)")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--keep-rolling", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="seven_wonders")
    p.add_argument("--wandb-entity", default=None, help="default: your wandb default entity")
    return p


def parse_args():
    """Command-line args; when resuming, options not given on the command line come from the
    checkpoint."""
    parser = build_parser()
    args = parser.parse_args()
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        given = {a.dest for a in parser._actions
                 if any(arg == opt or arg.startswith(opt + "=") for arg in sys.argv[1:] for opt in a.option_strings)}
        for key, value in ckpt["args"].items():
            if key not in given and key != "resume":
                setattr(args, key, value)
    return args, ckpt


def numpy_weights(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class OpponentPool:
    """Frozen opponents: fixed checkpoints (--pool-init) plus rolling learner snapshots."""

    def __init__(self, directory: Path, max_snapshots: int, prioritize: float = 0.0):
        self.directory = directory
        self.max_snapshots = max_snapshots
        self.prioritize = prioritize
        self.entries: list[dict] = []  # {"id", "path", "fixed", "win_rate"}
        self._weights: dict[str, tuple[dict, dict]] = {}  # id -> (config, numpy weights)
        self._sent: dict[int, set[str]] = {}  # worker -> ids it has cached

    def record(self, opponent_index: np.ndarray, learner_win: np.ndarray) -> None:
        """Update each opponent's running learner win rate (per learner seat; 0.25 = even)."""
        for i, e in enumerate(self.entries):
            wins = learner_win[opponent_index == i]
            if len(wins):
                e["win_rate"] = 0.9 * e.get("win_rate", 0.25) + 0.1 * float(wins.mean())

    def sampling_weights(self) -> list[float]:
        n = len(self.entries)
        hard = [(1.0 - e.get("win_rate", 0.25)) ** 2 for e in self.entries]
        total = sum(hard) or 1.0
        return [(1 - self.prioritize) / n + self.prioritize * h / total for h in hard]

    def add(self, entry_id: str, path: Path, fixed: bool, win_rate: float = 0.25) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._weights[entry_id] = (ckpt["model_config"], {k: v.numpy() for k, v in ckpt["model"].items()})
        self.entries.append({"id": entry_id, "path": str(path), "fixed": fixed, "win_rate": win_rate})

    def snapshot(self, model: PolicyValueNet, iteration: int) -> None:
        path = self.directory / f"iter_{iteration:05d}.pt"
        atomic_save({"model": model.state_dict(), "model_config": model.config(), "iteration": iteration}, path)
        self.add(f"iter_{iteration:05d}", path, fixed=False)
        snapshots = [e for e in self.entries if not e["fixed"]]
        for old in snapshots[:max(0, len(snapshots) - self.max_snapshots)]:
            self.entries.remove(old)
            del self._weights[old["id"]]
            Path(old["path"]).unlink(missing_ok=True)

    def payload(self, worker: int) -> list[tuple]:
        """(id, config, weights or None if the worker already has it) for every entry."""
        sent = self._sent.setdefault(worker, set())
        out = []
        for e in self.entries:
            config, weights = self._weights[e["id"]]
            out.append((e["id"], config, None if e["id"] in sent else weights))
            sent.add(e["id"])
        for gone in sent - {e["id"] for e in self.entries}:
            sent.discard(gone)
        return out

    def models(self) -> list[PolicyValueNet]:
        """Opponent networks for in-process collection (--workers 0)."""
        nets = []
        for e in self.entries:
            config, weights = self._weights[e["id"]]
            net = PolicyValueNet(**config)
            net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            nets.append(net)
        return nets

    def state(self) -> list[dict]:
        return [dict(e) for e in self.entries]


class Logger:
    """Console + log file + TensorBoard + (optional) Weights & Biases."""

    def __init__(self, args, config: dict, wandb_id: str | None = None):
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        self.file = open(log_dir / f"{args.run}.log", "a", encoding="utf-8")
        self.writer = SummaryWriter(str(ROOT / "runs" / args.run))
        self.wandb = None
        if args.wandb:
            import wandb

            self.wandb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                    name=args.run, config=config, dir=str(ROOT),
                                    id=wandb_id, resume="must" if wandb_id else None)
            self.line(f"wandb: {self.wandb.url}")

    def line(self, text: str) -> None:
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()

    def metrics(self, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.writer.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def close(self) -> None:
        self.writer.close()
        self.file.close()
        if self.wandb is not None:
            self.wandb.finish()


def selfplay_metrics(data: dict[str, np.ndarray], num_players: int,
                     enc: EncodingConfig) -> dict[str, float]:
    kind, pay = action_kind_and_pay(data["action"], enc)
    card_moves = kind != CHOOSE_SIDE
    kind, pay = kind[card_moves], pay[card_moves]
    kinds = np.bincount(kind, minlength=3) / len(kind)
    builds = kind != 2
    out = {
        "selfplay/frac_build": float(kinds[0]),
        "selfplay/frac_wonder": float(kinds[1]),
        "selfplay/frac_sell": float(kinds[2]),
        "selfplay/frac_pay_left": float((pay[builds] == PAY_LEFT).mean()),
        "selfplay/frac_pay_right": float((pay[builds] == PAY_RIGHT).mean()),
    }
    if len(data["breakdowns"]):
        breakdowns = data["breakdowns"].reshape(-1, num_players, len(SCORE_KEYS))
        total = breakdowns[..., SCORE_KEYS.index("total")]
        out.update({
            "selfplay/score_mean": float(total.mean()),
            "selfplay/winner_score_mean": float(total.max(axis=1).mean()),
            "selfplay/trade_coins_per_player": float(data["trade_paid"].mean()),
        })
        for i, key in enumerate(SCORE_KEYS):
            if key != "total":
                out[f"selfplay/points_{key}"] = float(breakdowns[..., i].mean())
    if len(data["pool_win"]):
        out["league/win_rate_vs_pool"] = float(data["pool_win"].mean())
    out["league/learner_seats_in_pool_games"] = float(len(data["pool_win"]))
    out["selfplay/repeated_wonder_games"] = float(data["repeated_games"].sum())
    return out


def main() -> None:
    args, ckpt = parse_args()
    if args.low_priority:
        lower_priority("below_normal")
    if args.keep_awake:
        keep_awake()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    enc = EncodingConfig(payment_choice=args.payment_choice, hidden_discard=args.hidden_discard,
                         side_choice=args.side_choice)
    model = PolicyValueNet(obs_dim(args.players, enc), num_actions(enc), args.hidden, args.arch,
                           args.layers, enc.to_dict()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    ckpt_dir = ROOT / "checkpoints" / args.run
    pool = OpponentPool(ckpt_dir / "pool", args.pool_size, args.pool_prioritize)
    start, total_games, wandb_id = 0, 0, None
    best = {"metric": args.best_metric, "value": -1.0, "iteration": -1}
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["iteration"] + 1
        total_games = ckpt.get("total_games", 0)
        wandb_id = ckpt.get("wandb_id")
        best = ckpt.get("best", best)
        for e in ckpt.get("pool", []):
            pool.add(e["id"], Path(e["path"]), e["fixed"], e.get("win_rate", 0.25))
        rng = ckpt.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
    else:
        if args.init:
            warm_start(model, torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        for path in args.pool_init:
            pool.add(Path(path).parent.name, Path(path), fixed=True)
    cfg = PPOConfig(lr=args.lr, epochs=args.epochs, minibatch=args.minibatch, ent_coef=args.ent_coef)

    # Fixed opponents for evaluation
    eval_opponents: list[tuple[str, str, tuple | None]] = [
        (name, name, None) for name in args.eval_bots.split(",") if name]
    for path in args.eval_checkpoint:
        ref = torch.load(path, map_location="cpu", weights_only=False)
        ref_weights = {k: v.numpy() for k, v in ref["model"].items()}
        eval_opponents.append((Path(path).parent.name, "checkpoint", (ref["model_config"], ref_weights)))
    if best["metric"] is None:
        first = next((name for name, kind, _ in eval_opponents if kind == "checkpoint"), "greedy")
        best["metric"] = f"eval_{first}/win_rate"

    n_params = sum(p.numel() for p in model.parameters())
    config = {**vars(args), **model.config(), "num_params": n_params,
              "games_per_iteration": max(1, args.workers) * args.games_per_worker}
    log = Logger(args, config, wandb_id)
    wandb_id = log.wandb.id if log.wandb is not None else None
    if ckpt is not None:
        log.line(f"resumed from {args.resume} at iteration {start} ({total_games:,} games, "
                 f"best {best['metric']} = {best['value']:.3f} at iteration {best['iteration']})")
    log.line(f"device {device}, obs {model.obs_dim}, actions {model.num_actions}, params {n_params:,}, "
             f"arch {args.arch} {args.hidden}x{args.layers}, workers {args.workers}, low priority "
             f"{args.low_priority}, encoding {enc}, pool {[e['id'] for e in pool.entries]}")

    cpu_model = PolicyValueNet(**model.config())
    ctx = mp.get_context("spawn")
    workers = []
    for _ in range(args.workers):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=worker_main, daemon=True,
                           args=(child, args.players, model.config(), args.reward, args.gamma,
                                 args.lam, "idle" if args.low_priority else None))
        proc.start()
        workers.append((parent, proc))

    def run_eval(weights, opponent_kind: str, opponent_ckpt) -> dict[str, float]:
        ids = list(range(args.eval_games))
        if not workers:
            cpu_model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            ref_model = None
            if opponent_ckpt is not None:
                ref_model = PolicyValueNet(**opponent_ckpt[0])
                ref_model.load_state_dict({k: torch.from_numpy(v) for k, v in opponent_ckpt[1].items()})
            return summarize(evaluate_games(cpu_model, opponent_kind, ids, args.players, EVAL_SEED, ref_model,
                                            args.side_choice))
        chunks = [ids[w::len(workers)] for w in range(len(workers))]
        for (conn, _), chunk in zip(workers, chunks):
            conn.send(("evaluate", (weights, opponent_kind, opponent_ckpt, chunk, EVAL_SEED, args.side_choice)))
        return summarize(concat([conn.recv() for conn, _ in workers]))

    def state(iteration: int) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model.config(),
            "iteration": iteration,
            "total_games": total_games,
            "num_players": args.players,
            "args": vars(args),
            "wandb_id": wandb_id,
            "pool": pool.state(),
            "best": dict(best),
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
        }

    def save(iteration: int) -> None:
        if not all_finite(model):
            log.line(f"NOT saving iteration {iteration}: the model has non-finite weights")
            return
        s = state(iteration)
        atomic_save(s, ckpt_dir / "latest.pt")
        atomic_save(s, ckpt_dir / f"iter_{iteration:05d}.pt")
        rolling = sorted(ckpt_dir.glob("iter_*.pt"))
        for old in rolling[:max(0, len(rolling) - args.keep_rolling)]:
            old.unlink(missing_ok=True)

    wonder_label = {w.id: w.label for w in WONDERS}
    wonder_wins: list[np.ndarray] = []
    wonder_ids: list[np.ndarray] = []
    run_start = time.time()
    last_done, last_saved = start - 1, start - 1
    crashed = False
    try:
        for it in range(start, args.iterations):
            if args.anneal_lr:
                lr = args.lr * (1.0 - it / args.iterations)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            if it > 0 and it % args.pool_every == 0:
                pool.snapshot(model, it)
            t0 = time.time()
            weights = numpy_weights(model)
            seed = args.seed + it * 1_000_003
            options = {"pool_frac": args.pool_frac, "opponent_weights": pool.sampling_weights() or None,
                       "choose_sides": args.side_choice, "repeat_frac": args.repeat_frac}
            if workers:
                for w, (conn, _) in enumerate(workers):
                    conn.send(("collect", (weights, seed + w * 10_007, args.games_per_worker,
                                           pool.payload(w), options)))
                data = concat([conn.recv() for conn, _ in workers])
            else:
                cpu_model.load_state_dict(model.state_dict())
                data = collect(cpu_model, args.games_per_worker, args.players, seed, args.reward,
                               args.gamma, args.lam, pool.models(), **options)
            pool.record(data["pool_opponent"], data["pool_win"])
            t_collect = time.time() - t0

            t1 = time.time()
            stats = ppo_update(model, optimizer, data, cfg, device)
            if not all_finite(model):
                raise FloatingPointError(f"non-finite weights after the update of iteration {it}")
            t_update = time.time() - t1

            games = int(data["games"].sum())
            total_games += games
            ret, val = data["ret"], data["value"]
            explained_var = float(1.0 - np.var(ret - val) / max(np.var(ret), 1e-8))
            metrics = {f"train/{k}": v for k, v in stats.items()}
            metrics.update({
                "train/explained_var": explained_var,
                "train/lr": optimizer.param_groups[0]["lr"],
                "perf/games_per_sec": games / t_collect,
                "perf/collect_sec": t_collect,
                "perf/update_sec": t_update,
                "perf/samples": float(len(data["action"])),
                "progress/total_games": float(total_games),
                "progress/hours": (time.time() - run_start) / 3600,
                "league/pool_size": float(len(pool.entries)),
            })
            metrics.update(selfplay_metrics(data, args.players, enc))
            for e in pool.entries:
                if e["fixed"]:
                    metrics[f"league/win_rate_vs_{e['id']}"] = e["win_rate"]
            wonder_wins.append(data["win"])
            wonder_ids.append(data["wonder"])
            log.metrics(metrics, it)
            log.line(f"it {it:5d} | games {total_games:9,d} | {games / t_collect:5.0f} g/s | "
                     f"score {metrics.get('selfplay/score_mean', float('nan')):5.1f} | "
                     f"vs pool {metrics.get('league/win_rate_vs_pool', float('nan')):5.3f} | "
                     f"ent {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f} | "
                     f"ev {explained_var:5.2f} | {t_collect:4.1f}s+{t_update:4.1f}s")
            last_done = it

            final = it == args.iterations - 1
            if it % args.eval_every == 0 or final:
                t2 = time.time()
                eval_metrics, parts = {}, []
                eval_weights = numpy_weights(model)
                for name, kind, opponent_ckpt in eval_opponents:
                    res = run_eval(eval_weights, kind, opponent_ckpt)
                    eval_metrics.update({f"eval_{name}/{k}": v for k, v in res.items()})
                    parts.append(f"vs {name}: win {100 * res['win_rate']:5.1f}% "
                                 f"score {res['mean_score']:5.1f} margin {res['mean_margin']:+5.1f}")
                # Win rate per wonder side in the self-play games since the last evaluation
                ids, wins = np.concatenate(wonder_ids), np.concatenate(wonder_wins)
                for wid, label in wonder_label.items():
                    if (ids == wid).sum() >= 50:
                        eval_metrics[f"wonder/{label}"] = float(wins[ids == wid].mean())
                if args.side_choice:  # how often each wonder is played on its Night side
                    for name in sorted({w.name for w in WONDERS}):
                        day, night = (next(w.id for w in WONDERS if w.name == name and w.side == s)
                                      for s in ("day", "night"))
                        n_day, n_night = (ids == day).sum(), (ids == night).sum()
                        if n_day + n_night >= 50:
                            eval_metrics[f"side/night_share/{name}"] = float(n_night / (n_day + n_night))
                wonder_ids.clear()
                wonder_wins.clear()
                eval_metrics["perf/eval_sec"] = time.time() - t2
                value = eval_metrics.get(best["metric"])
                if value is not None and value > best["value"]:
                    best.update(value=float(value), iteration=it)
                    if all_finite(model):
                        atomic_save(state(it), ckpt_dir / "best.pt")
                    parts.append(f"new best {best['metric']} = {value:.3f}")
                eval_metrics["progress/best"] = best["value"]
                log.metrics(eval_metrics, it)
                log.line("      eval | " + " | ".join(parts))
            if it % args.save_every == 0 or final:
                save(it)
                last_saved = it
    except (KeyboardInterrupt, EOFError, ConnectionError) as exc:
        # Interrupted from outside (Ctrl+C, shutdown): the model is fine, save it below
        log.line(f"interrupted ({type(exc).__name__}) after iteration {last_done}")
    except Exception as exc:
        # A crash may have left the model half-updated or broken: keep the last good files
        crashed = True
        log.line(f"crashed ({type(exc).__name__}: {exc}); not saving, resume from the last "
                 f"checkpoint (iteration {last_saved})")
        raise
    finally:
        if not crashed and last_done > last_saved:
            save(last_done)
            log.line(f"saved checkpoint at iteration {last_done}")
        for conn, proc in workers:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for _, proc in workers:
            proc.join(timeout=5)
        log.close()


if __name__ == "__main__":
    main()
