"""PPO self-play training.

    python scripts/train.py --run v2 --iterations 1000 --low-priority --eval-checkpoint checkpoints/v1/latest.pt

Rollouts and evaluation games run on CPU worker processes (one shared policy plays every
seat); the PPO update runs on the GPU. Metrics go to the console, logs/<run>.log,
TensorBoard (runs/<run>) and Weights & Biases. Checkpoints go to checkpoints/<run>/.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevenwonders.rl.agent import evaluate_games, summarize  # noqa: E402
from sevenwonders.rl.encoding import NUM_ACTIONS, SCORE_KEYS, obs_dim  # noqa: E402
from sevenwonders.rl.model import PolicyValueNet  # noqa: E402
from sevenwonders.rl.ppo import PPOConfig, ppo_update  # noqa: E402
from sevenwonders.rl.priority import lower_priority  # noqa: E402
from sevenwonders.rl.rollout import collect, worker_main  # noqa: E402

EVAL_SEED = 10_000  # same evaluation games every time, so evaluations are comparable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="baseline", help="run name (logs and checkpoints)")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="worker processes (default: CPU count - 1); 0 = everything in the main process")
    p.add_argument("--games-per-worker", type=int, default=128)
    p.add_argument("--low-priority", action="store_true",
                   help="run below normal priority so other programs (games) get the CPU first")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--reward", choices=["win", "rank"], default="win")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action="store_true", help="decay the learning rate linearly to 0")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=400, help="games per evaluation opponent")
    p.add_argument("--eval-checkpoint", default=None,
                   help="also evaluate against 3 copies of this (older) policy")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="seven_wonders")
    p.add_argument("--wandb-entity", default=None, help="default: your wandb default entity")
    return p.parse_args()


def numpy_weights(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def save(path: Path, model, optimizer, iteration: int, total_games: int, args,
         wandb_id: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.config(),
        "iteration": iteration,
        "total_games": total_games,
        "num_players": args.players,
        "args": vars(args),
        "wandb_id": wandb_id,
    }, path)


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


def selfplay_metrics(data: dict[str, np.ndarray], num_players: int) -> dict[str, float]:
    breakdowns = data["breakdowns"].reshape(-1, num_players, len(SCORE_KEYS))
    total = breakdowns[..., SCORE_KEYS.index("total")]
    kinds = np.bincount(data["action"] % 3, minlength=3) / len(data["action"])
    out = {
        "selfplay/score_mean": float(total.mean()),
        "selfplay/winner_score_mean": float(total.max(axis=1).mean()),
        "selfplay/decisions_per_seat": len(data["action"]) / total.size,
        "selfplay/frac_build": float(kinds[0]),
        "selfplay/frac_wonder": float(kinds[1]),
        "selfplay/frac_sell": float(kinds[2]),
    }
    for i, key in enumerate(SCORE_KEYS):
        if key != "total":
            out[f"selfplay/points_{key}"] = float(breakdowns[..., i].mean())
    return out


def main() -> None:
    args = parse_args()
    if args.low_priority:
        lower_priority("below_normal")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = PolicyValueNet(obs_dim(args.players), NUM_ACTIONS, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    start, total_games, wandb_id = 0, 0, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["iteration"] + 1
        total_games = ckpt.get("total_games", 0)
        wandb_id = ckpt.get("wandb_id")
    cfg = PPOConfig(lr=args.lr, epochs=args.epochs, minibatch=args.minibatch, ent_coef=args.ent_coef)

    # Fixed opponents for evaluation
    eval_opponents: list[tuple[str, str, tuple | None]] = [("greedy", "greedy", None), ("random", "random", None)]
    if args.eval_checkpoint:
        ref = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=False)
        ref_name = Path(args.eval_checkpoint).parent.name
        ref_weights = {k: v.numpy() for k, v in ref["model"].items()}
        eval_opponents.append((ref_name, "checkpoint", (ref["model_config"], ref_weights)))

    n_params = sum(p.numel() for p in model.parameters())
    config = {**vars(args), **model.config(), "num_params": n_params,
              "games_per_iteration": max(1, args.workers) * args.games_per_worker}
    log = Logger(args, config, wandb_id)
    wandb_id = log.wandb.id if log.wandb is not None else None
    if args.resume:
        log.line(f"resumed from {args.resume} at iteration {start}")
    log.line(f"device {device}, obs {model.obs_dim}, actions {NUM_ACTIONS}, params {n_params:,}, "
             f"workers {args.workers}, low priority {args.low_priority}")

    cpu_model = PolicyValueNet(**model.config())
    ckpt_dir = ROOT / "checkpoints" / args.run
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
            return summarize(evaluate_games(cpu_model, opponent_kind, ids, args.players, EVAL_SEED, ref_model))
        chunks = [ids[w::len(workers)] for w in range(len(workers))]
        for (conn, _), chunk in zip(workers, chunks):
            conn.send(("evaluate", (weights, opponent_kind, opponent_ckpt, chunk, EVAL_SEED)))
        return summarize(concat([conn.recv() for conn, _ in workers]))

    run_start = time.time()
    try:
        for it in range(start, args.iterations):
            if args.anneal_lr:
                lr = args.lr * (1.0 - it / args.iterations)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            t0 = time.time()
            weights = numpy_weights(model)
            seed = args.seed + it * 1_000_003
            if workers:
                for w, (conn, _) in enumerate(workers):
                    conn.send(("collect", (weights, seed + w * 10_007, args.games_per_worker)))
                data = concat([conn.recv() for conn, _ in workers])
            else:
                cpu_model.load_state_dict(model.state_dict())
                data = collect(cpu_model, args.games_per_worker, args.players, seed,
                               args.reward, args.gamma, args.lam)
            t_collect = time.time() - t0

            t1 = time.time()
            stats = ppo_update(model, optimizer, data, cfg, device)
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
            })
            metrics.update(selfplay_metrics(data, args.players))
            log.metrics(metrics, it)
            log.line(f"it {it:4d} | games {total_games:8d} | {games / t_collect:5.0f} g/s | "
                     f"score {metrics['selfplay/score_mean']:5.1f} | ent {stats['entropy']:.3f} | "
                     f"kl {stats['approx_kl']:.4f} | vloss {stats['value_loss']:.4f} | "
                     f"ev {explained_var:5.2f} | {t_collect:4.1f}s+{t_update:4.1f}s")

            last = it == args.iterations - 1
            if it % args.eval_every == 0 or last:
                t2 = time.time()
                eval_metrics, parts = {}, []
                eval_weights = numpy_weights(model)
                for name, kind, opponent_ckpt in eval_opponents:
                    res = run_eval(eval_weights, kind, opponent_ckpt)
                    eval_metrics.update({f"eval_{name}/{k}": v for k, v in res.items()})
                    parts.append(f"vs {name}: win {100 * res['win_rate']:5.1f}% "
                                 f"score {res['mean_score']:5.1f} margin {res['mean_margin']:+5.1f}")
                eval_metrics["perf/eval_sec"] = time.time() - t2
                log.metrics(eval_metrics, it)
                log.line("      eval | " + " | ".join(parts))
            if it % args.save_every == 0 or last:
                for name in ("latest.pt", f"iter_{it:05d}.pt"):
                    save(ckpt_dir / name, model, optimizer, it, total_games, args, wandb_id)
    finally:
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
