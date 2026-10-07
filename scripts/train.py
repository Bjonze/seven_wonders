"""PPO self-play training.

    python scripts/train.py --run baseline --iterations 500

Rollouts run on CPU worker processes (one shared policy plays every seat); the PPO update
runs on the GPU. Progress goes to the console and to TensorBoard (runs/<run>), checkpoints
to checkpoints/<run>/.
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

from sevenwonders.rl.agent import evaluate  # noqa: E402
from sevenwonders.rl.encoding import NUM_ACTIONS, obs_dim  # noqa: E402
from sevenwonders.rl.model import PolicyValueNet  # noqa: E402
from sevenwonders.rl.ppo import PPOConfig, ppo_update  # noqa: E402
from sevenwonders.rl.rollout import collect, worker_main  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="baseline", help="run name (logs and checkpoints)")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="rollout processes (default: CPU count - 1); 0 = collect in the main process")
    p.add_argument("--games-per-worker", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--reward", choices=["win", "rank"], default="win")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    return p.parse_args()


def numpy_weights(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def save(path: Path, model, optimizer, iteration: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.config(),
        "iteration": iteration,
        "num_players": args.players,
        "args": vars(args),
    }, path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = PolicyValueNet(obs_dim(args.players), NUM_ACTIONS, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["iteration"] + 1
        print(f"resumed from {args.resume} at iteration {start}")
    cfg = PPOConfig(lr=args.lr, epochs=args.epochs, minibatch=args.minibatch, ent_coef=args.ent_coef)

    cpu_model = PolicyValueNet(**model.config())
    ckpt_dir = ROOT / "checkpoints" / args.run
    writer = SummaryWriter(str(ROOT / "runs" / args.run))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device}, obs {model.obs_dim}, actions {NUM_ACTIONS}, params {n_params:,}")

    ctx = mp.get_context("spawn")
    workers = []
    for _ in range(args.workers):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=worker_main, daemon=True,
                           args=(child, args.players, model.config(), args.reward, args.gamma, args.lam))
        proc.start()
        workers.append((parent, proc))

    total_games = 0
    try:
        for it in range(start, args.iterations):
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
            explained_var = 1.0 - np.var(ret - val) / max(np.var(ret), 1e-8)
            stats.update({
                "games_per_sec": games / t_collect,
                "samples": float(len(data["action"])),
                "mean_score": float(data["scores"].mean()),
                "explained_var": float(explained_var),
                "collect_sec": t_collect,
                "update_sec": t_update,
            })
            for k, v in stats.items():
                writer.add_scalar(f"train/{k}", v, it)
            print(f"it {it:4d} | games {total_games:7d} | {stats['games_per_sec']:5.0f} g/s | "
                  f"score {stats['mean_score']:5.1f} | ent {stats['entropy']:.3f} | "
                  f"kl {stats['approx_kl']:.4f} | vloss {stats['value_loss']:.4f} | "
                  f"ev {explained_var:5.2f} | {t_collect:4.1f}s+{t_update:4.1f}s", flush=True)

            last = it == args.iterations - 1
            if it % args.eval_every == 0 or last:
                cpu_model.load_state_dict(model.state_dict())
                line = []
                for opponent in ("greedy", "random"):
                    res = evaluate(cpu_model, opponent, games=args.eval_games, num_players=args.players)
                    for k, v in res.items():
                        writer.add_scalar(f"eval_{opponent}/{k}", v, it)
                    line.append(f"vs {opponent}: win {100 * res['win_rate']:5.1f}% "
                                f"score {res['mean_score']:5.1f} margin {res['mean_margin']:+5.1f}")
                print("      eval | " + " | ".join(line), flush=True)
            if it % args.save_every == 0 or last:
                save(ckpt_dir / "latest.pt", model, optimizer, it, args)
                save(ckpt_dir / f"iter_{it:05d}.pt", model, optimizer, it, args)
    finally:
        for conn, proc in workers:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for _, proc in workers:
            proc.join(timeout=5)
        writer.close()


if __name__ == "__main__":
    main()
