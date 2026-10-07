# 7 Wonders RL

A rules engine for 7 Wonders (2020 / 2nd edition, base game), baseline bots, and later a
self-play reinforcement-learning agent. The goal is to rank cards by how much they help you
win.

## Setup
```bash
conda create -n 7wonders python=3.12 -y
conda activate 7wonders
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas matplotlib pytest tqdm tensorboard
```
(CUDA 12.8 wheels are needed for RTX 50-series GPUs.)

## Usage
```bash
pytest -q                                                    # rules tests
python scripts/simulate.py --games 2000                      # 4 greedy bots
python scripts/simulate.py --bots greedy,random,random,random
```

## Training
```bash
python scripts/train.py --run v1 --iterations 300
tensorboard --logdir runs
```
One shared policy plays every seat (self-play). Games are played on CPU worker processes and
the PPO update runs on the GPU. Every `--eval-every` iterations the policy plays one seat
against three greedy bots and against three random bots, and logs its win rate. Checkpoints
are saved to `checkpoints/<run>/`. Use `--resume checkpoints/<run>/latest.pt` to continue a
run.

Throughput is limited by the CPU (the Python game engine), at roughly 100 games/s per core.

Useful flags:
- `--low-priority`: games and other programs get the CPU first.
- `--payment-choice`: the bot also picks how to pay for bought resources: cheapest, pay the
  left neighbour where possible, or pay the right neighbour where possible. This lets it avoid
  funding a leading neighbour even when that costs more. It changes the network's outputs, so
  these models don't mix with models trained without the flag (they can still play each other).
- `--eval-checkpoint <path>` (repeatable): also evaluate against three copies of an older model.
- `--anneal-lr`: decay the learning rate linearly to 0.
- `--no-wandb`: skip Weights & Biases logging.

What every logged metric means: [docs/METRICS.md](docs/METRICS.md).

## Layout
- `sevenwonders/cards.py`: Age cards and guilds (2nd edition)
- `sevenwonders/wonders.py`: all 14 wonder sides
- `sevenwonders/resources.py`: resources and the cheapest-payment solver
- `sevenwonders/game.py`: game state, legal actions, turn flow, scoring
- `sevenwonders/bots.py`: random and greedy baseline bots
- `sevenwonders/runner.py`: play full games, optionally logging every decision
- `sevenwonders/rl/encoding.py`: game state → observation vector + legal-action mask
- `sevenwonders/rl/model.py`: policy/value MLP
- `sevenwonders/rl/rollout.py`: batched self-play collection, rewards, GAE
- `sevenwonders/rl/ppo.py`: PPO update
- `sevenwonders/rl/agent.py`: `PolicyBot` and evaluation against baseline bots
- `scripts/train.py`: training entry point
- `DATA_NOTES.md`: data sources, uncertain entries, rule interpretations

## Roadmap
1. ~~Rules engine, card data, tests~~
2. ~~Baseline bots~~
3. ~~RL environment (observation encoding, action masking) and PPO self-play on the GPU~~ (v1 done; tuning ongoing)
4. Card analysis: pick rates, picked-vs-passed win rates, the value network's preferences, forced-pick experiments
