# 7 Wonders RL

A rules engine for 7 Wonders (2020 / 2nd edition, base game), baseline bots, and a
self-play reinforcement-learning agent. The goal is to rank cards by how much they help you
win.

## Results
**[Card rankings (current, bot v5a)](results/v5a/README.md)**: every card's value per Age
(overall and in the first vs. second half of the Age), checked against played-out games, plus
which wonder side the bot picks and how each side does.

Earlier: [rankings with bot v3](results/v3/README.md) (tag `baseline-v3`). v3 could see the
face-down discard pile, so prefer the v5a rankings.

| Bot | What changed | Games trained | Win rate vs. 3 greedy bots | vs. 3 copies of the previous best |
|---|---|---:|---:|---:|
| v1 | first PPO self-play bot | 0.19M | 73% | |
| v2 | longer training | 1.4M | 81% | 48% (v1) |
| v3 | chooses whom to pay for resources | 1.4M | 84% | 24% (v2) |
| v4 | discard pile hidden, 8.5M-parameter network, opponent pool | 5.6M | 87% | 27% (v3)* |
| **v5a** | picks its wonder side; fine-tuned from v4 | +2.1M | 89% | 29% (v4) |
| v5b | as v5a + 15% repeated-wonder games | +2.1M | 88% | 28% (v4) |

4-player games; 25% = equal strength. *v3 still sees the discard pile in that comparison,
an advantage v4 doesn't have. v5a and v5b are equally strong (25% / 26% against each other
over 2,000 deals).

**Ensembles** (`scripts/ensemble_eval.py`, 2,000 deals each, ±2 points): averaging several
models' move probabilities helps only slightly. Against 3× v4, v4+v5a+v5b wins 31.1% vs.
28.3% for v5a alone (+2.8 ± 2.1 on the same deals); against 3× v5a every ensemble stays within
+1 point of 25%. Adding v2 and v3 (given the legal discard view) doesn't help further.

## Setup
```bash
conda create -n 7wonders python=3.12 -y
conda activate 7wonders
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas matplotlib pytest tqdm tensorboard wandb
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

## Card rankings
```bash
python scripts/card_rankings.py --checkpoint checkpoints/v3/latest.pt --name v3 --low-priority
```
Writes `results/<name>/` (report, CSV, figures). Raw simulation data goes to
`results/<name>/raw/` (not committed); `--report-only` rebuilds the report from it.

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
- `sevenwonders/rl/analysis.py`: card-value measurements (value-network lookahead, build-vs-sell games)
- `scripts/train.py`: training entry point
- `scripts/card_rankings.py`: card-ranking report
- `DATA_NOTES.md`: data sources, uncertain entries, rule interpretations

## Roadmap
1. ~~Rules engine, card data, tests~~
2. ~~Baseline bots~~
3. ~~RL environment (observation encoding, action masking) and PPO self-play on the GPU~~
4. ~~Card analysis: pick rates, picked-vs-passed win rates, value-network estimates, build-vs-sell games~~ (baseline with v3)
5. Mastering the game: a much stronger bot, then re-run the card rankings
