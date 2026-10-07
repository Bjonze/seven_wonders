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

## Layout
- `sevenwonders/cards.py`: Age cards and guilds (2nd edition)
- `sevenwonders/wonders.py`: all 14 wonder sides
- `sevenwonders/resources.py`: resources and the cheapest-payment solver
- `sevenwonders/game.py`: game state, legal actions, turn flow, scoring
- `sevenwonders/bots.py`: random and greedy baseline bots
- `sevenwonders/runner.py`: play full games, optionally logging every decision
- `DATA_NOTES.md`: data sources, uncertain entries, rule interpretations

## Roadmap
1. ~~Rules engine, card data, tests~~
2. ~~Baseline bots~~
3. RL environment (observation encoding, action masking) and PPO self-play on the GPU
4. Card analysis: pick rates, picked-vs-passed win rates, the value network's preferences, forced-pick experiments
