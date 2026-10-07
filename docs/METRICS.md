# Training metrics (wandb / TensorBoard)

The x-axis (`step`) is the training **iteration**. One iteration = every worker plays
`games-per-worker` self-play games (11 × 128 = 1,408 games by default), then one PPO update.
To plot against games played instead, choose `progress/total_games` as the x-axis in wandb.

## `eval_<opponent>/*`: how strong is the bot? (every `--eval-every` iterations)
4-player games: the current policy plays **one** seat and the other **three** seats are
`<opponent>`:

| Group | The three opponents |
|---|---|
| `eval_greedy` | the hand-written greedy bot (`GreedyBot`) |
| `eval_random` | bots that pick a random legal action |
| `eval_v1`, `eval_v2`, ... | three copies of an older trained checkpoint (`--eval-checkpoint`) |

- The policy's seat rotates every game, and wonders are dealt at random. The **same 400 deals**
  are reused at every evaluation, so the curves can be compared point to point.
- During evaluation the policy always plays its single most likely action (no randomness).
  During training it samples.

| Metric | Meaning |
|---|---|
| `win_rate` | Share of games the policy won (ties split). **With 4 players, 25% = as good as the opponents**; above that it's stronger. |
| `mean_score` | Policy's average final score. |
| `mean_margin` | Policy's score minus the **best** opponent's score, averaged. Positive = usually finishes ahead of everyone. |
| `points_military`, `points_science`, ... | Policy's average points per scoring category: military, treasure (coins), wonder, civil (blue), commerce (yellow), guilds (purple), science (green). Shows *how* it wins. |

`eval_<checkpoint>/win_rate` is the most useful long-run signal: once the bot crushes greedy and
random bots, beating its own older versions shows it's still improving.

## `selfplay/*`: what the training games look like
All four seats are the current policy. Averaged over all seats and games of the iteration.

| Metric | Meaning |
|---|---|
| `score_mean` | Average final score. It can *drop* while the bot gets stronger (e.g. more military conflict means more lost battles). |
| `winner_score_mean` | Average score of the winning player. |
| `points_<category>` | Average points per category; shows the strategy mix shifting. |
| `frac_build`, `frac_wonder`, `frac_sell` | Share of decisions that build a card / build a wonder stage / sell a card for 3 coins. |
| `decisions_per_seat` | Decisions per player per game: 18, plus extras for Babylon's 7th card and Halikarnassos' discard builds. |
| `trade_coins_per_player` | Coins a player pays its neighbours for resources over a game. |
| `frac_pay_left`, `frac_pay_right` | (payment-choice runs) Share of builds/wonder stages paid with the non-cheapest "pay left neighbour" / "pay right neighbour" option. |

## `train/*`: is the learning healthy?
| Metric | Meaning | Healthy range |
|---|---|---|
| `entropy` | How random the policy is. Starts near 2, falls as the bot commits to choices. | Slowly falling. A crash to ~0 early means it stopped exploring. |
| `approx_kl` | How much one update changed the policy. | ~0.005-0.03. Above 0.045 the update stops early. |
| `clip_frac` | Share of samples where PPO's step-size limit kicked in. | ~0.05-0.3 |
| `updates` | Gradient steps taken this iteration. Fewer than usual means the KL early stop triggered. | |
| `value_loss` | Error of the value head, which predicts the chance of winning from each position. | Should fall then plateau. |
| `explained_var` | How much of the game outcome the value head predicts (0 = no better than guessing the average, 1 = perfect). | Card games have luck and hidden hands, so 0.3-0.5 is good. |
| `policy_loss` | PPO's surrogate objective. Its absolute value isn't meaningful; watch the others. | |
| `lr` | Learning rate (decays linearly to 0 with `--anneal-lr`). | |

## `perf/*` and `progress/*`
| Metric | Meaning |
|---|---|
| `perf/games_per_sec` | Self-play throughput (drops while you game with `--low-priority`). |
| `perf/collect_sec`, `perf/update_sec`, `perf/eval_sec` | Time per iteration spent playing games, updating the network (GPU), and evaluating. |
| `perf/samples` | Decisions collected this iteration (training examples). |
| `progress/total_games`, `progress/hours` | Running totals. |
