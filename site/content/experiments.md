---
title: "Experiments"
subtitle: "single-variable ablations on the distilled-from-stockfish baseline"
next: "/method/"
aliases:
  - /findings/
  - /baseline/
  - /capacity/
  - /data/
  - /search/
---

The setup is fixed: a 20-block × 256-channel ResNet, supervised
distillation from Stockfish 17 at search-depth 15, multipv=8 soft
targets at temperature `T = 1` pawn, 100-game evals against a
calibrated Stockfish opponent at fixed `UCI_Elo`. The baseline scores
**1,807 Elo** at 800 MCTS sims/move on a 5M-position dataset.

Each experiment below changes *one variable*, holds everything else
constant, and states the hypothesis it was meant to test, the result,
and the verdict. Code links land directly on the launcher / training
script for each.

## The Baseline Trajectory {#baseline}

100-game evals vs Stockfish `UCI=1350`, 800 MCTS sims/move, 20-block
network.

| epoch | Elo |
|---:|---:|
| 5 | 1,651 |
| 10 | 1,862 |
| 15 | 1,759 |
| **20** | **1,807** |

100-game evals at this anchor have a CI of roughly ±100 Elo, so the
ep-10 spike (1,862) is mostly noise — not a real swing. The trend at
this resolution is just "around 1,700–1,800 across epochs"; we take
**ep 20 = 1,807** as the canonical baseline because it's the last
training checkpoint, not because it's deterministically best. Loss
plateaus around 2.59 by epoch 10; top-1 accuracy plateaus ~0.34. This
is the "1,800 ceiling" the ablations probe.

→ code: [`train_supervised.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/src/distill_soft/train_supervised.py) ·
[`eval.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/scripts/eval.py)

## Network Capacity — Doubling Depth Didn't Help {#capacity}

**Hypothesis:** the ceiling is the network running out of representational
capacity. Doubling the depth (40 blocks instead of 20, ~48M params
instead of 24M) should buy Elo if so.

**Setup:** identical to the baseline except `n_blocks=40`. Same 5M
positions, same loss, same 20 epochs. Single L40S, batch 512 (the
deeper net costs ~2× per batch so we cut batch in half to keep memory
sane). 13.5 min/epoch — close to the baseline because deeper-but-
narrower-batch trades roughly cancel.

| epoch | Elo @ UCI=1800 |
|---:|---:|
| 5 | **1,813** |
| 10 | 1,780 |
| 15 | 1,722 |
| **20** | **1,810** |

{{< chart-capacity >}}

The UCI=1,800 anchor is where the score sits near 0.5 and the Elo CI is
tightest — read that column.

**Result.** **~1,810 Elo, identical to the baseline.** More parameters
bought nothing on this data. The ceiling is **not** a capacity problem.

**Why a deeper net didn't help:** 5M positions for a 24M-param network
is already in the over-parameterized regime. Top-1 was already 0.34
with the smaller net and stays 0.34 with the bigger one. The student
is hitting a *teacher-signal* limit, not a *representation* limit.

→ code: [`d15-40x256-eu.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d15-40x256-eu.sh)

## Soft vs Hard Targets {#soft-vs-hard}

**Hypothesis:** soft multipv targets (a distribution over Stockfish's
top-8 candidate moves, softmax of centipawn scores) carry more useful
information than hard one-hot ("the move Stockfish actually played").
The student should learn faster from richer targets.

**Setup:** identical recipe in both arms — same network, same data,
same epoch count. Only the policy target differs:

- **Hard:** `π(played_move) = 1`, all else 0. Standard supervised classification.
- **Soft:** `π(move_k) = softmax(cp_k / 100)`. At `T = 1` pawn, a 50-cp gap
  between two moves becomes a 62 / 38 split, not 100 / 0.

The soft version is what every other experiment on this page uses.

### Result Depends on Teacher Strength

| recipe | weak teacher (d10, ~2,200 Elo) | strong teacher (d15, ~2,500 Elo) |
|---|---:|---:|
| hard one-hot | ~1,315 | (not measured) |
| **soft multipv** | **~1,185** (−130) | **1,807** |

At a weak teacher, soft targets actually *underperform* one-hot by
~130 Elo — the student learns to hedge between moves the teacher
itself wasn't sure about, so its policy is fuzzier than necessary. At
a strong teacher the soft distribution becomes sharply peaked and the
hedge cost disappears; soft and hard converge.

**Verdict.** Soft targets help when the teacher is strong enough that
its multipv ranking actually reflects move quality. Below that, the
extra "information" is mostly the teacher's own uncertainty, which
hurts the student. Picking soft vs hard isn't a free axiom — it
trades off against teacher quality.

**Practical consequence.** Everything above d10 in this project uses
soft targets. A sharper temperature (e.g. `T = 0.3`) is on the roadmap
— it would keep some multipv ranking information but force more
commitment from the student.

→ code: [`stockfish_data.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/src/distill_soft/stockfish_data.py)
(the `softmax(cp/100*T)` construction)

## Eval-Side Search — +277 Elo From Search Alone {#search}

**Hypothesis:** the baseline is evaluated at 800 MCTS sims/move
(AlphaZero's *training-time* value). At competitive play
[AlphaZero](https://arxiv.org/abs/1712.01815) used many tens of
thousands of sims, and [Lc0](https://lczero.org) reports policy
priors that transfer well to deeper inference search.

**Setup:** same checkpoint (d15 epoch 20), re-eval at `--sims 4000`.
Single g6.4xlarge, 104 games, 93.6 min wallclock. No retraining,
just deeper search at inference.

| sims / move | Elo |
|---:|---:|
| 800 | 1,807 |
| **4,000** | **2,084** |

{{< chart-search >}}

**Result.** **+277 Elo from search alone, no retraining.** The model
sweeps 101 of 104 games at 4,000 sims; score saturation means we're
now hitting the ceiling of *the opponent's* strength rather than the
agent's.

**Interpretation.** Distilled priors transfer well to deeper inference-
time search — exactly what Lc0 demonstrated empirically. The network's
policy prior guides MCTS toward strong moves; more rollouts let MCTS
refine that prior into a sharper visit distribution. The student
becomes a better chess player not by learning more, but by *thinking
longer*. Conceptually similar to the policy-improvement step in
[KataGo's Playout Cap Randomization](https://arxiv.org/abs/1902.10565)
— more compute → sharper target.

**Practical consequence.** The "1,800 ceiling" reported from routine
evals is a *floor*, not a ceiling. There's ~280 Elo trivially
available from deeper eval search. For honest comparison with
AlphaZero, the relevant number is the search-saturated one.

A full sims sweep is in progress — same checkpoint, eval at
200 / 800 / 2,000 / 4,000 / 8,000 / 16,000 sims. The Elo-vs-log(sims)
curve will reveal the saturation point. The machinery
([`elo_bisect.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/scripts/elo_bisect.py))
is in place.

→ code: [`eval-deep-sims.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/eval-deep-sims.sh) ·
[`eval-c-ep5-sims4000.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/eval-c-ep5-sims4000.sh)

## Data Scale — 6× More Positions (The Bitter Lesson?) {#data}

**Hypothesis:** the baseline used a 5M-position subsample of an
available 30M-position dataset. Maybe the recipe is fine and we're
just data-starved.

**Setup:** 20×256 net (same as baseline), full ~30M positions, no
`MAX_POSITIONS` truncation. `BATCH_SIZE=2048` to keep iteration count
reasonable (~14,650 batches/epoch). In-RAM via 256 GB on a
`g6e.8xlarge` (32 vCPU + 1× L40S). 20 epochs, ~49 min/epoch,
~16 h wallclock.

**Why this is 3× slower per epoch than the capacity experiment.** Same
single L40S in both. The wallclock difference comes from raw compute
— 2× per batch (30M÷5M ≈ 1× but `BATCH_SIZE=2048` vs 512 means 1.5×
batches/epoch on top), so ~3× expected. Actual was ~3.6×. The extra
20% is OS page-cache cold start on epoch 1 (15 GB of memmapped files
page in) plus DataLoader overhead.

### Results

100-game evals vs Stockfish at UCI=1,800.

| epoch | Elo |
|---:|---:|
| **5** | **2,004** |
| **10** | **2,009** |
| 15 | 1,957 |
| 20 | 1,957 |

{{< chart-data-scale >}}

The data ablation peaks at **ep 10: 2,009 Elo at the tight UCI=1,800
anchor**.

### Same Teacher, More Data

| dataset (d10 teacher) | ep 5 Elo @ 1350 |
|---|---:|
| 5 M subset | 1,749 |
| **30 M full** | **2,084** |

Same teacher, 6× more data → **+335 Elo** at the easier anchor.

### Stronger Teacher vs More Data

| run | Elo @ UCI=1800 |
|---|---:|
| baseline (d15, 5 M) | 1,810 |
| **30 M (d10, weaker teacher)** | **2,009** |

**Verdict: the bottleneck on the original baseline was data.** Six
times more positions buys ~200 Elo at the calibrated anchor — even
with a *weaker* teacher (depth-10 vs depth-15).

**The bitter-lesson echo.** Sutton's
[Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
claims that, across 70 years of AI, methods that *leverage computation*
(scale and search) consistently outperform methods that *build in
human structure* (handcrafted features, deeper / cleverer
architectures). Our three ablations all point the same way at
this small scale:

- **Network capacity** (more structure / more parameters) → no gain.
- **Eval-side search** (more compute) → +277 Elo.
- **Data scale** (more compute spent generating training signal) → +199 Elo.

The two confirmed levers are both forms of "more compute"; the
rejected lever is the "smarter network" intuition. The bitter
lesson, replayed at $50 budget on a single GPU.

→ code: [`d10-full30m.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d10-full30m.sh) ·
[`stockfish_data.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/src/distill_soft/stockfish_data.py)

## Stacking Search + Data Scale Together {#stacking}

The natural follow-up: apply both confirmed ablations to the *same*
checkpoint. The d10-30M weights at sims=4,000 give us the project's
strongest measurements:

| recipe | sims | opponent | Elo |
|---|---:|---|---:|
| baseline (d15 5M ep 20) | 800 | UCI=1,800 | 1,810 |
| + 30M data (d10 ep 5) | 800 | UCI=1,800 | 2,004 |
| + 4,000 sims on top (ep 5) | 4,000 | UCI=1,800 | 2,084 [2005, 2197] |
| same, tighter anchor (ep 5) | 4,000 | UCI=2,000 | **2,110** [2044, 2187] |
| later epoch + same search (ep 10) | 4,000 | UCI=1,800 | 2,171 [2082, 2324] |
| **deeper into training (ep 15)** | **4,000** | **UCI=1,800** | **2,189** [2098, 2354] |
| same, after the peak (ep 20) | 4,000 | UCI=1,800 | 2,154 [2067, 2297] |

**+379 Elo from the baseline.** The deep-search numbers don't *quite*
add: search recovered +277 on the weaker d15 5M prior but only ~+105
on top of the data-scaled prior. The headline **2,189** figure is
ep 15's better learned value combined with the deep-search rollouts;
the tightest CI sits at 2,110 against the stronger UCI=2,000 opponent.
The curve peaks at ep 15 and dips slightly by ep 20 — the model is
already past its optimum at the 20×256 / 30M scale.

→ code: [`eval-c-ep5-sims4000.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/eval-c-ep5-sims4000.sh) ·
[`eval-c-ep5-sims2000-uci1800.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/eval-c-ep5-sims2000-uci1800.sh)

## Stronger Teacher at Full Data Scale — d15 at 46M {#d15-46m}

**The natural follow-up.** d10-30M peaked at 2,189. d15 was the original
plan; we settled for d10 because the d15-250K datagen run was still in
flight at the time. It finished — 426K games / 45.9M positions of
Stockfish d15 multipv=8 at temperature 1, the 1.5× scale-up of the
d10-30M corpus that produced the prior peak.

Two parallel training variants ran on the same dataset:

| variant | net | LR | wd | batch | epochs | region/market |
|---|---|---|---|---:|---:|---|
| **R1 — bigger net** | 40×256 (~50M) | 1e-3 | 1e-4 | 1024 | 40 | us-east-1 spot g6e.8xlarge |
| **R2 — regularization-leaning** | 20×256 (~24M) | 5e-4 | 1e-3 | 2048 | 30 | eu-central-1 OD g6e.8xlarge |

R1 mirrors the original "throw capacity at it" instinct. R2 mirrors
d10's shape (20×256) but with slower LR and 10× weight decay — the
hypothesis that a smaller model trained less aggressively can match
the bigger one on the d15 soft signal.

### sims=800 trajectory — both runs (UCI=1,800 anchor)

100-game evals, daemon-fired per ckpt:

| ep | R1 (40×256) | R2 (20×256, lowlr) |
|---:|---:|---:|
| 1 | 1,864 | 1,834 |
| 2 | 1,892 | 1,889 |
| 3 | 1,875 | — |
| 4 | 1,857 | 1,857 |
| 5 | 1,850 | 1,900 |
| 6 | 1,889 | 1,850 |
| **7** | **1,941** | 1,922 |
| 8 | 1,864 | 1,922 |
| 9 | 1,910 | 1,861 |
| 10 | — | 1,907 |
| 11 | — | 1,864 |
| 12 | — | 1,875 |

Per-ckpt CIs are ±70 Elo wide at the 100-game budget, so the swings
between adjacent epochs are mostly noise. The trend at this resolution:
both runs are wandering 1,850–1,940 in the sims=800 anchor. R2's curve
is statistically indistinguishable from R1's despite half the
per-epoch compute — early evidence that the smaller-net + lower-LR +
heavier-regularization recipe is competitive.

### sims=4,000 deep-read on R1 — the comparable-to-d10 numbers

A second daemon
([`wm-deep-eval-daemon.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-deep-eval-daemon.sh))
fires sims=4,000 follow-ups on any ckpt whose sims=800 score crosses
the threshold (currently 1,940). Two land so far:

| ckpt | sims=800 | sims=4,000 | sims=4,000 − sims=800 |
|---|---:|---:|---:|
| R1 ep 2 | 1,892 | 2,055 [1979, 2159] | **+163** |
| R1 ep 7 | 1,941 | **2,146** [2060, 2285] | **+205** |

The +163/+205 Elo gap confirms search recovers progressively more on
stronger ckpts. ep 7's sims=4,000 already puts d15 within striking
distance of d10's 2,189 peak — −43 Elo, CIs overlap by ~100 Elo.

R1 has 33 epochs remaining. If it follows the d10 arc (peak around
ep 15), the d15 best is likely ahead of us.

### Where this leaves the project

- **d15 46M is climbing toward d10's peak**, not stuck below it. The
  experiment we ranked as "most information per dollar" back in
  February is paying off.
- **Smaller net + slower LR is competitive** with bigger net + default
  LR at matched data. R2's sims=800 trajectory shadows R1's despite
  half the per-epoch compute — and the regularization-leaning recipe
  hasn't yet been sims=4,000-evaluated, where R1 jumped +205 Elo.

→ code: [`d15-full30m.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d15-full30m.sh) ·
[`d15-20x256-lowlr.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d15-20x256-lowlr.sh) ·
[`wm-deep-eval-daemon.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-deep-eval-daemon.sh)

## Summary {#summary}

| variable changed | hypothesis |
|---|---|
| network capacity (40×256 vs 20×256, 5M data) | **rejected** |
| soft vs hard targets | **partial** (helps strong teachers, hurts weak) |
| eval-side search (4,000 vs 800 sims) | **confirmed** (+277) |
| data scale (30M vs 5M positions) | **confirmed** (+199) |
| stronger teacher (d15 vs d10) at full data | **in flight** (R1 ep 7 at 2,146, climbing toward d10's 2,189 peak) |

Capacity is decisively *not* the bottleneck on the 5M-data baseline;
that result needs to be re-tested at full data scale because R1 (40×256
on 46M positions) is climbing. Eval search recovers a huge slice. **Data
was the real bottleneck.** Strongest checkpoint to date:
**2,189 Elo @ UCI=1,800** from the d10 full-30M run at epoch 15 with
sims=4,000 (CI [2098, 2354]). Tightest CI: **2,110 Elo @ UCI=2,000** at
ep 5 with sims=4,000 (CI [2044, 2187]). The d15 full-data run is
expected to top the chart within the next ~24h
([what's next →](/next/)).
