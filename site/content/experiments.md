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
| 10 | 1,882 | 1,907 |
| 11 | 1,885 | 1,864 |
| 12 | — | 1,875 |
| **13** | — | **1,925** |

Per-ckpt CIs are ±70 Elo wide at the 100-game budget, so the swings
between adjacent epochs are mostly noise. The trend at this resolution:
both runs are wandering 1,850–1,940 in the sims=800 anchor. R2's
curve is statistically indistinguishable from R1's despite half the
per-epoch compute — R2 ep 13 = 1,925 is within 16 Elo of R1's best so
far (ep 7 = 1,941) on the noisier sims=800 anchor. Early evidence
that the smaller-net + lower-LR + heavier-regularization recipe is
competitive.

### sims=4,000 deep-read — the comparable-to-d10 numbers

A second daemon
([`wm-deep-eval-daemon.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-deep-eval-daemon.sh))
fires sims=4,000 follow-ups on any ckpt whose sims=800 score crosses
threshold (started at 1,940; dropped to 1,920 to catch R2 ckpts).

| ckpt | sims=800 | sims=4,000 | Δ |
|---|---:|---:|---:|
| R1 ep 2 | 1,892 | 2,055 [1979, 2159] | +163 |
| R1 ep 7 | 1,941 | **2,146** [2060, 2285] | **+205** ← R1 best |
| R2 ep 6 | 1,922 | **2,123** [2041, 2252] | +201 |
| R2 ep 7 | 1,922 | 2,109 [2028, 2232] | +187 |
| R2 ep 12 | 1,925 | 2,029 [1956, 2126] | +104 |

The +160 to +200 Elo gap between sims=800 and sims=4,000 confirms
search recovers real strength that sims=800 systematically
underestimates. Surprise: R2 ep 12 (highest sims=800 result for R2)
came in *lower* at sims=4,000 than ep 6/7 — the sims=800 spike was
noise, not a real peak.

### R2 final outcome (30 epochs, finished 2026-05-25 01:45 UTC)

R2 completed all 30 epochs cleanly. Per-ckpt sims=800 evals drained
through ep 28 (ep 29's final is pending). The trajectory across
epochs 13–28 stays in the 1,860–1,925 band — no upward drift after
the early-epoch climb. R2 saturated around ep 6 in true-strength
terms (sims=4,000 = 2,123), with later epochs trading noise without
gain.

### Head-to-head — d10 ep 15 vs R1 ep 7 at sims=4,000

To check whether the +43 Elo Stockfish-anchored "gap" between
d10's 2,189 and R1's 2,146 was real, we put the two networks in
direct play with both using MCTS at sims=4,000. 104 games,
alternating colors:

| | result |
|---|---|
| d10 ep 15 wins | **0** |
| draws | **104** |
| R1 ep 7 wins | **0** |
| Elo gap | **0** |
| Score CI | [0.404, 0.596] |

Every single game drew. The Stockfish-anchored "gap" was noise.
**At sims=4,000, d10's top ckpt and R1's top ckpt are
indistinguishable.** The d15 teacher didn't give us strength
beyond d10's ceiling — but it also didn't underperform.

→ code: [`h2h-d10-vs-d15.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/h2h-d10-vs-d15.sh) ·
[`h2h_mp.py`](https://github.com/shehio/world-models/blob/main/experiments/selfplay/scripts/h2h_mp.py)

### Cosine-LR follow-on — R1 v2 and R2 v2

Two follow-on variants test whether the constant-LR plateau was the
binding bottleneck. Both run with linear warmup (3 epochs) → cosine
decay to 1e-5 on the same 45.9M-position d15 corpus:

| Variant | Net | Region / market | Started | Target epochs |
|---|---|---|---|---:|
| **R1 v2** (cosine 40×256) | 40×256 (~50M) | `eu-central-1` OD g6e.8xlarge `i-0a6c44e043b2d241e` | 2026-05-25 04:00 | 40 |
| **R2 v2** (cosine 20×256) | 20×256 (~24M) | `ap-northeast-1` spot g6e.8xlarge `i-008f0d7d3a15974b9` | 2026-05-25 09:35 | 30 |

R2 v2 moved to ap-northeast-1 (Tokyo) on spot — us-east-1's G OD quota
was saturated by deep-eval boxes, and Tokyo's spot market had headroom
the home regions didn't.

### sims=4,000 deep-reads — R1 v2 and R2 v2 vs the d10 peak

The autoeval daemon's sims=4,000 follow-up fires on any ckpt whose
sims=800 score crosses threshold. Current results, sorted by point
estimate:

| ckpt | sims=800 | sims=4,000 | Δ |
|---|---:|---:|---:|
| **R2 v2 ep 4** | (skipped — daemon offline) | **2,285** [2177, 2554] | — ← project high |
| **R1 v2 ep 7** | 1,978 | **2,209** [2115, 2389] | +231 |
| R1 v2 ep 2 | 1,961 | 2,138 [2054, 2274] | +177 |
| R2 v2 ep 10 | 1,937 | 2,090 [2011, 2205] | +153 |
| R2 v2 ep 6 | 1,922 | 2,078 [2000, 2189] | +156 |
| R2 v2 ep 9 | 1,861 | 2,024 [1951, 2119] | +163 |

**R2 v2 ep 4 is the highest point estimate measured in the project to
date.** But its 95% CI is ±200 Elo (single 104-game match), so the
ordering against d10 ep 15 (2,189 [2098, 2354]) is *not* statistically
significant. R1 v2 ep 7 at 2,209 is tighter and lands just above the
d10 peak.

### Verdict so far

**The cosine LR schedule is working** — both R1 v2 and R2 v2 produce
ckpts that meet or beat the d10-30M peak (2,189) at sims=4,000. The
constant-LR plateau in R1 + R2 was a recipe artifact, not a teacher /
data ceiling.

The relative ordering between R2 v2 ep 4 / R1 v2 ep 7 / d10 ep 15 is
inside eval noise at 100 games. A 200-game head-to-head among the
three (no Stockfish anchor) would settle it — see
[`h2h-d10-vs-d15.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/h2h-d10-vs-d15.sh)
for the launcher template.

Open caveat: **the autoeval daemon runs on the operator's laptop**.
As of 2026-05-26 16:00 UTC the laptop is offline, so R1 v2 ckpts ≥ ep 13
and R2 v2 ckpts ≥ ep 18 are still unevaluated. Training continues; eval
catches up when the daemon comes back.

→ code: [`d15-full30m.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d15-full30m.sh) ·
[`d15-20x256-lowlr.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d15-20x256-lowlr.sh) ·
[`wm-deep-eval-daemon.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-deep-eval-daemon.sh)

## Peak Elo Across Every Checkpoint We've Ever Run {#peak-elo}

The full top-8 across the entire project, sorted by point estimate.
All UCI=1,800 anchor, ~100 games each.

| # | Run | Ckpt | sims | Elo | CI |
|---|---|---|---:|---:|---|
| 1 | **R2 v2 (d15 46M, 20×256 cosine)** | **ep 4** | **4,000** | **2,285** | **[2177, 2554]** ← project high |
| 2 | **R1 v2 (d15 46M, 40×256 cosine)** | **ep 7** | **4,000** | **2,209** | **[2115, 2389]** |
| 3 | d10 30M (20×256) | ep 15 | 4,000 | 2,189 | [2098, 2354] |
| 4 | d10 30M (20×256) | ep 10 | 4,000 | 2,171 | [2082, 2324] |
| 5 | d10 30M (20×256) | ep 20 | 4,000 | 2,154 | [2067, 2297] |
| 6 | d15 46M (40×256 constant LR — R1) | ep 7 | 4,000 | 2,146 | [2060, 2285] |
| 7 | R1 v2 (d15 46M, 40×256 cosine) | ep 2 | 4,000 | 2,138 | [2054, 2274] |
| 8 | d15 46M (20×256 constant LR — R2) | ep 6 | 4,000 | 2,123 | [2041, 2252] |

The top three numbers are within ~96 Elo and their CIs all overlap.
At 100 games per measurement the score CI is ±10 percentage points,
which converts to a wide Elo CI when the agent is far from the
opponent. A direct head-to-head between the three top ckpts (rows 1, 2,
3) would put a much tighter bound on the ordering than the
Stockfish-anchored numbers can.

## Did the stronger teacher (d15) finally beat d10? {#d10-vs-d15}

The earlier write-up of this section asked "*why is d10 ahead of d15?*"
because R1 (constant LR) and R2 (constant LR) both topped out at or
below the d10 peak. The cosine-LR re-runs flipped that ordering:

| | best sims=4,000 Elo @ UCI=1,800 | CI |
|---|---:|---|
| d10 30M (20×256, LR=1e-3 constant) | 2,189 (ep 15) | [2098, 2354] |
| d15 46M constant LR — R1 (40×256) | 2,146 (ep 7) | [2060, 2285] |
| d15 46M constant LR — R2 (20×256) | 2,123 (ep 6) | [2041, 2252] |
| **d15 46M cosine LR — R1 v2** (40×256) | **2,209 (ep 7)** | **[2115, 2389]** |
| **d15 46M cosine LR — R2 v2** (20×256) | **2,285 (ep 4)** | **[2177, 2554]** |

**The cosine schedule unlocked the d15 teacher.** Both cosine variants
beat their constant-LR siblings (R1 v2 +63 over R1; R2 v2 +162 over
R2) and both meet or exceed the d10 30M peak. The "d15 ≈ d10" verdict
from the first round was a recipe artifact — constant LR=1e-3 was
oscillating instead of converging.

The cosine numbers are still a 100-game-eval-noise away from being
statistically distinguishable: R2 v2's 2,285 has a 95% CI of ±200 Elo.
A 200-game head-to-head between the three (d10 ep 15 vs R1 v2 ep 7 vs
R2 v2 ep 4, MCTS sims=4,000 on all sides) is the only way to pin the
real ordering. The launcher exists — see below.

### What's still uncertain

- **R2 v2 ep 4 might be a noise spike.** Later R2 v2 ckpts (ep 6, 9,
  10) all came in lower at sims=4,000. That's consistent either with
  "ep 4 was a lucky 104-game match" or with "cosine peaks early then
  decays" — only a re-eval of ep 4 at higher game count tells us which.
- **R1 v2 hasn't finished.** Currently at ep 15 of a 40-epoch run.
  The d10 baseline added +105 Elo between ep 5 and ep 15; if R1 v2
  does anything similar, the peak is still ahead of us.
- **Soft-target dilution** at d15 is still a plausible micro-loss
  ([details](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/results.md#three-honest-hypotheses-for-the-underperformance))
  but it's no longer the dominant story — the cosine schedule
  recovered most of what the constant LR was losing.

## Head-to-head launcher — d10 best vs d15 best {#head-to-head-launcher}

To remove the Stockfish anchor noise from the comparison, run the two
networks directly against each other using MCTS at the same sim count
on both sides:

```
bash infra-eks/launchers/h2h-d10-vs-d15.sh
```

Defaults to d10 ep 15 vs d15 ep 7 at sims=4,000 (the current best of
each), 104 games, alternating colors. Result lands at
`s3://wm-chess-library-594561963943/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-40x256/20260522T2229Z-full46M-40x256/h2h-<run-id>.txt`.
Override `CKPT_B_S3=...` and `NB_B/NF_B` env vars to re-run against d15's
actual peak when training completes.

Underlying script: [`experiments/selfplay/scripts/h2h_mp.py`](https://github.com/shehio/world-models/blob/main/experiments/selfplay/scripts/h2h_mp.py)
(extended with per-agent `--n-blocks-a/--n-blocks-b` so 20×256 and
40×256 can play each other).

## Summary {#summary}

| variable changed | hypothesis |
|---|---|
| network capacity (40×256 vs 20×256, 5M data) | **rejected** |
| soft vs hard targets | **partial** (helps strong teachers, hurts weak) |
| eval-side search (4,000 vs 800 sims) | **confirmed** (+277) |
| data scale (30M vs 5M positions) | **confirmed** (+199) |
| stronger teacher (d15 vs d10) at full data, constant LR | **rejected** (R1 / R2 both topped out below d10's 2,189) |
| stronger teacher (d15 vs d10) at full data, **cosine LR** | **confirmed** (R2 v2 ep 4 = **2,285**, R1 v2 ep 7 = **2,209**) |

The bitter-lesson levers (more compute via search + data) confirmed
again. The "smarter network" lever (more parameters at constant data)
still rejected. The "stronger teacher" lever turned out to be gated on
**LR schedule** — d15 doesn't outperform d10 with constant LR, but
does with cosine. Strongest checkpoint to date:
**2,285 Elo @ UCI=1,800** from R2 v2 (d15 46M, 20×256 cosine) at epoch 4
with sims=4,000 (CI [2,177, 2,554]). Tighter-CI second-place:
**2,209 Elo** from R1 v2 (d15 46M, 40×256 cosine) at epoch 7,
CI [2,115, 2,389].

Self-play RL on top of the strongest distilled prior has been
[attempted six times](/next/#selfplay-postmortem) on spot and
evicted before iter 1 finished each time — see the postmortem on the
[next steps page](/next/).
