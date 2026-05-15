---
title: "Findings"
subtitle: "three ablations on a distilled-from-stockfish baseline"
next: "/method/"
---

The setup is fixed: a 20-block × 256-channel ResNet, supervised
distillation from Stockfish 17 at search-depth 15, multipv=8 soft
targets at temperature `T = 1` pawn, 100-game evals against a
calibrated Stockfish opponent at fixed `UCI_Elo`. The baseline scores
**1,807 Elo** at 800 MCTS sims/move on a 5M-position dataset.

Each finding below changes *one variable*, holds everything else
constant, and asks what Elo it buys.

## the baseline trajectory

100-game evals vs Stockfish `UCI=1350`, 800 MCTS sims/move, 20-block
network.

| epoch | UCI=1350 elo | W / D / L |
|---:|---:|---:|
| 4 | 1,701 | 62 / 35 / 3 |
| 9 | 1,693 | 62 / 36 / 2 |
| 14 | 1,759 | 91 / 8 / 5 |
| **19** | **1,807** | 93 / 8 / 3 |

Loss plateaus around 2.59 by epoch 10; top-1 accuracy plateaus ~0.34.
Each additional epoch buys 10–50 Elo within noise. This is the
"1,800 ceiling" the three ablations probe.

## network capacity — doubling depth didn't help

**Hypothesis:** the ceiling is the network running out of representational
capacity. Doubling the depth (40 blocks instead of 20, ~48M params
instead of 24M) should buy Elo if so.

**Setup:** identical to the baseline except `n_blocks=40`. Same 5M
positions, same loss, same 20 epochs. Single L40S, batch 512 (the
deeper net costs ~2× per batch so we cut batch in half to keep memory
sane). 13.5 min/epoch — close to the baseline because deeper-but-
narrower-batch trades roughly cancel.

| epoch | UCI=1350 elo | UCI=1800 elo | W/D/L vs UCI=1800 |
|---:|---:|---:|---:|
| 4 | 1,851 | **1,813** | 37 / 34 / 33 |
| 9 | 1,851 | 1,780 | 26 / 46 / 32 |
| 14 | 1,759 | 1,722 | 22 / 37 / 45 |
| **19** | **1,820** | **1,810** | 32 / 43 / 29 |

The UCI=1,800 anchor is where the score sits near 0.5 and the Elo CI is
tightest — read that column.

**Result.** **~1,810 Elo, identical to the baseline.** More parameters
bought nothing on this data. The ceiling is **not** a capacity problem.

**Why a deeper net didn't help:** 5M positions for a 24M-param network
is already in the over-parameterized regime. Top-1 was already 0.34
with the smaller net and stays 0.34 with the bigger one. The student
is hitting a *teacher-signal* limit, not a *representation* limit.

## eval-side search — +277 elo from search alone

**Hypothesis:** the baseline is evaluated at 800 MCTS sims/move
(AlphaZero's *training-time* value). At competitive play AZ used many
tens of thousands of sims. Maybe the model is stronger than the
routine eval reveals.

**Setup:** same checkpoint (d15 epoch 19), re-eval at `--sims 4000`.
Single g6.4xlarge, 104 games, 93.6 min wallclock. No retraining,
just deeper search at inference.

| sims / move | W / D / L vs UCI=1350 | score | Elo |
|---:|---:|---:|---:|
| 800 (baseline) | 93 / 8 / 3 | 0.933 | **1,807** |
| **4,000** | **101 / 3 / 0** | **0.986** | **2,084** |

**Result.** **+277 Elo from search alone, no retraining.** The model
sweeps 101 of 104 games at 4,000 sims; score saturation means we're
now hitting the ceiling of *the opponent's* strength rather than the
agent's.

**Interpretation.** Distilled priors transfer well to deeper inference-
time search — exactly what Lc0 demonstrated empirically. The network's
policy prior guides MCTS toward strong moves; more rollouts let MCTS
refine that prior into a sharper visit distribution. The student
becomes a better chess player not by learning more, but by *thinking
longer*.

**Practical consequence.** The "1,800 ceiling" reported from routine
evals is a *floor*, not a ceiling. There's ~280 Elo trivially
available from deeper eval search. For honest comparison with
AlphaZero, the relevant number is the search-saturated one.

The natural follow-up is a full sims sweep — same checkpoint, eval at
200 / 800 / 2,000 / 4,000 / 8,000 / 16,000 sims. The Elo-vs-log(sims)
curve will reveal the saturation point. The machinery
([`elo_bisect.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/scripts/elo_bisect.py))
is in place.

## data scale — 6× more positions

**Hypothesis:** the baseline used a 5M-position subsample of an
available 30M-position dataset. Maybe the recipe is fine and we're
just data-starved.

**Setup:** 20×256 net (same as baseline), full ~30M positions, no
`MAX_POSITIONS` truncation. `BATCH_SIZE=2048` to keep iteration count
reasonable (~14,650 batches/epoch). In-RAM via 256 GB on a
`g6e.8xlarge` (32 vCPU + 1× L40S). 20 epochs.

**Why this is 3× slower per epoch than the capacity experiment.**
Same single L40S in both. The wallclock difference comes from raw
compute:

| | capacity (40×256, 5M) | data (20×256, 30M) | ratio |
|---|---:|---:|---:|
| Compute per batch | 20,480 | 40,960 | 2× |
| Batches per epoch | ~9,766 | ~14,648 | 1.5× |
| **Expected epoch time** | — | — | **3×** |
| Actual epoch time | 13.5 min | 49 min | **3.6×** |

The extra ~20% above the napkin estimate is OS page-cache cold start
on epoch 0 (15 GB of memmapped files page in) + DataLoader overhead
on the bigger dataset.

**Results so far:**

| epoch | loss | top-1 |
|---:|---:|---:|
| 0 | 2.843 | 0.308 |
| 1 | 2.668 | 0.347 |
| ... | running | running |

Top-1 at epoch 1 (0.347) is already higher than the baseline reached
at epoch 4 (~0.32) — the larger dataset is teaching the network
meaningfully faster *per epoch*, which suggests the baseline was at
least partially data-starved. ETA: full 20 epochs around 03:30Z May
15 (first eval ckpt) → 15:30Z May 15 (final).

**Outstanding question:** if the final UCI=1,800 number lands above
1,810, dataset size was the bottleneck and the next round should
generate ~5× more distillation data. If it lands at 1,810 too, the
bottleneck is the *training procedure itself*, which directs the next
experiment toward self-play RL ([next →](/next/)).

## summary

| variable changed | result | hypothesis |
|---|---|---|
| **network capacity** (40×256 vs 20×256) | ~1,810 → 1,810 (no change) | **rejected** |
| **eval-side search** (4,000 vs 800 sims) | 1,807 → 2,084 (+277) | **confirmed** |
| **data scale** (30M vs 5M positions) | running | TBD |

Capacity is decisively *not* the bottleneck. Eval search recovers a
huge slice. The data result will inform whether more distillation data
helps or whether the procedure is the real bottleneck.
