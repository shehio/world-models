---
title: "Findings"
subtitle: "three ablations on a distilled-from-stockfish baseline"
next: "/method/"
aliases:
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

Each finding below changes *one variable*, holds everything else
constant, and asks what Elo it buys.

## the baseline trajectory

100-game evals vs Stockfish `UCI=1350`, 800 MCTS sims/move, 20-block
network.

| epoch | Elo |
|---:|---:|
| 5 | 1,701 |
| 10 | 1,693 |
| 15 | 1,759 |
| **20** | **1,807** |

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

| epoch | Elo @ UCI=1800 |
|---:|---:|
| 5 | **1,813** |
| 10 | 1,780 |
| 15 | 1,722 |
| **20** | **1,810** |

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

**Setup:** same checkpoint (d15 epoch 20), re-eval at `--sims 4000`.
Single g6.4xlarge, 104 games, 93.6 min wallclock. No retraining,
just deeper search at inference.

| sims / move | Elo |
|---:|---:|
| 800 | 1,807 |
| **4,000** | **2,084** |

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

## data scale — 6× more positions (the winning lever)

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
20% is OS page-cache cold start on epoch 0 (15 GB of memmapped files
page in) plus DataLoader overhead.

### results

100-game evals vs Stockfish at UCI=1,800.

| epoch | Elo |
|---:|---:|
| **5** | **2,004** |
| **10** | **2,009** |
| 15 | 1,957 |
| 20 | 1,957 |

The data ablation peaks at **ep 10: 2,009 Elo at the tight UCI=1,800
anchor**.

### same teacher, more data

| dataset (d10 teacher) | ep 5 Elo @ 1350 |
|---|---:|
| 5 M subset | 1,749 |
| **30 M full** | **2,084** |

Same teacher, 6× more data → **+335 Elo** at the easier anchor.

### stronger teacher vs more data

| run | Elo @ UCI=1800 |
|---|---:|
| baseline (d15, 5 M) | 1,810 |
| **30 M (d10, weaker teacher)** | **2,009** |

**Verdict: the bottleneck on the original baseline was data.** Six
times more positions buys ~200 Elo at the calibrated anchor — even
with a *weaker* teacher (depth-10 vs depth-15).

## summary

| variable changed | hypothesis |
|---|---|
| network capacity (40×256 vs 20×256) | **rejected** |
| eval-side search (4,000 vs 800 sims) | **confirmed** (+277) |
| data scale (30M vs 5M positions) | **confirmed** (+199) |

Capacity is decisively *not* the bottleneck. Eval search recovers a
huge slice. **Data was the real bottleneck.** Strongest checkpoint to
date: **2,084 Elo @ UCI=1,350 / 2,004 @ UCI=1,800** from the d10
full-30M run at epoch 5 — the headline number for the whole project.
A 5× larger d15 dataset is the natural next step
([what's next →](/next/)).
