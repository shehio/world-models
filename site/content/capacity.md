---
title: "Capacity — Experiment A"
subtitle: "is the 1800-elo ceiling a network-capacity problem?"
next: "/data/"
---

## hypothesis

The [baseline](/baseline/) is a 20-block × 256-channel ResNet. Maybe 24M
params is too few to absorb the 5M-position dataset, and the loss-plateau
we see is the model running out of representational capacity. Doubling
the depth to 40 blocks (≈48M params) should buy Elo if so.

## setup

Identical to the baseline except `n_blocks=40`. Same 5M-position d15
dataset, same loss, same training schedule. Smaller batch (`512` instead
of `1024`) because the deeper network costs ~2× per batch. Runs on a
single L40S in eu-central-1 via
[`infra-eks/launchers/d15-40x256-eu.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d15-40x256-eu.sh).
~13.5 min/epoch — almost identical to baseline because the deeper-but-
smaller-batch trade cancels out on a single L40S.

## results

100-game evals vs Stockfish `UCI=1350` (the historical anchor) and
`UCI=1800` (a tighter calibrated anchor — see [eval](/eval/)).

| epoch | UCI=1350 elo | UCI=1800 elo | W/D/L vs UCI=1800 |
|---:|---:|---:|---:|
| 4 | 1851 | **1813** | 37 / 34 / 33 |
| 9 | 1851 | 1780 | 26 / 46 / 32 |
| 14 | 1759 | 1722 | 22 / 37 / 45 |
| **19** | **1820** | **1810** | 32 / 43 / 29 |

The UCI=1800 anchor is where the score is near 0.5 and the Elo CI is
tightest — read that column.

## verdict

**Hypothesis rejected.** The 40×256 net lands at the same ~1810 Elo as
the 20×256 baseline (1807). More parameters bought nothing on this data.

That means the 1800 ceiling is **not** a capacity problem. It's either a
data problem (more positions / better teacher) or a procedure problem
(better targets / RL on top of distillation). Both are explored in the
[data](/data/) and [self-play](/selfplay/) experiments.

## why a deeper net didn't help

The most likely explanation: 5M positions for a 24M-param net is already
in the over-parameterized regime, so adding more parameters doesn't add
fitting power that the data could meaningfully exploit. Top-1 accuracy
on training data was already 0.34 with the smaller net — and it stays
0.34 with the bigger net. The student is hitting a *teacher signal*
limit, not a *representation* limit.

A natural follow-up is [Experiment C](/data/): same network, ~6× the
data. If C climbs above 1810, data was the bottleneck. If it lands at
1810 too, the bottleneck is the distillation procedure itself.
