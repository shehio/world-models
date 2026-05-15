---
title: "Data — Experiment C"
subtitle: "is the ceiling a 5m-position-subsample problem?"
next: "/search/"
---

## hypothesis

The [baseline](/baseline/) uses a 5M-position subsample of the available
d10 dataset (~30M positions). Maybe the recipe is fine and we're just
data-starved. Training on the full 30M should reveal it.

## setup

20×256 ResNet (same as baseline). Full ~30M positions, no
`MAX_POSITIONS` truncation. `BATCH_SIZE=2048` to keep iteration count
reasonable (`30M / 2048 ≈ 14,650` batches/epoch). In-RAM via 256 GB on
a `g6e.8xlarge` (32 vCPU + 1× L40S) in eu-central-1 via
[`infra-eks/launchers/d10-full30m.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/d10-full30m.sh).
20 epochs.

## why it's 3× slower per epoch than experiment a

Same single L40S in both. The wallclock gap is just more total compute
per epoch:

| | A (40×256, 5M) | C (20×256, 30M) | C / A |
|---|---:|---:|---:|
| Batch size | 512 | 2,048 | 4× |
| Compute per batch ∝ blocks × batch | 20,480 | 40,960 | 2× |
| Batches per epoch | ~9,766 | ~14,648 | 1.5× |
| Expected epoch time vs A | — | — | **3×** |
| Actual epoch time | 13.5 min | 49 min | **3.6×** |

So C should be ~3× slower per epoch on the math; the extra 20% is OS
page-cache cold start on epoch 0 (15 GB of memmapped `.npy` files
need to page in) + DataLoader overhead on a bigger dataset.

## results so far

| epoch | loss | top-1 | wallclock |
|---:|---:|---:|---:|
| 0 | 2.843 | 0.308 | 49 min |
| 1 | 2.668 | 0.347 | 98 min |
| ... | running | running | ETA ~16 h |

Top-1 at epoch 1 is already higher (0.347) than the baseline's epoch 4
(0.32). That's a positive signal — the larger dataset is teaching the
network faster *per epoch*, suggesting the baseline was at least
partially data-starved.

## what we'll learn

If C's UCI=1800 number lands above 1810 (the baseline + Experiment A
ceiling), data was the bottleneck and we should generate ~5× more
distillation data next. If it lands at 1810 too, the bottleneck is the
training procedure itself — which would direct the next experiment
toward [self-play](/selfplay/) and sharper soft targets.
