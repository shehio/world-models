---
title: "Vs AlphaZero"
subtitle: "same network · different training signal · ~10,000× less compute"
next: "/next/"
aliases:
  - /alphazero/
---

## The Comparison

| | AlphaZero (2017) | this project |
|---|---|---|
| approach | tabula-rasa self-play RL | supervised distillation from stockfish |
| network | 20-block ResNet, 256 filters | **identical** (20×256) |
| MCTS at training | 800 sims/move on agent's own search | none — stockfish provides targets |
| training positions | ~3.5B (44M games × 80 moves) | 5M–30M |
| training compute | 64 TPU-v2 × 9h + ~45k TPU-v1-hr self-play | 1 GPU × 5 h ≈ 5 GPU-hours (~10,000× less) |
| eval MCTS sims | 800 training / many tens of thousands tournament | 800 routine / 4,000 on the search ablation |
| teacher | none | stockfish d10 (~2,200) or d15 (~2,500), multipv=8, T=1 |
| peak Elo | ~3,500–3,600 | **2,084** (d15 ep19 @ 4,000 sims vs UCI=1350) |

## What's Literally Identical

The architecture. AlphaZero is a 20-block × 256-channel ResNet with a
policy head producing 4,672 move logits (8×8×73 chess move encoding)
and a value head producing a scalar in `[-1, +1]`. That's the exact
network we use here — about 23.7M parameters. Same residual blocks,
same batch-norm-after-conv ordering, same `tanh` on the value head.

The eval anchor (Stockfish at calibrated `UCI_Elo`) is a measurement
choice, not part of the model. Everything else — *how* the weights
get there — is different.

## What's Deliberately Different

### Approach

AlphaZero is **tabula rasa**. The weights start random; the only
training signal is "did this position lead to a win?" Targets are
MCTS visit counts from the agent's *own* search.

In this project, the network is initialized by **supervised
distillation**. Targets are Stockfish's top-8 moves with softmax-cp
weights, not MCTS visit counts. The model never plays a game during
training — it watches Stockfish play 100k of them.

### Compute

AlphaZero used 5,000 first-generation TPUs for self-play data
generation and 64 second-generation TPUs for training, for ~9 hours.
Conservatively, that's ~45,000 TPU-v1-hours plus ~576 TPU-v2-hours.
Our supervised distillation runs cost ~5 hours on a single L4 or L40S
GPU — roughly four orders of magnitude less.

### Data

AZ's 44M self-play games × ~80 moves/game = ~3.5 billion training
positions. The largest dataset we've trained on is ~30M positions —
about 100× less. The 5M subsample used for the headline result is
~700× less.

### Inference Search

AZ used 800 sims during training and many tens of thousands at
competitive play. Our routine evals run at 800 sims; the
[search ablation](/experiments/#search)
showed that the same checkpoint scores +277 Elo when re-evaluated at
4,000 sims — and we haven't pushed it further yet.

## Why a Small Project Can Still Be Interesting

Holding the architecture constant and varying the training procedure
isolates *the procedure*. Each ablation tells you something specific
about which knob controls the ceiling:

- **capacity** (40×256 vs 20×256) → ceiling is **not** in the network.
- **search** (4,000 vs 800 sims) → ~280 Elo of the ceiling is in the
  eval budget, not the model.
- **data** (full 30M positions, running) → tests whether the rest is
  data-driven.
- **self-play** (running) → tests Lc0's "distill first, RL second"
  recipe on a single GPU.

If we had AZ's compute we wouldn't run these experiments — we'd just
do AZ's thing and report ~3,500 Elo. The interesting question for a
small lab is: *which procedural changes are worth a lot per unit of
compute?* That's what the ablations are measuring.

## The Realistic Ceiling

Without a multi-GPU scale-up the realistic ceiling is somewhere in the
2,200–2,800 Elo range — strong-club to FIDE-master level on the same
hardware AZ used a single accelerator for one second on. That's not
grandmaster. But it *is* enough to evaluate every conceptual change
in the AZ recipe (search-during-training, target sharpness, self-play
quantity, network depth) for ~$10–$100 of cloud compute per
experiment. That's the project.
