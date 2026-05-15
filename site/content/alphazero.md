---
title: "Vs AlphaZero"
subtitle: "same network · different training signal · ~10,000× less compute"
next: "/infra/"
---

## the headline comparison

| | AlphaZero (2017) | This repo |
|---|---|---|
| Approach | Tabula-rasa self-play RL | Supervised distillation from Stockfish |
| Network | 20-block ResNet, 256 filters | **Identical** (20×256) |
| MCTS at training | 800 sims/move on agent's own search | None — Stockfish provides targets |
| Training positions | ~3.5B (44M games × 80 moves) | 5M–30M |
| Training compute | 64 TPU-v2 × 9h + ~45k TPU-v1-hr self-play | 1 GPU × 5 h ≈ 5 GPU-hours (~10,000× less) |
| Eval MCTS sims | 800 training / many tens of thousands tournament | 800 routine / 4,000 on Experiment E |
| Teacher | None | Stockfish d10 (~2200) or d15 (~2500), multipv=8, T=1 |
| Peak Elo | ~3500–3600 | 2,084 (d15 ep19 @ 4,000 sims vs UCI=1350) |

## what's literally identical

The architecture. AlphaZero is a 20-block × 256-channel ResNet with a
policy head outputting 4,672 move logits (8×8×73 chess move encoding)
and a value head outputting a scalar in `[-1, +1]`. That's the exact
network we use here — about 23.7M params. Same residual blocks, same
batch-norm-after-conv ordering, same tanh on the value head.

The eval anchor (Stockfish at calibrated `UCI_Elo`) is a measurement
choice, not part of the model. Everything else — *how* the weights
get there — is different.

## what's deliberately different

**Approach.** AlphaZero is tabula rasa. The weights start random; the
*only* training signal is "did this position lead to a win?" In this
repo, the network is initialized by supervised distillation from
Stockfish — the targets are Stockfish's top-8 moves with softmax-cp
weights, not MCTS visit counts from the agent itself.

**Compute.** AlphaZero used 5,000 first-generation TPUs for self-play
data generation and 64 second-generation TPUs for training, for ~9
hours. Conservatively, that's ~45,000 TPU-v1-hours plus ~576
TPU-v2-hours. Our supervised distillation runs cost ~5 hours on a
single L4 or L40S GPU — roughly four orders of magnitude less.

**Data.** AZ's 44M self-play games × ~80 moves/game = ~3.5 billion
training positions. The largest dataset we've trained on (Experiment
C) is ~30M positions — about 100× less. The 5M subsample used for the
baseline is ~700× less.

**Inference search.** AZ used 800 sims during training and many tens
of thousands at competitive play. Our routine evals run at 800 sims;
[Experiment E](/search/) showed that the same checkpoint scores +277
Elo when re-evaluated at 4,000 sims — and we haven't pushed it
further yet.

## why a small project can still be interesting

Holding the architecture constant and varying the training procedure
isolates *the procedure*. Each ablation tells you something specific
about which knob controls the ceiling:

- [Capacity ablation](/capacity/) (40×256) → ceiling is **not** in the
  network.
- [Search ablation](/search/) (sims=4000) → ~280 Elo of the ceiling
  is in the eval budget, not the model.
- [Data ablation](/data/) (full 30M positions, running) → tests
  whether the rest is data-driven.
- [Self-play](/selfplay/) (running) → tests Lc0's "distill first, RL
  second" recipe on a single GPU.

If we had AZ's compute we wouldn't run these experiments — we'd just
do AZ's thing and report 3,500 Elo. The interesting question for a
small lab is: *which procedural changes are worth a lot per unit of
compute?* That's what the ablations are measuring.

## the realistic ceiling

Without a multi-GPU scale-up the realistic ceiling is somewhere in the
2,200–2,800 Elo range — strong-club to FIDE-master level on the same
hardware AZ used a single accelerator for one second on. That's not
grandmaster. But it *is* enough to evaluate every conceptual change
in the AZ recipe (search-during-training, target sharpness, self-play
quantity, network depth) for ~$10–$100 of cloud compute per
experiment. That's the project.
