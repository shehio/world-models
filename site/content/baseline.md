---
title: "Baseline — d15 20×256"
subtitle: "the reference run that everything else is measured against"
next: "/capacity/"
---

## the setup

A 20-block × 256-channel ResNet — literally AlphaZero's architecture — trained
by supervised distillation from Stockfish 17 at search-depth 15. The teacher
is asked for its top-8 candidate moves per position with their centipawn
scores, softened into a probability distribution at temperature `T = 1`
pawn. The student learns to match that distribution (cross-entropy) and the
final game outcome (MSE on `[-1, +1]`).

Dataset: ~100k games of Stockfish self-play, 5M positions after subsampling.
Network: 23.7M params. Training: 20 epochs on a single L4 GPU, bf16 autocast,
batch 1024, ~13 min/epoch, ~5 hours wallclock.

## the trajectory

100-game evals vs Stockfish `UCI_Elo=1350`, 800 MCTS sims/move.

| epoch | UCI=1350 Elo | W / D / L | notes |
|---:|---:|---:|---|
| 4 | 1701 | 62 / 35 / 3 | loss 2.81, top-1 0.31 |
| 9 | 1693 | 62 / 36 / 2 | loss plateaus ~2.59 |
| 14 | 1759 | 91 / 8 / 5 | top-1 ~0.34, fewer draws |
| **19** | **1807** | 93 / 8 / 3 | the headline number |

Top-1 plateaus around 0.34 by epoch 10; loss is essentially flat after that.
Each additional epoch buys 10–50 Elo within noise. This is the "1800
ceiling" that the [capacity](/capacity/), [data](/data/), and
[search](/search/) experiments were designed to probe.

## what it actually scores

The model wins 93 of 100 games against a calibrated 1350-Elo opponent.
That's solidly stronger than a club player. Plug the win rate into the
standard Elo formula:

`gap = -400 · log10(1/score - 1) = -400 · log10(0.067/0.933) = +457`

So `1350 + 457 = 1807` Elo. The 95% confidence interval on the score
gives an Elo CI of roughly [1704, 2034] — wide because we only played
100 games, but the lower bound is comfortably above 02b's hard-target
distillation result (1185 Elo).

## why this isn't AlphaZero-level

The teacher is Stockfish at depth 15, which is itself ~2500 Elo. The
student can never exceed teacher minus search contribution; in practice
the gap is ~600–800 Elo because the student does a single forward pass
where Stockfish did 10-ply alpha-beta. Distillation alone can't close
that gap. See [self-play](/selfplay/) for the next step.
