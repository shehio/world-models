# 02c results — multipv soft targets + 20×256, 30 epochs

> **Negative result.** Bigger network + multipv soft targets, fully
> trained, **underperformed 02b's smaller-network + hard-target recipe**
> by ~130 Elo at the same teacher quality (Stockfish depth=10).

## Headline numbers

100-game-equivalent eval vs Stockfish UCI_Elo=1320 (02b's anchor; 02c
ran at SF1350 because apt-Stockfish-18 enforces min 1350, normalized).

| Run | Network | Targets | Training | Real Elo | W/D/L (per 100g) |
|---|---|---|---|---:|---:|
| **02b d10** (baseline) | 10×128 (~3M) | hard one-hot | 20 epochs MPS | **1185** [1104, 1254] | 19 / 25 / 56 |
| 02c 10-epoch | 20×256 (~24M) | multipv=8 soft (T=1) | 10 epochs | 998 (~968 normalized) | 7.5 / 8.3 / 84 |
| **02c 30-epoch** | 20×256 (~24M) | multipv=8 soft (T=1) | 30 epochs | **1086** [988, 1157] (~**1056** normalized) | 7.5 / 20.8 / 71.7 |

**Gap to 02b after fully training**: ~130 Elo behind.

**What 20 more epochs bought (10ep → 30ep)**: +88 Elo. Real but small.

## Training trajectory (from train_history.json)

| epoch | loss | top-1 | top-K | grad-norm |
|---:|---:|---:|---:|---:|
| 0   | 4.02 | 14.7% | 54.6% | 3.68 |
| 9   | 2.65 | 31.3% | 81.8% | 1.48 |
| 14  | 2.53 | 33.9% | 84.3% | 1.32 |
| 19  | 2.46 | 35.3% | 85.6% | 1.37 |
| 24  | 2.42 | 36.0% | 86.4% | 1.71 |
| **29** | **2.40** | **36.7%** | **87.0%** | 1.55 |

Loss kept dropping (slope flattening). Top-K kept climbing. Top-1 mostly
plateaued after epoch ~20. The model wasn't at hard convergence at
epoch 29, but per-epoch returns were already < 0.2% top-1.

## Three honest hypotheses for the underperformance

Ranked by what the data supports:

### 1. Soft targets dilute signal at d10 teacher quality
Top-K=87% says "Stockfish's chosen move is in the network's top-8
predictions almost every time." Top-1=36.7% says "the network often
picks a *different* move from that top-8 than Stockfish would." That's
the **hedging failure mode** — the student learned to spread probability
across candidates instead of committing. Chess punishes hedging at
decision time: at 800 sims, MCTS amplifies a sharp prior more than a
flat one.

02b's hard targets force the student to commit to Stockfish's actual
choice. Empirically at d10, that turns out to be the better signal,
even though information-theoretically soft targets carry more bits.

### 2. 20×256 is undertrained for 250k positions
21M params × 243k positions ≈ **86 positions per parameter**. For 02b's
3M-param net on the same dataset: **80 positions per parameter** — about
the same! But 02b had hard targets that compress more cleanly into a
small network.

The bigger net needs *more* data, not less. We doubled the network and
kept the same dataset, which means we under-supplied gradient signal.

### 3. Temperature T=1 pawn is too smooth
With T=1 pawn, a 50cp gap between two moves yields a 62/38 probability
split. At d10, many positions have several moves within 30–50cp. That
gives target distributions with effective entropy H≈1.54 bits — closer
to "uniform over 4 candidates" than "this one move is best." Sharper
T (e.g. T=0.3) would force more concentration and may close the gap
to hard targets while keeping some secondary-ranking info.

## What this **does not** show

- It does **not** show distillation is bad. d10 distillation (02b) is
  still the best chess result in this repo.
- It does **not** show multipv soft targets are bad in general. It
  shows they're not strictly better than hard one-hot at d10 teacher
  quality.
- It does **not** isolate the bigger-net effect from the soft-target
  effect — we changed two variables at once.

## Clean follow-up experiments (ranked by information per dollar)

| Exp | Change vs 02c-30ep | Tests | Cost |
|---|---|---|---:|
| **A** | hard one-hot targets (everything else same) | "was the soft-target the problem, or the bigger net?" | ~$5 |
| **B** | T=0.3 soft (sharper) | "are soft targets per se bad, or is T=1 too smooth?" | ~$5 |
| **C** | d15 teacher, multipv=8 (full original plan) | "does multipv help when the teacher is stronger?" | ~$25 |
| D | distill→AZ-self-play (Lc0 path) | "can self-play close the search gap?" | $50+, days |

**Recommended next step**: A. It's the cleanest single-variable ablation
— same data, same net, same training time, just different loss. If A
beats 02b's 1185 with hard targets + 20×256, **the soft target was the
problem**. If A also lands near 1086, **the bigger net was the problem**
(undertraining-per-parameter).

## Wall-clock and cost (the full AWS run)

| Phase | What | Wall | Cost on g5.8xlarge |
|---|---|---:|---:|
| 0 | terraform apply, bootstrap, code rsync, key rotation, IP fights | ~30 min | $1.20 |
| 1 | Data gen — d10 mpv8 2000g on 28 CPU workers | 28 min | $1.15 |
| 2 | Training — 30 epochs on A10G GPU | 75 min | $3.10 |
| 3a | Eval — 120g vs SF1350 @ 800 sims | 34 min | $1.40 |
| 3b | Eval — 120g vs SF1350 @ 1600 sims | (still running at writeup) | ~$2.50 |
| | **Total spent so far** | **~3 h** | **~$9** |

The training was 2× slower per epoch than baseline because of the new
per-batch observability (grad-norm + JSON writes). I'd accept that
cost again — debugging this negative result without the observability
would have been blind.

## Lessons that generalize

1. **More compute on the same recipe rarely beats a better recipe at
   the same compute.** 30 epochs of multipv-soft + 20×256 ≈ 20 epochs
   of hard-target + 10×128 wallclock-wise, and the simpler recipe wins.

2. **Top-K accuracy is a *training* metric, not a *play* metric.** 87%
   top-K sounds excellent; the resulting Elo says it's not. The bridge
   between supervised metrics and game-playing Elo is uncertain and
   sometimes inverted.

3. **Observability changes you wish you had** *during* a run are the
   ones you need to bake in *before* the next run. Per-batch progress,
   gradient norms, and a saved training-history JSON cost almost
   nothing in code complexity and saved this analysis from being pure
   guesswork.

## Files

- `aws_results/run02_30ep/train_history.json` — full per-epoch metrics
- `aws_results/run02_30ep/distilled_epoch029.pt` — gitignored; pulled locally only
- `aws_results/run02_30ep/az_02c_e2e30.log` — full combined log
