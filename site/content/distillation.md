---
title: "Distillation"
subtitle: "soft multipv targets from a strong teacher"
next: "/eval/"
---

## the setup

Stockfish plays self-games at fixed search-depth `D = 15`. At each
move, before the chosen move is played, the teacher reports its
`multipv = 8` top candidate moves with their centipawn scores. Those
scores are passed through a softmax at temperature `T = 1` pawn to
produce a target distribution over the 8 candidates:

`π(move_k) = softmax( cp_k / (100·T) )`

The student is trained to match `π` (cross-entropy) and the eventual
game outcome `z` (MSE on `[-1, +1]`). The game is played out by
Stockfish to provide the value target.

## the npz schema

```
states            (N, 19, 8, 8)  float32   — board encoding (same as AZ)
moves             (N,)           int64     — stockfish's chosen move idx
zs                (N,)           float32   — game outcome from STM POV
multipv_indices   (N, K=8)       int64     — top-K candidate move indices
multipv_logprobs  (N, K=8)       float32   — softmax-logprobs over those K
```

Padding for late-game positions with fewer than 8 legal moves:
`multipv_indices[i, j] = -1` and `multipv_logprobs[i, j] = -inf`.
The training loss masks these out by construction since `exp(-inf) = 0`.

## why soft instead of hard

Hard targets ("the move Stockfish played") collapse multipv data into
one-hot. Soft targets preserve the *relative* preference between the
top-8 candidates — at temperature `T = 1`, a 50cp gap between two
moves yields a 62 / 38 split, not 100 / 0. The student learns "this
move is best, but these three are also reasonable, and these four are
clearly worse."

In practice, at the d10 teacher depth used for [02b](/baseline/#why-this-isnt-alphazero-level),
soft targets *underperform* hard one-hot by ~130 Elo — the student
learns to *hedge* rather than commit. At d15 they catch up and the
absolute number jumps from 1,185 to 1,807 Elo — a +622 lift mostly
from the stronger teacher.

A natural follow-up is **sharper T** (e.g. T = 0.3) which keeps some
ranking info but forces the student to commit. That's on the roadmap
in [EXPERIMENTS_LOG.md](https://github.com/shehio/world-models/blob/main/EXPERIMENTS_LOG.md)
but not yet run.

## the loss

```python
policy_loss = -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
value_loss  = F.mse_loss(value_pred, z)
total       = policy_loss + value_weight * value_loss
```

where `target_dist` is the dense reconstruction of the sparse multipv
distribution at each position. `value_weight = 1.0` by default.

## why the student can't exceed teacher

A Stockfish at depth 15 ≈ 2,500 Elo, but its strength comes from
calculating ~10 plies deep through millions of positions per move.
The student is being asked to predict the *output* of that search in
a single forward pass. There's no calculation, only a learned prior.

Empirically, supervised distillation lands ~600–800 Elo below the
teacher — that's the alpha-beta search contribution that *can't* be
transferred to a small ResNet. To close the gap requires
[search at inference time](/search/) (Experiment E showed +277 Elo
from 800 → 4,000 sims) or [self-play RL on top](/selfplay/) (the Lc0
path, currently running).
