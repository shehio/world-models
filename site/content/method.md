---
title: "Method"
subtitle: "the distillation loss, MCTS at inference, Elo bisection"
next: "/vs-alphazero/"
aliases:
  - /distillation/
  - /eval/
  - /elo-bisect/
---

## The Loss

Stockfish plays self-games at fixed search-depth `D = 15`. At each
position, before the chosen move is played, the teacher reports its
`multipv = 8` candidate moves with their centipawn scores. Those
scores are passed through a softmax at temperature `T = 1` pawn to
produce a target distribution over the 8 candidates:

```
π(move_k) = softmax( cp_k / (100·T) )
```

The student is trained to match `π` (cross-entropy) and the eventual
game outcome `z` (MSE on `[-1, +1]`).

```python
policy_loss = -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
value_loss  = F.mse_loss(value_pred, z)
total       = policy_loss + 1.0 * value_loss
```

`target_dist` is the dense reconstruction of the sparse multipv
distribution. Padding for late-game positions with fewer than 8 legal
moves: `multipv_indices[i, j] = -1` and `multipv_logprobs[i, j] = -inf`,
masked out by `exp(-inf) = 0`.

The choice of *soft* multipv targets vs *hard* one-hot targets is its
own ablation — see
[Experiments → Soft vs Hard Targets](/experiments/#soft-vs-hard).

## Why The Student Can't Exceed The Teacher

Stockfish at depth 15 is ~2,500 Elo, but its strength comes from
calculating ~10 plies deep over millions of positions. The student is
asked to predict the *output* of that search in a single forward pass.
There's no calculation, only a learned prior.

Empirically, supervised distillation lands ~600–800 Elo below the
teacher. That's the alpha-beta search contribution that *can't* be
transferred to a small ResNet. To close the gap requires either
deeper search at inference time (a free +277 Elo, see
[Experiments → Search](/experiments/#search))
or self-play RL on top (the Lc0 path, [running now](/next/)).

## How We Measure Elo

The default for milestone checkpoints is **Elo bisection** — a binary
search over Stockfish's `UCI_Elo` setting that converges to the score
near 0.5, where the Elo conversion is most accurate. The legacy
"fixed-anchor" approach (a single 100-game match against UCI=1,350)
is kept as a fast sanity check during training but isn't authoritative.

### The Formula

Each probe plays N games against a Stockfish opponent at fixed
`UCI_Elo = O` and computes the agent's score `s ∈ [0, 1]` (W + ½·D out
of N). The agent's Elo relative to the opponent is the inverse of the
logistic CDF:

```
ΔElo = -400 · log10(1/s - 1)
Elo  = O + ΔElo
```

So a score of `s = 0.5` puts the agent at the opponent's Elo. A score
of `s = 0.933` (the d15 baseline vs UCI=1,350) implies
`-400 · log10(0.067/0.933) = +457`, giving an absolute Elo of
`1,350 + 457 = 1,807`.

Worked example from a real result file:

```
=== eval vs Stockfish UCI=1350 ===
W/D/L: 93 / 8 / 3
score: 0.933   95% CI: [0.885, 0.981]
Elo gap to opponent: +457
Agent absolute Elo (anchor 1350): 1807 [1704, 2034]
```

The Elo CI is wide because at N=100 games the score CI is ±10
percentage points, which translates to a wide *implied* Elo range
when the opponent is far from the agent's strength. That's the
problem the bisection solves.

### The Bisection

1. Bracket: `lo = 1,350`, `hi = 2,800` (covers all checkpoints to date).
2. Probe at the midpoint with N=104 games.
3. If `score ≥ 0.5`: agent is stronger than `mid` → raise `lo`.
4. If `score < 0.5`: agent is weaker than `mid` → lower `hi`.
5. Stop when bracket ≤ 100 Elo, **or** when a probe lands in
   `[0.45, 0.55]` (where the Elo conversion is sharpest).

Three to four probes converge to ±50 Elo precision in roughly 30–40
minutes on a g6.4xlarge. Implementation at
[`experiments/distill-soft/scripts/elo_bisect.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/scripts/elo_bisect.py),
covered by 13 unit tests for the Elo math, bracket halving, stop
conditions, and end-to-end convergence on synthetic agents at 1,400 /
1,800 / 2,400 Elo.

### Legacy: Fixed-Anchor Evals

Before bisection we used a simpler protocol: play 100 games at each of
two fixed `UCI_Elo` anchors (1,350 and 1,800) and report both
absolute Elos. This is what the [auto-eval
daemon](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-autoeval-daemon.sh)
still runs on every new checkpoint — fast, comparable across the
whole training run, but the Elo CI can be wide when the score is far
from 0.5. The bisection is the authoritative number for milestones.

## Why Not Just Track Training Loss

Training loss and top-1 accuracy plateau by epoch 10, but Elo keeps
climbing through epoch 20. The *training* objective and the
*play-strength* objective aren't perfectly aligned. Top-K accuracy of
87% sounds excellent; the actual Elo says the model is 1,807, not
2,500 (the teacher's strength). The bridge between supervised metrics
and game-playing Elo is uncertain, sometimes inverted, and worth
measuring directly with games.
