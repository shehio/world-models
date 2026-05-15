---
title: "Method"
subtitle: "the distillation loss, MCTS at inference, how we measure elo"
next: "/vs-alphazero/"
---

## the loss

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

## why soft, not hard

Hard targets ("the move Stockfish played") collapse multipv data into
one-hot. Soft targets preserve the relative preference: at `T = 1`, a
50-centipawn gap between two moves yields a 62 / 38 split, not 100 / 0.
The student learns "this move is best, but these three are also
reasonable, and these four are clearly worse."

At a *weak* teacher (depth 10, ~2,200 Elo) soft targets actually
underperform hard one-hot by ~130 Elo — the student learns to hedge.
At depth 15 they catch up, and the absolute number jumps from
1,185 → 1,807 Elo. Most of that lift is from the *stronger teacher*,
not from soft vs hard.

A sharper temperature (e.g. `T = 0.3`) is on the roadmap — it would
keep some multipv ranking information but force more commitment.

## why the student can't exceed the teacher

Stockfish at depth 15 is ~2,500 Elo, but its strength comes from
calculating ~10 plies deep over millions of positions. The student is
asked to predict the *output* of that search in a single forward pass.
There's no calculation, only a learned prior.

Empirically, supervised distillation lands ~600–800 Elo below the
teacher. That's the alpha-beta search contribution that *can't* be
transferred to a small ResNet. To close the gap requires either
deeper search at inference time (a free +277 Elo, see
[findings → search](/findings/#eval-side-search--277-elo-from-search-alone))
or self-play RL on top (the Lc0 path, [running now](/next/)).

## how we measure Elo

Every research repo invents its own definition of "this model is X Elo
strong." Ours is the most boring: play 100 games against a Stockfish
opponent set to a calibrated `UCI_Elo` and report the win rate.

Stockfish 17 implements `UCI_LimitStrength = true` / `UCI_Elo = N`.
It plays at *roughly* `N` Elo (validated against community matches),
still using its full evaluation engine but deliberately introducing
blunders proportional to the strength target.

If our model scores 0.5 against UCI=1,350 → model ≈ 1,350 Elo. If our
model scores 0.93:

```
gap = -400 · log10(1/score - 1) = -400 · log10(0.067/0.933) = +457
absolute = 1,350 + 457 = 1,807
```

## the auto-eval daemon

[`infra-eks/daemons/wm-autoeval-daemon.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-autoeval-daemon.sh)
polls S3 every 5 min. For every new `distilled_epochNNN.pt` without
a sibling `eval_results-<ckpt>.txt`:

1. Writes a `.claimed-eval-<ckpt>` marker (concurrent-poller safety).
2. Launches a one-shot `g6.4xlarge` EC2 in us-east-1.
3. Falls back to eu-central-1 if us-east hits `VcpuLimitExceeded`.
4. The EC2 pulls the GPU image, downloads the ckpt, plays 100 games
   vs `UCI=1350` and 100 vs `UCI=1800`, uploads results, and
   self-terminates.

The daemon parses `net-NxF` out of the ckpt path and passes the right
`--n-blocks` / `--n-filters` so 20×256 and 40×256 evals both work
transparently.

## two anchors per checkpoint

| anchor | purpose |
|---|---|
| `UCI = 1,350` | the historical reference. Every checkpoint ever evaluated is comparable on this scale. |
| `UCI = 1,800` | calibrated opponent near current model strength. CI is tightest where the score is near 0.5 — this is the most accurate Elo for d15-class models. |

An earlier "depth=1 top-skill" secondary test produced high-variance
drawish games and was replaced — see
[infra → failure modes](https://github.com/shehio/world-models/blob/main/docs/notes/failure-modes.md).

## reading a result file

```
=== eval vs Stockfish UCI=1350 ===
W/D/L: 93 / 8 / 3
score: 0.933   95% CI: [0.885, 0.981]
Elo gap to opponent: +457
Agent absolute Elo (anchor 1350): 1807 [1704, 2034]
```

The Elo CI looks wide because at 100 games the score CI is ±10
percentage points, which translates to a wide *implied* Elo range
when the opponent is far from model strength. The UCI=1,800 anchor
gives much tighter Elo bounds for the current generation of
checkpoints.

## elo bisect — for the single most accurate number

A single eval at one anchor reports a wide Elo CI when the score is
far from 0.5. For milestone checkpoints we use a binary search over
`UCI_Elo` in `[1,350, 2,800]`:

1. Probe at the midpoint with N games.
2. If score ≥ 0.5: agent is stronger than `mid` → raise `lo`.
3. If score < 0.5: agent is weaker than `mid` → lower `hi`.
4. Stop when bracket ≤ 100 Elo, **or** when a probe lands in
   `[0.45, 0.55]` (where the Elo conversion is most accurate).

Three to four probes converge to ±50 Elo precision in roughly 30–40
minutes on a g6.4xlarge. Implementation at
[`experiments/distill-soft/scripts/elo_bisect.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/scripts/elo_bisect.py),
tested by 13 unit tests covering the Elo math, bracket halving,
stop conditions, and end-to-end convergence on a fake agent at 1,400 /
1,800 / 2,400 Elo.

## why not just track training loss

Training loss / top-1 accuracy plateau by epoch 10, but Elo keeps
climbing through epoch 19. The *training* objective and the
*play-strength* objective aren't perfectly aligned. Top-K accuracy of
87% sounds excellent; the actual Elo says it's 1,807, not 2,500 (the
teacher's strength). The bridge between supervised metrics and
game-playing Elo is uncertain, sometimes inverted, and worth measuring
directly with games.
