---
title: "Eval pipeline"
subtitle: "calibrated stockfish opponent · multi-region auto-eval · daemon-managed"
next: "/elo-bisect/"
---

## the question

Every research repo invents its own definition of "this model is X Elo
strong." Ours is the most boring: play 100 games against a Stockfish
opponent set to a calibrated `UCI_Elo` and report the win rate.
Stockfish 17 implements the `UCI_LimitStrength = true` /
`UCI_Elo = N` interface — it plays at *roughly* `N` Elo (validated
against community matches), still using its full evaluation engine
but deliberately introducing blunders proportional to the strength
target.

Stockfish at `UCI_Elo = 1350` is a *calibrated* opponent. If our
model scores 0.5 against it → model ≈ 1,350 Elo. If our model
scores 0.93 → plug into the Elo formula:

`gap = -400 · log10(1/score - 1)`

`1350 + 457 = 1807 Elo`.

## the auto-eval daemon

[`infra-eks/daemons/wm-autoeval-daemon.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-autoeval-daemon.sh)
polls S3 every 5 minutes. For every new `distilled_epochNNN.pt` that
doesn't have a sibling `eval_results-<ckpt>.txt`:

1. Writes a `.claimed-eval-<ckpt>` marker (concurrent-poller safety).
2. Launches a one-shot g6.4xlarge EC2 in us-east-1.
3. Falls back to eu-central-1 if us-east hits `VcpuLimitExceeded`.
4. The EC2 pulls the GPU image, downloads the ckpt, plays 100 games
   vs `UCI=1350` and 100 vs `UCI=1800`, uploads results next to the
   ckpt, and self-terminates via
   `instance-initiated-shutdown-behavior=terminate`.

The daemon parses `net-NxF` out of the ckpt path and passes the right
`--n-blocks` / `--n-filters` to `eval.py` so 20×256 and 40×256 evals
both work transparently.

## two sub-tests per checkpoint

| sub-test | purpose |
|---|---|
| **UCI = 1,350** | the historical anchor; every checkpoint ever evaluated is comparable on this scale. Reports "agent absolute Elo (anchor 1350)". |
| **UCI = 1,800** | calibrated opponent near current model strength. CI is tightest where the score is near 0.5, so this anchor reports the *most accurate* Elo for d15-class models. |

The earlier "`depth=1` Stockfish" secondary test produced unreliable
results (high variance, drawish games, no Elo calibration) and was
replaced — see [failures](/failures/) for the story.

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
when the opponent is far from the model strength. The UCI=1800
anchor gives much tighter Elo bounds for the current generation of
checkpoints.

## multi-region capacity

AWS quota L-DB2E81BA caps G/VT vCPU at 32 per region. Two
concurrent g6.4xlarge evals (each 16 vCPU) saturate that. The daemon
walks `REGION_ORDER=(us-east-1 eu-central-1)`: first launch wins,
fallback on `VcpuLimitExceeded`. The two regions have independent
quotas → ~64 effective vCPU.

Cross-region wiring is invisible to the workload: ECR pull and S3
read/write target us-east-1 regardless of where the EC2 lives. See
[infra](/infra/) for details.

## why not just track training loss

Training loss / top-1 accuracy plateau by epoch 10, but Elo keeps
climbing through epoch 19 — the *training* objective and the
*play-strength* objective aren't perfectly aligned. Top-K accuracy
of 87% sounds excellent; the actual Elo says it's 1,807, not 2,500
(the teacher's strength). The bridge between supervised metrics and
game-playing Elo is uncertain, sometimes inverted, and worth
measuring directly with games.
