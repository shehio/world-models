---
title: "Search — Experiment E"
subtitle: "is the ceiling an eval-side under-read?"
next: "/selfplay/"
---

## hypothesis

The [baseline](/baseline/) is evaluated with 800 MCTS sims/move. AlphaZero
used 800 at *training* time; at competitive inference it used many tens
of thousands. Maybe the model is stronger than the eval shows and we
just aren't reading it deeply enough.

## setup

Take the **same checkpoint** (d15 20×256 epoch 19, 1,807 Elo at 800
sims), re-evaluate at `--sims 4000` instead of 800. Single g6.4xlarge
in us-east-1 via
[`infra-eks/launchers/eval-deep-sims.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/eval-deep-sims.sh).
No retraining, just deeper search at inference. 104 games, 93.6 min wall.

## results

| sims / move | W / D / L vs UCI=1350 | score | Elo |
|---:|---:|---:|---:|
| 800 (baseline) | 93 / 8 / 3 | 0.933 | **1,807** |
| **4,000** | **101 / 3 / 0** | **0.986** | **2,084** |

**+277 Elo from search alone.** The model is significantly *stronger*
than the routine 800-sim eval was revealing. At 4,000 sims it sweeps
every game except 3 draws — the score saturation means we're now
hitting the ceiling of *the opponent's* strength rather than the
agent's.

## what this confirms

Distilled priors get a large boost from deeper inference-time search —
exactly what Lc0 demonstrated empirically. The intuition: the network's
policy *prior* is good enough to guide MCTS toward strong moves, and
more rollouts mean MCTS gets to refine that prior into a sharper visit-
count distribution. The student becomes a better chess player not by
learning more, but by *thinking longer*.

The natural follow-up is the **sims sweep**: same checkpoint, eval at
200 / 800 / 2,000 / 4,000 / 8,000 / 16,000 sims. The Elo-vs-log(sims)
curve will reveal where the saturation point actually is. We have the
machinery in [`elo-bisect.py`](/elo-bisect/) — it's still on the
roadmap.

## practical consequence

The 1,807 / 2,084 split changes how to interpret the other experiments:
- The "1,800 ceiling" we report from routine evals is a *floor*, not a
  ceiling. There's another ~280 Elo trivially available from deeper
  eval search.
- For honest comparison with AlphaZero, the relevant number is the
  search-saturated one. AlphaZero's published Elo is also at peak
  inference settings, not the training-time 800 sims.
