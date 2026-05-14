# Eval pipeline & reading the results

This doc covers how the auto-eval daemon measures the model's Elo against
Stockfish, what the numbers mean, and the knobs available to make eval
faster.

## How a checkpoint is evaluated

1. Trainer saves `distilled_epochNNN.pt` to S3.
2. `wm-autoeval-daemon.sh` (in `infra-eks/daemons/`) polls every 5 min,
   finds the new `.pt`, writes a `.claimed-eval-<ckpt>` marker, and
   launches a one-shot EC2.
3. The EC2 pulls the GPU image, downloads the `.pt`, runs `eval.py`
   twice (vs two Stockfish settings), uploads
   `eval_results-<ckpt>.txt` next to the checkpoint, and self-terminates.

Each eval today plays **100 games** (8 worker processes × 13 games each)
with the agent doing **800 MCTS sims per move**.

## Reading the result file

```
=== eval vs Stockfish UCI=1350 ===
...
=== 100 games in 48.8 min ===
W/D/L: 89 / 9 / 2
score: 0.935   95% CI: [0.887, 0.983]
Elo gap to opponent: +463
Agent absolute Elo (anchor 1350): 1813 [1707, 2058]
```

- **W/D/L** — wins / draws / losses for our agent.
- **score** — wins + 0.5 × draws, all over games played.
- **95% CI** — confidence interval on the *score* (not the Elo). With
  100 games the CI is roughly ±10 percentage points around the point
  estimate. More games → tighter CI.
- **Elo gap to opponent** — converted from score via
  `-400 × log10(1/score - 1)`. The sign convention: positive means our
  agent is *stronger* than the opponent.
- **Agent absolute Elo** — `opponent_elo + elo_gap`. Only meaningful
  if the opponent has a calibrated Elo (the UCI=1350 case below).

## What `UCI=1350` actually means

UCI is the Universal Chess Interface — the protocol Stockfish speaks.
It has a calibrated weakening mode:

- `UCI_LimitStrength = true`
- `UCI_Elo = 1350` ← target playing strength

Stockfish 17 (the version baked into our GPU image) plays at *roughly*
1350 Elo when this is set. It still uses its full evaluation engine but
deliberately introduces blunders proportional to a strength target
validated against community-tested matches.

So Stockfish-1350 is a **calibrated opponent**:

- If our model scores 0.5 against it → model ≈ 1350 Elo.
- If our model scores 0.935 (89-9-2, like d15 ep4) → model is meaningfully
  stronger. Plug in: `1350 + 400·log10(0.935/0.065) = 1350 + 463 = 1813 Elo`.

Stockfish exposes UCI_Elo from 1320 to 3190 in our build (note: the
older 1320 minimum got bumped to 1350 in newer Stockfish releases —
which is why our first eval launch crashed before we patched the
daemon). When the model gets meaningfully stronger than 1813, we should
bump the anchor (e.g., to 1800) so the win-rate stays near 50% and the
CI tightens.

## The `UCI=-1 depth=1` sub-test

The second eval sets `UCI_LimitStrength=false` (top skill) but caps
search depth to 1 ply. The result is a Stockfish that uses its full
evaluation function but can only look one move ahead. It's good at
choosing positionally-sound moves but blunders tactically.

Result interpretation: against this opponent, our model tends to draw
a lot in early ckpts (both sides play "solid" moves and end up in
known drawish lines) and wins start showing up as the model gains
tactical understanding. The first calibrated number is the UCI=1350
one; the depth=1 number is a sanity check.

## Speeding eval up

What we did:
- **Bumped eval EC2 from `g6.xlarge` (4 vCPU) → `g6.4xlarge` (16 vCPU).**
  Lets us run 8 workers × 13 games in parallel instead of 4 × 25 →
  roughly 3× faster wallclock per ckpt.

Other knobs (not currently enabled, kept here as a menu):

- **Drop the depth-1 sub-test** — saves ~40% of eval time. The
  calibrated UCI=1350 number is what we report; depth=1 is a sanity
  check.
- **MCTS sims 800 → 400** — halves agent compute per move. Slightly
  weaker play but consistent across all ckpts, so relative comparisons
  are unaffected.
- **Games 100 → 50** — widens CI (±100 Elo instead of ±70) but halves
  eval time.
- **bf16 autocast in `eval.py`** — code change. Currently
  `--agent-device cuda` runs the agent NN in fp32. Adding autocast
  would give ~2× on the NN portion.
- **Batched MCTS / virtual loss** — proper code change. Could give
  3-5× by batching leaf-node NN inferences across MCTS rollouts.
- **Persistent eval pod on the EKS cluster** — skip the EC2 cold-start
  cost (~5 min/ckpt) by keeping a warm container that pulls work
  off an SQS queue.
