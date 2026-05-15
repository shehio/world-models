---
title: "What's next"
subtitle: "distill-then-RL — lc0's recipe, running on an EKS pod right now"
next: "/infra/"
aliases:
  - /selfplay/
---

## the central observation

The biggest difference between this project and AlphaZero is that AZ
has **no teacher**. Its policy targets come from its own MCTS visit
counts during self-play, not from Stockfish. Once you stop needing
Stockfish to label every position, the data flywheel is
self-sustaining and the only ceiling is compute.

A single GPU can't reproduce AlphaZero from scratch — it would take
wall-weeks to reach 1,500 Elo from random init. But it *can* run the
realistic path: **distill-then-RL**, which is what Leela Chess Zero
actually did to reach grandmaster level on volunteer hardware.

## the loop

1. Take the [d15 ep19 ckpt](/findings/) as the starting prior
   (1,807 Elo at 800 sims, 2,084 at 4,000 sims).
2. Self-play: agent-vs-agent games with 800-sim MCTS at every
   position.
3. Record `(state, π = MCTS visit distribution, z = game outcome)`
   for every position the loop kept (KataGo's playout-cap
   randomization keeps ~25% of positions — the ones where MCTS ran
   at full depth).
4. Train the network to predict `π` (cross-entropy) and `z` (MSE on
   `[-1, +1]`).
5. The new network plays the next round of self-play. Repeat.

The loss is exactly the AlphaZero loss; the only difference from
"tabula-rasa AZ" is that the starting weights aren't random.

## what's running now

| | value |
|---|---|
| Cluster | `wm-chess-selfplay` (eksctl, us-east-1) |
| Node | g6.8xlarge — 32 vCPU + 1× L4 + 128 GiB |
| Workers | 24 (one MCTS process per vCPU; trainer uses the GPU) |
| Network | 20×256 (~23.7M params, same as baseline) |
| Sims | 800 full + 160 reduced (KataGo PCR, p_full=0.25) |
| Train steps / iter | 200 SGD steps with Adam, lr=1e-3 |
| Time budget | 12 hours |
| Prior | d15 ep19 distilled checkpoint |

Job manifest:
[`infra-eks/k8s/job-selfplay.yaml`](https://github.com/shehio/world-models/blob/main/infra-eks/k8s/job-selfplay.yaml).
Entrypoint:
[`infra-eks/entrypoint-selfplay.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/entrypoint-selfplay.sh).
Orchestrator:
[`infra-eks/run-selfplay.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/run-selfplay.sh).

## why PCR

Playout Cap Randomization is a KataGo trick worth ~+80–150 Elo at
fixed compute. The idea: most positions get cheap reduced-sim MCTS
for move selection only (don't record a training target); a random
~25% get full-sim MCTS with Dirichlet noise *and* contribute to
training.

Why this helps: at low sims, MCTS visits ≈ the network's prior,
which is not a meaningful improvement target. At high sims, MCTS
produces a sharply better policy than the prior. So spend the
expensive sims on the positions whose targets you'll actually use.

## what we expect

If the loop works at all, gains should land in the +200 to +500 Elo
range over the distilled prior. That's what Lc0's first
"distill-then-RL" iteration produced for them. If it stalls or
diverges, we have the per-iter eval data to find where.

## what else is on the roadmap

Ranked by expected information per unit of effort:

1. **Distill-then-self-play** — running now.
2. **Eval-sims sweep** — same d15 ep19 checkpoint, eval at 200 / 800 /
   2,000 / 4,000 / 8,000 / 16,000 sims. ~6 h on one g6.4xlarge. The
   curve from 1,807 → 2,084 at 800 → 4,000 sims says there's more
   here.
3. **Sharper soft targets** (T=0.3 instead of T=1) — regenerate
   labels from the existing PGNs, retrain. Tests whether T=1 was too
   smooth.
4. **Stronger teacher** (Stockfish d20 / d25) — raises the ceiling
   but cost scales steeply.
5. **Compute scale-up** — multi-GPU DDP, multi-week self-play. The
   only path to AlphaZero-style numbers, realistic only as a
   follow-on.

## what we're explicitly NOT going to do

- **Bigger net.** The capacity ablation rejected this — 40×256 =
  20×256 = ~1,810 Elo. Network isn't the bottleneck.
- **More epochs of the same distillation.** The d15 baseline
  plateaued by epoch 10.
- **Tabula-rasa AZ from scratch on one GPU.** d15 ep19 is a much
  better starting point than random; no point burning weeks to
  rediscover it.
