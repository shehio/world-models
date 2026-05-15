---
title: "Self-play — Distill then RL"
subtitle: "lc0's recipe: take the distilled prior, then iterate alphazero-style"
next: "/distillation/"
---

## the central observation

The biggest difference between this project and AlphaZero is that AZ
has **no teacher**. Its policy targets come from its own MCTS visit
counts during self-play, not from Stockfish. Once you stop needing
Stockfish to label every position, the data flywheel is self-sustaining
and the only ceiling is compute.

A single GPU can't reproduce AlphaZero from scratch (would take wall-
weeks to reach 1,500 Elo from random init). But it *can* run the
realistic path: **distill-then-RL**, which is what Leela Chess Zero
actually did to reach grandmaster level on volunteer hardware.

## the loop

1. Take the [d15 ep19 ckpt](/baseline/) as the starting prior. It scores
   1,807 Elo at 800 sims, 2,084 at 4,000 sims.
2. Self-play: agent-vs-agent games with 800-sim MCTS at every position.
3. Record `(state, π = MCTS visit distribution, z = game outcome)` for
   every position the loop kept (KataGo's PCR keeps ~25% of positions —
   the ones where MCTS ran at full depth).
4. Train the network to predict `π` (cross-entropy) and `z` (MSE on
   `[-1, +1]`).
5. The new network plays the next round of self-play. Repeat.

The loss is exactly the AlphaZero loss; the only difference from
"tabula rasa AZ" is that the starting weights aren't random.

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
| Output | `s3://…/d15-…/selfplay/net-20x256/20260515T0157Z/` |

The job manifest lives at
[`infra-eks/k8s/job-selfplay.yaml`](https://github.com/shehio/world-models/blob/main/infra-eks/k8s/job-selfplay.yaml),
the entrypoint at
[`infra-eks/entrypoint-selfplay.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/entrypoint-selfplay.sh),
the orchestrator at
[`infra-eks/run-selfplay.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/run-selfplay.sh).

## why pcr

Playout Cap Randomization is a KataGo trick worth ~+80–150 Elo at
fixed compute. The idea: most positions get cheap reduced-sim MCTS for
move selection only (don't record a training target); a random ~25%
get full-sim MCTS with Dirichlet noise *and* contribute to training.

Why this helps: at low sims, MCTS visits ≈ the network's prior, which
is not a meaningful improvement target. At high sims, MCTS produces a
sharply better policy than the prior. So spend the expensive sims on
the positions whose targets you'll actually use.

## what we expect

If the loop works at all, gains should land in the +200 to +500 Elo
range over the distilled prior. That's what Lc0's first
"distill-then-RL" iteration produced for them. If it stalls or
diverges, we have the per-iter eval data to find where.

The expensive next step after the loop succeeds is **scale-up** —
multi-GPU DDP for self-play data generation, longer time budgets,
larger replay buffers. That's the path to AZ-style numbers, but it's
weeks-to-months of compute even with good engineering. See
[`alphazero`](/alphazero/) for the full size-of-gap analysis.
