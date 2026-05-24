---
title: "What's next"
subtitle: "distill-then-RL — lc0's recipe, running on an EKS pod right now"
next: "/infra/"
aliases:
  - /selfplay/
---

## The Central Observation

The biggest difference between this project and AlphaZero is that AZ
has **no teacher**. Its policy targets come from its own MCTS visit
counts during self-play, not from Stockfish. Once you stop needing
Stockfish to label every position, the data flywheel is
self-sustaining and the only ceiling is compute.

A single GPU can't reproduce AlphaZero from scratch — it would take
wall-weeks to reach 1,500 Elo from random init. But it *can* run the
realistic path: **distill-then-RL**, which is what Leela Chess Zero
actually did to reach grandmaster level on volunteer hardware.

## The Loop

1. Take the [d15 ep20 ckpt](/experiments/#baseline) as the starting prior
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

## What's Running Now

| | value |
|---|---|
| Cluster | `wm-chess-selfplay` (eksctl, us-east-1) |
| Node | g6.8xlarge — 32 vCPU + 1× L4 + 128 GiB |
| Workers | 24 (one MCTS process per vCPU; trainer uses the GPU) |
| Network | 20×256 (~23.7M params, same as baseline) |
| Sims | 800 full + 160 reduced (KataGo PCR, p_full=0.25) |
| Train steps / iter | 200 SGD steps with Adam, lr=1e-3 |
| Time budget | 12 hours |
| Prior | d15 ep 20 distilled checkpoint |

Job manifest:
[`infra-eks/k8s/job-selfplay.yaml`](https://github.com/shehio/world-models/blob/main/infra-eks/k8s/job-selfplay.yaml).
Entrypoint:
[`infra-eks/entrypoint-selfplay.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/entrypoint-selfplay.sh).
Orchestrator:
[`infra-eks/run-selfplay.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/run-selfplay.sh).

## Why PCR

Playout Cap Randomization is a KataGo trick worth ~+80–150 Elo at
fixed compute. The idea: most positions get cheap reduced-sim MCTS
for move selection only (don't record a training target); a random
~25% get full-sim MCTS with Dirichlet noise *and* contribute to
training.

Why this helps: at low sims, MCTS visits ≈ the network's prior,
which is not a meaningful improvement target. At high sims, MCTS
produces a sharply better policy than the prior. So spend the
expensive sims on the positions whose targets you'll actually use.

## What We Expect

If the loop works at all, gains should land in the +200 to +500 Elo
range over the distilled prior. That's what Lc0's first
"distill-then-RL" iteration produced for them. If it stalls or
diverges, we have the per-iter eval data to find where.

## Also In Flight — d15 at Full Data Scale

The data ablation said +199 Elo from a 6× larger dataset *with a weaker
teacher* (d10). The natural next step was to combine the strongest
teacher (d15) with the same data scale.

**Update 2026-05-24** — datagen done (426K games / 45.9M positions,
1.5× the d10-30M corpus). Two training variants are now running in
parallel:

- **R1 (40×256, LR=1e-3)** — us-east-1 spot g6e.8xlarge, 40 epochs.
  Currently at epoch 12, best ckpt so far is ep 7 at
  **2,146 Elo (sims=4,000, UCI=1,800)** — within 43 Elo of the d10
  peak (2,189) and climbing.
- **R2 (20×256, LR=5e-4, weight_decay=1e-3)** — eu-central-1 OD
  g6e.8xlarge, 30 epochs. Currently at epoch 13. The
  regularization-leaning variant we're using to test "is the
  capacity rejection from the 5M baseline still right at full data?"

Per-ckpt sims=800 evals fire automatically via the autoeval daemon;
ckpts crossing 1,940 trigger a sims=4,000 deep-read via the
[`wm-deep-eval-daemon`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-deep-eval-daemon.sh).
Full per-epoch numbers and trajectory on the
[experiments page](/experiments/#d15-46m).

Headline question: does d15 ep 15 (or 20) at sims=4,000 break past
d10's 2,189? Answer lands inside the next ~24h.

## What Else Is on the Roadmap

Ranked by expected information per unit of effort:

1. **Distill-then-self-play** — running now.
2. **d15 at full data scale** — datagen running now (see above).
3. **Eval-sims sweep** — same d15 ep 20 checkpoint, eval at 200 / 800 /
   2,000 / 4,000 / 8,000 / 16,000 sims. ~6 h on one g6.4xlarge. The
   curve from 1,807 → 2,084 at 800 → 4,000 sims says there's more
   here.
4. **Sharper soft targets** (T=0.3 instead of T=1) — regenerate
   labels from the existing PGNs, retrain. Tests whether T=1 was too
   smooth.
5. **Stronger teacher** (Stockfish d20 / d25) — raises the ceiling
   but cost scales steeply.
6. **Compute scale-up** — multi-GPU DDP, multi-week self-play. The
   only path to AlphaZero-style numbers, realistic only as a
   follow-on.

## What We're Explicitly NOT Going to Do

- ~~**Bigger net.** The capacity ablation rejected this — 40×256 =
  20×256 = ~1,810 Elo. Network isn't the bottleneck.~~ *Update
  2026-05-24: re-testing at full data scale. R1 (40×256 on 46M
  positions) is at ep 7 ≈ 2,146 sims=4,000, well above the
  5M-data baseline. The capacity rejection may have been
  data-bound, not architecture-bound. Will update when R1 finishes.*
- **More epochs of the same distillation at small data.** The d15
  5M-position baseline plateaued by epoch 10. The full-46M run
  separately may plateau later — d10-30M peaked at ep 15.
- **Tabula-rasa AZ from scratch on one GPU.** d15 ep 20 is a much
  better starting point than random; no point burning weeks to
  rediscover it.
