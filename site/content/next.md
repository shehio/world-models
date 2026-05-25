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

## The 2,500 Ceiling — What's in the Way and Why

After the d15 46M experiments tied d10 30M at sims=4,000 (and the
direct H2H drew 104/104), we have enough data to identify the actual
bottleneck and rank the levers honestly.

**Verdict: pure distillation has saturated near ~2,150–2,200 Elo.**

Adding more teacher quality (d15 vs d10) and more data (46M vs 30M)
bought zero Elo. The d15 R1 plateau at constant LR=1e-3 is being
re-tested with cosine (R1 v2 + R2 cosine) but the H2H tie suggests
recipe tweaks won't close 350 Elo to 2,500. **The teacher *itself* is
~2,500 — pure SL can asymptotically match it, not exceed.**

### Levers ranked by Elo per dollar

| # | Lever | Estimated Δ Elo | Effort | Status |
|---|---|---|---|---|
| 1 | **Self-play RL** on top of distilled prior (the Lc0 recipe) | +200 to +500+ | Weeks of GPU + careful LR tuning | Code exists in [`experiments/selfplay/`](https://github.com/shehio/world-models/tree/main/experiments/selfplay); a first attempt regressed (LR was 100× too high). Re-fire incoming. |
| 2 | **Higher eval sim count** (sims=16,000+) | +100 to +200 | $0 retraining, ~4× per-game time at eval | Untested above sims=4,000 |
| 3 | **Sharper teacher targets** (multipv=1 hard, or T=0.3 soft) | +50 to +150 | One retraining run (~$80) | Tested at small scale (250K data); 02b's hard targets beat 02c's soft targets there. Not tested at full data scale. |
| 4 | More data (100M+ positions) | Probably +0 to +50 (saturating) | New 200K+ game datagen + 60h training (~$300) | The 30M→46M step gave 0 Elo; data scale looks saturated |
| 5 | Bigger net (60×384, ~100M params) | Probably +0 to +50 | Major training cost (~$400) | Capacity rejected at 5M; partially re-tested at 46M (R1 40×256 = same as R2 20×256) |
| 6 | Stronger teacher (Stockfish d20+) | Uncertain (+30 to +100 maybe) | New datagen at higher depth (~$500 — d depth doubles datagen cost) | Untested |

**Lever #1 (self-play) is the only one in this list that can plausibly
hit 2,500.** Everything else is incremental. Self-play replaces "match
the teacher's policy" with "find moves the teacher missed via search,
then train on those" — which is *also* what got Leela from ~2,500 to
~3,600 over years of community compute.

### Order of operations

1. Finish R1 v2 + R2 cosine cosine experiments (in flight). Even if
   they only confirm "constant LR was a small win at best," we close
   the recipe-ablation chapter.
2. Run an eval-sim sweep on the best ckpt across sims = {800, 2000,
   4000, 8000, 16000, 32000}. Free Elo if sims=16,000 holds the
   trend; tells us the search-side ceiling.
3. **Wire up self-play on top of the strongest distilled prior.** First
   pass: 24h on a single g6e.8xlarge, LR=1e-5 (the regressed-first-attempt
   used 1e-3, now fixed). If it doesn't regress *or* improve, the
   distill ceiling is the project ceiling and we accept ~2,200 as the
   answer. If it improves cleanly, scale up.

### What we're NOT going to do (and why)

- **More data alone.** 30M → 46M gave 0 Elo. Returns saturated.
- **Tabula-rasa AlphaZero from scratch on one GPU.** d15 ep 20 (or
  d10 ep 15) is a much better starting prior than random; the
  distill-then-RL recipe was always the realistic path.
- **Bigger network alone.** R1 (40×256) tied R2 (20×256) in direct play.
  Pure capacity was already rejected.

