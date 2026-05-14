# Eval pipeline & reading the results

How the auto-eval daemon measures the model's Elo against Stockfish, what the
numbers mean, and how to pin Elo precisely once a model gets strong.

## How a checkpoint is evaluated

1. Trainer saves `distilled_epochNNN.pt` to S3 under
   `checkpoints/net-<blocks>x<filters>/<run-id>/`.
2. `wm-autoeval-daemon.sh` (in `infra-eks/daemons/`) polls every 5 min,
   finds the new `.pt`, writes a `.claimed-eval-<ckpt>` marker, and
   launches a one-shot EC2.
3. The EC2 pulls the GPU image, downloads the `.pt`, runs `eval.py`
   twice (vs two Stockfish settings — see below), uploads
   `eval_results-<ckpt>.txt` next to the checkpoint, and self-terminates.

The daemon parses `net-NxF` out of the checkpoint path and passes the right
`--n-blocks` / `--n-filters` to `eval.py`, so 20×256 and 40×256 networks both
work transparently.

Each routine eval plays **100 games** (8 worker processes × 13 games each)
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
  `-400 × log10(1/score - 1)`. Positive means our agent is *stronger*
  than the opponent.
- **Agent absolute Elo** — `opponent_elo + elo_gap`. Only meaningful
  if the opponent has a calibrated Elo (the UCI=1350 case below).

## What `UCI=N` actually means

UCI is the Universal Chess Interface — the protocol Stockfish speaks.
It has a calibrated weakening mode:

- `UCI_LimitStrength = true`
- `UCI_Elo = N` ← target playing strength

Stockfish 17 plays at *roughly* `N` Elo when this is set. It still uses
its full evaluation engine but introduces blunders proportional to a
strength target validated against community-tested matches.

So Stockfish-N is a **calibrated opponent**: if our model scores 0.5
against it, the model ≈ N Elo. The build accepts UCI_Elo in
[1350, 3190] (the older 1320 minimum got bumped to 1350 in newer
Stockfish releases — which is why our first eval launch crashed before
we patched the daemon).

## The two sub-tests the daemon runs

1. **Stockfish UCI=1350** — the anchor. Always the same, so all
   checkpoints are comparable on a single scale. The reported "absolute
   Elo" comes from this match.
2. **Stockfish UCI=1800** — a stronger calibrated opponent, close to
   where our 20×256 d15 model sits (~1750-1800 Elo). When the model is
   well-matched against this, the CI is tightest and the gap converts
   to an accurate Elo. Replaced the earlier `depth=1` sub-test, which
   produced a lot of draws and wasn't on the UCI Elo scale.

When the model passes 1800 meaningfully, bump the second anchor (e.g.
to 2000) so the win-rate stays near 50%.

## Pinning Elo precisely: `elo_bisect.py`

`experiments/distill-soft/scripts/elo_bisect.py` is a binary-search wrapper
around `eval.py`. It probes Stockfish at successive UCI_Elo levels and
halves the bracket each round until either the bracket is ≤100 Elo or a
probe lands in [0.45, 0.55] (the score range where the implied Elo is most
trustworthy). Useful to convert a "model is around 1800" estimate into a
specific number with a tight CI.

Usage:

```
python scripts/elo_bisect.py \
    --ckpt /work-tmp/distilled_epoch019.pt \
    --workers 8 --games-per-probe 40 \
    --sims 800 --n-blocks 20 --n-filters 256 \
    --out /work-tmp/bisect.json
```

Three to four probes is typical (~30-40 min on g6.4xlarge).

## Multi-region launch capability

Eval EC2s burn G/VT vCPU, which AWS quota-limits to 32 per region. Routine
training (g6e on L40S) also burns from the same quota. The daemon walks
`REGION_ORDER=(us-east-1 eu-central-1)`: tries us-east-1 first, falls back
to eu-central-1 if launch fails (e.g. `VcpuLimitExceeded`). The two regions
have independent quotas, so this roughly doubles concurrent-eval capacity.

Cross-region wiring is invisible to the workload: ECR pulls and S3
reads/writes target us-east-1 regardless of where the EC2 lives.

## Idempotency

For each checkpoint the daemon checks two markers in S3:

- `eval_results-<ckpt>.txt` — the result file. If present, skip.
- `.claimed-eval-<ckpt>` — a claim marker (so two pollers don't
  double-launch). Removed automatically if the launch ultimately fails
  in every region.

Per-experiment one-off evals (e.g. `--sims 4000` on the same ckpt) write to
a distinct filename so the idempotency check still works.

## Speeding eval up

What we did:
- **Bumped eval EC2 from `g6.xlarge` (4 vCPU) → `g6.4xlarge` (16 vCPU).**
  8 workers × 13 games in parallel instead of 4 × 25 → roughly 3× faster
  wallclock per ckpt.
- **Multi-region fallback** doubles concurrent-eval throughput by using
  eu-central-1's separate quota.

Other knobs (not enabled by default, kept here as a menu):

- **Drop the UCI=1800 sub-test** — saves ~50% of eval time. The
  UCI=1350 anchor is what we report.
- **MCTS sims 800 → 400** — halves agent compute per move. Slightly
  weaker play but consistent across all ckpts, so relative comparisons
  are unaffected.
- **Games 100 → 50** — widens CI (±100 Elo instead of ±70) but halves
  eval time.
- **bf16 autocast in `eval.py`** — code change. `--agent-device cuda`
  currently runs the NN in fp32. Adding autocast would give ~2× on the
  NN portion.
- **Batched MCTS / virtual loss** — proper code change. Could give
  3-5× by batching leaf-node NN inferences across MCTS rollouts.
- **Persistent eval pod on the EKS cluster** — skip the EC2 cold-start
  cost (~5 min/ckpt) by keeping a warm container that pulls work
  off an SQS queue.
