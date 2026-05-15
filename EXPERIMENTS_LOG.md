# Experiments log — distill-soft, May 2026

Running log of the ablation experiments on the d15/d10 multi-pv-soft
distillation pipeline. Drafted for a blog post; numbers update as
checkpoints land.

## TL;DR (so far)

| | d15 20×256 | A (d15, 40×256) | E (d15 ep19, sims=4000) | C (d10 full 30M) |
|---|---:|---:|---:|---:|
| Best Elo (UCI=1350 anchor, 800 sims) | **1807** | 1851 | **2084** | running |
| Best Elo (UCI=1800 anchor, 800 sims) | n/a | **1813** | n/a | running |
| Verdict | baseline | no gain → not capacity-limited | **+277 Elo from 5× more search at eval** | tests data-limited hypothesis |

The 1800-Elo plateau we saw on the 20×256 baseline is **(a) not
capacity-limited** (A confirms) and **(b) partly an eval-search
artifact** (E confirms: same checkpoint reads ~+280 Elo stronger at
4000 sims). Whether it's data-limited is what C is currently asking.

## How we're different from AlphaZero

| | AlphaZero (2017) | This repo |
|---|---|---|
| Approach | Tabula-rasa self-play RL | Supervised distillation from Stockfish |
| Network | 20-block ResNet, 256 filters | Identical (20×256) |
| MCTS at training | 800 sims/move on agent's own search | None — Stockfish provides targets |
| Training positions | ~3.5B (44M games × 80 moves) | 5M–30M |
| Training compute | 64 TPU-v2 × 9h + ~45k TPU-v1-hr self-play | 1 GPU × 5h ≈ 5 GPU-hours (~10,000× less) |
| Eval MCTS sims | 800 training / many tens of thousands tournament | 800 routine / 4000 on Experiment E |
| Teacher | None | Stockfish d10 (~2200) or d15 (~2500), multipv=8, T=1 |
| Peak Elo | ~3500–3600 | 2084 (d15 ep19 @ 4000 sims vs UCI=1350) |

The architecture (20×256 ResNet) is *literally identical* to AlphaZero's.
Everything that's different is on the **training-procedure** side:
distillation vs self-play, 600× less data, 10 000× less compute, and an
external teacher providing the policy target instead of MCTS.

That's the point of this set of experiments — hold the network fixed,
vary only the training procedure, and measure where each path tops out.
See [DISTILLATION_VS_ALPHAZERO.md](./DISTILLATION_VS_ALPHAZERO.md) for the
conceptual framing.

## Baseline: d15 20×256, 5 M positions

The reference run. d15 teacher (Stockfish at search-depth 15 ≈ 2500 Elo),
multipv=8 soft targets, 20-block × 256-channel ResNet.

| epoch | UCI=1350 Elo | W/D/L vs UCI=1350 | UCI=1800 Elo (sub-test before May 14) |
|---:|---:|---:|---:|
| 4 | 1701 | 62 / 35 / 3 | n/a |
| 9 | 1693 | 62 / 36 / 2 | n/a |
| 14 | 1759 | 91 / 8 / 5 | n/a |
| **19** | **1807** | 93 / 8 / 3 | n/a |

Loss plateaus around 2.59 by ep 10; top-1 plateaus ~0.34. Gains per epoch
shrink to noise. This is the "1800 ceiling" we're trying to break.

## Experiment A — capacity-limited?

**Hypothesis:** the 1800 ceiling is the 20-block × 256-channel network
running out of representation capacity. Doubling the depth should help.

**Setup:** 40 × 256 (≈48 M params vs 24 M), same data (5 M d15 positions),
same training schedule. Single L40S in eu-central-1 via
[`infra-eks/launchers/d15-40x256-eu.sh`](./infra-eks/launchers/d15-40x256-eu.sh).
13.5 min/epoch.

| epoch | UCI=1350 Elo | UCI=1800 Elo | W/D/L vs UCI=1800 |
|---:|---:|---:|---:|
| 4 | 1851 | **1813** | 37 / 34 / 33 |
| 9 | 1851 | 1780 | 26 / 46 / 32 |
| 14 | 1759 | 1722 | 22 / 37 / 45 |
| 19 | 1820 | **1810** | 32 / 43 / 29 |

**Verdict — hypothesis rejected.** The UCI=1800 anchor (where the
score sits near 0.5 and the CI is tightest) shows the 40×256 net lands
at the same ~1810 as the 20×256 baseline. **More parameters don't
help on this dataset.** Either the data is the bottleneck or the
training procedure is.

## Experiment E — under-read at eval?

**Hypothesis:** the daily eval runs at 800 MCTS sims/move but the model
might be stronger than 800 sims can demonstrate. AlphaZero used 800 at
*training* time; tournament inference used many tens of thousands.

**Setup:** same checkpoint (d15 20×256 ep19), eval re-run at
`--sims 4000` instead of 800. Single g6.4xlarge in us-east-1 via
[`infra-eks/launchers/eval-deep-sims.sh`](./infra-eks/launchers/eval-deep-sims.sh).
93.6 min wallclock for 104 games.

| sims/move | W/D/L vs UCI=1350 | score | Elo |
|---:|---:|---:|---:|
| 800 (baseline) | 93 / 8 / 3 | 0.933 | 1807 |
| **4 000** | **101 / 3 / 0** | **0.986** | **2084** |

**+277 Elo from search alone, no retraining.** The model knew more
than the 800-sim eval was revealing. This is exactly what Lc0
demonstrated empirically: distilled priors get a large boost from
deeper inference-time search.

A natural follow-up is the *function of Stockfishes* — sweep eval-side
sims from 200 → 8000 and report the Elo curve. That's what
[`elo_bisect.py`](./experiments/distill-soft/scripts/elo_bisect.py)
is built for.

## Experiment C — data-limited?

**Hypothesis:** 5 M positions is too small for a 24 M-param network.
The full d10 dataset has ~30 M positions; using all of it should help
*if* the ceiling is data-driven.

**Setup:** 20 × 256, 30 M positions (no `MAX_POSITIONS` truncation),
`BATCH_SIZE=2048` to keep iteration count reasonable, in-RAM via the
256 GB on g6e.8xlarge in eu-central-1 via
[`infra-eks/launchers/d10-full30m.sh`](./infra-eks/launchers/d10-full30m.sh).

| epoch | loss | top-1 | epoch_time (s) |
|---:|---:|---:|---:|
| 0 | 2.843 | 0.308 | 2 936 |
| ... | running | running | running |

**ETA:** 49 min/epoch × 20 = ~16 h total → first eval ckpt (ep4) around
**03:30 UTC May 15**; full run around **15:30 UTC May 15**.

### Why C is slow vs A (per-batch math)

Same L40S GPU in both cases. The wall-clock gap comes from raw compute:

| | A (40×256, 5M) | C (20×256, 30M) | C / A |
|---|---:|---:|---:|
| Batch size | 512 | 2 048 | 4× |
| Compute per batch ∝ blocks × batch | 20 480 | 40 960 | 2× |
| Batches per epoch | 9 766 | 14 648 | 1.5× |
| Expected epoch time vs A | — | — | **3×** |
| Actual epoch time | 13.5 min | 49 min | **3.6×** |

A 6× larger dataset on a half-depth network at 4× the batch comes out
to 3× more total compute per epoch on the same hardware. The extra
~20% is OS page-cache cold start on ep 0 (15 GB of memmapped files
need to page in) + DataLoader overhead from streaming the bigger
dataset. Later epochs should land closer to ~40 min.

## Roadmap to close the AlphaZero gap

The 20×256 ResNet is literally AZ's network. Everything that's different
is on the *training-procedure* side. Ordered by expected information
per unit of effort:

### 1. Distill-then-self-play (the big one — Lc0's recipe)

**The defining difference between us and AZ is that AZ doesn't have a
teacher.** Its policy targets come from its own MCTS visit counts
during self-play, not from Stockfish. Once you stop needing Stockfish
to label every position, the data flywheel becomes self-sustaining and
the only ceiling is compute.

The realistic path for a small lab is **distill-then-RL**, which is
what Lc0 actually did:

1. Take the d15 ep19 ckpt as the starting prior (we have this).
2. Run self-play: agent vs agent with 800-sim MCTS at every position.
3. Record `(state, π = MCTS visit distribution, z = game outcome)`.
4. Train the network to predict π (cross-entropy) and z (MSE).
5. Repeat with the newer net playing newer games.

We already have all the pieces — `wm_chess/src/wm_chess/mcts.py`, the
20×256 net, and the selfplay experiment (`experiments/selfplay/`) has
v1-v5 of an end-to-end self-play loop. The work is wiring the
distilled ckpt in as the starting prior and pointing the self-play
loop at the larger network. Single-GPU at 800 sims/move generates
maybe 200-400 games/hour; a weekend run gets ~10-20k self-play games.
That's tiny compared to AZ's 44M but is enough to test whether the
loop helps at all.

Expected gain: **+200-500 Elo if the loop works.** This is where Lc0
got most of its strength.

### 2. Eval-sims sweep (free Elo, no retraining)

Experiment E showed +277 Elo on the same checkpoint just from going
800 → 4 000 sims at eval. Where does that curve flatten? Run the same
ckpt at 200 / 800 / 2 000 / 4 000 / 8 000 / 16 000 sims, plot Elo vs
log(sims). That tells us:

- How much of the "1800 ceiling" was eval-side undermeasurement.
- What the *true* search-saturated Elo of d15 ep19 is.
- Where to set the eval budget for the next round of experiments.

[`elo_bisect.py`](./experiments/distill-soft/scripts/elo_bisect.py) is
already built for this — point it at d15 ep19 with `--sims` overridden
per probe. ~6h on a single g6.4xlarge.

### 3. Sharper soft targets (T=0.3 instead of T=1.0)

`multipv=8` with `T=1` pawn gives target distributions with average
entropy ~1.5 bits — closer to "uniform over 4 candidates" than "this
move is best." The 02c-30ep negative result analysis pinned this as a
likely cause of the *hedging* failure mode (top-K=87%, top-1=36%).

Concrete: regenerate the labels from the existing PGNs with `T=0.3`,
retrain. No new self-play needed. Tests whether soft targets per se
are bad, or T=1 specifically is too smooth.

Expected gain: **+50-150 Elo if T=1 was the bottleneck.**

### 4. Stronger / fresher teacher

`d15` ≈ 2500 Elo. Bumping to `d20` (~2700) or `d25` (~2900) raises the
ceiling, but the cost scales with branching factor ^ depth — d20 is
~5× the wallclock per move vs d15. Worth it only if (1) and (3) tap out.

### 5. Compute scale-up (brute-force AZ)

The mechanical path: add a second GPU and a DDP wrapper, run the
self-play loop for weeks, scale data 100×. This is what would
*actually* close the gap to AZ-style numbers. Realistic only as a
follow-on.

### What we are NOT going to do (and why)

- **Bigger network.** Experiment A already showed 40×256 buys nothing
  on this data. The network isn't the bottleneck.
- **More distillation epochs of the same recipe.** d15 plateaued by
  ep10. More of the same isn't going to move it.
- **Tabula-rasa AZ from scratch.** Without distilled initialization,
  a single-GPU AZ run takes wall-weeks to reach 1500 Elo. d15 ep19 is
  a much better starting point than random.

## Pending tracker (will update as runs land)

- [x] Experiment A — done (1810 Elo, capacity hypothesis rejected)
- [x] Experiment E — done (2084 Elo @ 4000 sims, +277 from search)
- [ ] Experiment C — running (ep0 done, ETA 15:30Z May 15)
- [ ] Sims sweep on d15 ep19 (200 → 16000) — not yet started
- [ ] Self-play loop initialized from d15 ep19 — not yet started
- [ ] T=0.3 ablation — not yet started

## Where to find the artifacts

- Checkpoints: `s3://wm-chess-library-594561963943/<prefix>/checkpoints/net-NxF/<run-id>/`
- Eval results: `eval_results-<ckpt>.txt` siblings of each ckpt
- Launchers used: [`infra-eks/launchers/`](./infra-eks/launchers/)
- Multi-region auto-eval daemon: [`infra-eks/daemons/wm-autoeval-daemon.sh`](./infra-eks/daemons/wm-autoeval-daemon.sh)
- Eval pipeline doc: [`EVALS.md`](./EVALS.md)

## Last updated

2026-05-15 01:04 UTC — A complete, E complete, C ep 0 done.
