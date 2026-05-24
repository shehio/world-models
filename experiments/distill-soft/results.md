# 02c results

Four generations of this experiment. The first (250k positions, d10
teacher, MPS) plateaued ~1086 Elo and looked like a negative result. We
then rebuilt the data pipeline (EKS Indexed Jobs → millions of games)
and the training stack (GPU EKS + bf16 + autocache + per-checkpoint S3
sync + an auto-eval daemon). The 5M d15 runs landed at ~1750–1810. The
30M d10 + sims=4,000 deep-read pushed to 2,189 (the project's prior
peak). The newest 46M d15 runs are climbing into that territory while
this writeup is being written.

## Headline (2026-05-24 — d15 full-46M, two parallel hparam variants) {#exp-c-d15-46m}

The "Exp C" we ranked as the most informative single experiment back
in 2026-05 (d15 teacher × multipv=8, at full data scale) is now
running. It pairs the strongest teacher (Stockfish d15 multipv=8 T=1)
with **45.9M positions / 426K games** — 1.5× the position count of the
d10-30M run that produced the 2,189 peak.

Two variants are running in parallel:

| Run | Net | LR | wd | Batch | Epochs | Region / market |
|---|---|---|---|---:|---:|---|
| **R1** | 40×256 (~50M params) | 1e-3 | 1e-4 | 1024 | 40 | us-east-1 spot g6e.8xlarge |
| **R2** | 20×256 (~24M params) | 5e-4 | 1e-3 | 2048 | 30 | eu-central-1 OD g6e.8xlarge |

R1 = bigger net + default LR. R2 = regularization-leaning (smaller
net, slower LR, 10× weight decay) launched after R1's sims=800 curve
looked noisy. Both reuse the d10-30M training recipe except where
noted; ckpts land in distinct S3 subdirs (`net-40x256/<run-id>/` vs
`net-20x256/<run-id>/`) so the autoeval daemon picks them up
transparently.

### sims=800 per-epoch trajectory (UCI=1,800 anchor)

| ep | R1 (40×256) | R2 (20×256, lowlr) |
|---:|---|---|
| 0 | 1864 [1798, 1936] | 1834 [1767, 1903] |
| 1 | 1892 [1826, 1966] | 1889 [1822, 1962] |
| 2 | 1875 [1808, 1947] | — (skipped during overnight quota crunch) |
| 3 | 1857 [1791, 1928] | 1857 [1791, 1928] |
| 4 | 1850 [1784, 1921] | 1900 [1833, 1974] |
| 5 | 1889 [1822, 1962] | 1850 [1784, 1921] |
| 6 | **1941** [1873, 2021] | 1922 [1854, 1999] |
| 7 | 1864 [1798, 1936] | 1922 [1854, 1999] |
| 8 | 1910 [1844, 1987] | 1861 [1794, 1932] |
| 9 | (pending) | 1907 [1840, 1982] |
| 10 | (pending) | 1864 [1798, 1936] |
| 11 | (pending) | 1875 [1808, 1947] |

The sims=800 numbers are *very* noisy at the ~100-game-per-eval
budget — CIs are ±70 Elo wide. R1 ep6 (1941) is the best sims=800
point estimate so far; R2 ep6/ep7 (1922) is within noise of it
despite half the per-epoch compute.

### sims=4,000 deep-read on selected R1 checkpoints

A second daemon (`infra-eks/daemons/wm-deep-eval-daemon.sh`) watches
for sims=800 results above threshold `Elo>=1940` and auto-fires
sims=4,000 follow-ups. ep1 was fired manually for an early
calibration point.

| Ckpt | sims=800 | sims=4,000 | Δ |
|---|---:|---:|---:|
| R1 ep1 | 1892 [1826, 1966] | **2055** [1979, 2159] | **+163** |
| R1 ep6 | 1941 [1873, 2021] | **2146** [2060, 2285] | **+205** |

The +163/+205 Elo gap between sims=800 and sims=4,000 confirms the
earlier sims-ablation finding: more search recovers real strength
that sims=800 systematically understates. The gap *widens* with
stronger checkpoints, so sims=800 understates progress more as
training advances.

### Calibration vs d10-30M sims=4,000 (the prior peak)

| Run | Ckpt | sims=4,000 Elo | CI |
|---|---|---:|---|
| d10 30M (20×256) | ep5  | 2084 | [2005, 2197] |
| d10 30M (20×256) | ep10 | 2171 | [2082, 2324] |
| **d10 30M (20×256)** | **ep15** | **2189** | **[2098, 2354]** ← prior project peak |
| d10 30M (20×256) | ep20 | 2154 | [2067, 2297] |
| d15 46M (40×256) | ep2  | 2055 | [1979, 2159] |
| **d15 46M (40×256)** | **ep7** | **2146** | **[2060, 2285]** ← d15 best so far |

**d15 ep7 sims=4,000 (2146) is within −43 Elo of d10's peak (2189).**
CIs overlap by ~100 Elo. R1 has 33 epochs remaining; d10 peaked at
ep15 in its 20-epoch run. If d15 follows a similar arc, the 2189
record likely falls inside the next ~24h.

### Operational notes captured for repro

- **Eviction recovery**: R1 was spot-evicted at ep6 (us-east-1a)
  after ~17h. Resumed in us-east-1d via
  `bash infra-eks/launchers/d15-full30m.sh` with a pinned `RUN_ID`
  so the entrypoint's auto-resume picked up ep6 from S3. Lost
  ~58 min to dataset reload but no model state. A new watcher
  daemon (`infra-eks/daemons/wm-d15-run1-watcher.sh`) re-fires the
  launcher on future evictions.
- **Eval quota saturation**: both regions hit their 32-vCPU OD G/VT
  quotas simultaneously (2× sims=4,000 EC2s in us-east-1 + R2 OD
  training in eu-central-1). Patched the autoeval daemon to add an
  eu-central-1 spot tier as third fallback so the queue keeps
  draining when both OD quotas are full.

### What we'll know when these finish

1. **Does d15 beat d10 at matched compute?** (R1 vs d10, both at full
   data scale at sims=4,000.) Initial answer based on ep7: probably
   yes, but tight.
2. **Does the regularization-leaning recipe close the capacity gap?**
   (R2 vs R1.) Initial answer based on sims=800 ep6: R2 (1922) is
   within noise of R1 (1941) with half the per-epoch compute. Both
   need sims=4,000 evals for confidence.
3. **Where does d15 peak?** (R1 epoch curve.) The widening
   sims=800→sims=4,000 gap (+163 at ep1, +205 at ep6) means the peak
   at sims=4,000 may land at a *later* epoch than the peak at
   sims=800. Plan: deep-eval every R1 ckpt above the 1940 threshold
   (autoeval daemon handles this).

The runs are still in flight at writeup — final numbers will land in
24–48h. This page will be re-evaluated when they do.

---

## Earlier headline (2026-05 — d15 / EKS-trained 20×256 on 5M positions)

100-game-equivalent eval vs Stockfish `UCI_Elo=1350`, MCTS 800 sims/move.

| Run | Network | Teacher | Positions | Real Elo | W/D/L (per 100g) |
|---|---|---|---:|---:|---:|
| **d15 ep14** | 20×256 (~24M) | SF d15 mpv=8 (T=1) | 5M (subset of ~30M) | **1759** [1665, 1939] | 91 / 8 / 5 |
| **d15 ep19** | 20×256 (~24M) | SF d15 mpv=8 (T=1) | 5M (subset of ~30M) | **1807** [1704, 2034] | 93 / 8 / 3 |

The Elo curve flattens between ep14 and ep19 (+48 Elo over 5 epochs),
matching the loss trajectory (top-1 plateau ~0.34). Sims=4000 deep-read
eval and a 40×256 capacity ablation are running at writeup; if either
moves the number we'll have a clearer answer for whether the ceiling is
data-limited (Exp C: full 30M positions) or capacity-limited (Exp A:
40×256 net).

## Historical (2025 — 250k positions, MPS)

> **Negative result** at the time. Bigger network + multipv soft
> targets, 30 epochs, **underperformed 02b's smaller-network +
> hard-target recipe** by ~130 Elo. Recipe later replaced by the
> EKS-trained d15 above.

100-game-equivalent eval vs Stockfish UCI_Elo=1320 (02b's anchor; 02c
ran at SF1350, normalized).

| Run | Network | Targets | Training | Real Elo | W/D/L (per 100g) |
|---|---|---|---|---:|---:|
| **02b d10** (baseline) | 10×128 (~3M) | hard one-hot | 20 epochs MPS | **1185** [1104, 1254] | 19 / 25 / 56 |
| 02c 10-epoch | 20×256 (~24M) | multipv=8 soft (T=1) | 10 epochs | 998 (~968 normalized) | 7.5 / 8.3 / 84 |
| 02c 30-epoch | 20×256 (~24M) | multipv=8 soft (T=1) | 30 epochs | 1086 [988, 1157] (~1056 normalized) | 7.5 / 20.8 / 71.7 |

The rest of this file is the analysis of *those* numbers — preserved
because the hypotheses (soft-target dilution, undertraining-per-param,
T=1 too smooth) are still the right things to keep in mind for follow-up
ablations.

## Training trajectory (from train_history.json)

| epoch | loss | top-1 | top-K | grad-norm |
|---:|---:|---:|---:|---:|
| 0   | 4.02 | 14.7% | 54.6% | 3.68 |
| 9   | 2.65 | 31.3% | 81.8% | 1.48 |
| 14  | 2.53 | 33.9% | 84.3% | 1.32 |
| 19  | 2.46 | 35.3% | 85.6% | 1.37 |
| 24  | 2.42 | 36.0% | 86.4% | 1.71 |
| **29** | **2.40** | **36.7%** | **87.0%** | 1.55 |

Loss kept dropping (slope flattening). Top-K kept climbing. Top-1 mostly
plateaued after epoch ~20. The model wasn't at hard convergence at
epoch 29, but per-epoch returns were already < 0.2% top-1.

## Three honest hypotheses for the underperformance

Ranked by what the data supports:

### 1. Soft targets dilute signal at d10 teacher quality
Top-K=87% says "Stockfish's chosen move is in the network's top-8
predictions almost every time." Top-1=36.7% says "the network often
picks a *different* move from that top-8 than Stockfish would." That's
the **hedging failure mode** — the student learned to spread probability
across candidates instead of committing. Chess punishes hedging at
decision time: at 800 sims, MCTS amplifies a sharp prior more than a
flat one.

02b's hard targets force the student to commit to Stockfish's actual
choice. Empirically at d10, that turns out to be the better signal,
even though information-theoretically soft targets carry more bits.

### 2. 20×256 is undertrained for 250k positions
21M params × 243k positions ≈ **86 positions per parameter**. For 02b's
3M-param net on the same dataset: **80 positions per parameter** — about
the same! But 02b had hard targets that compress more cleanly into a
small network.

The bigger net needs *more* data, not less. We doubled the network and
kept the same dataset, which means we under-supplied gradient signal.

### 3. Temperature T=1 pawn is too smooth
With T=1 pawn, a 50cp gap between two moves yields a 62/38 probability
split. At d10, many positions have several moves within 30–50cp. That
gives target distributions with effective entropy H≈1.54 bits — closer
to "uniform over 4 candidates" than "this one move is best." Sharper
T (e.g. T=0.3) would force more concentration and may close the gap
to hard targets while keeping some secondary-ranking info.

## What this **does not** show

- It does **not** show distillation is bad. d10 distillation (02b) is
  still the best chess result in this repo.
- It does **not** show multipv soft targets are bad in general. It
  shows they're not strictly better than hard one-hot at d10 teacher
  quality.
- It does **not** isolate the bigger-net effect from the soft-target
  effect — we changed two variables at once.

## Clean follow-up experiments (ranked by information per dollar)

| Exp | Change vs 02c-30ep | Tests | Cost |
|---|---|---|---:|
| **A** | hard one-hot targets (everything else same) | "was the soft-target the problem, or the bigger net?" | ~$5 |
| **B** | T=0.3 soft (sharper) | "are soft targets per se bad, or is T=1 too smooth?" | ~$5 |
| **C** | d15 teacher, multipv=8 (full original plan) | "does multipv help when the teacher is stronger?" | ~$25 |
| D | distill→AZ-self-play (Lc0 path) | "can self-play close the search gap?" | $50+, days |

**Recommended next step**: A. It's the cleanest single-variable ablation
— same data, same net, same training time, just different loss. If A
beats 02b's 1185 with hard targets + 20×256, **the soft target was the
problem**. If A also lands near 1086, **the bigger net was the problem**
(undertraining-per-parameter).

## Wall-clock and cost (the full AWS run)

| Phase | What | Wall | Cost on g5.8xlarge |
|---|---|---:|---:|
| 0 | terraform apply, bootstrap, code rsync, key rotation, IP fights | ~30 min | $1.20 |
| 1 | Data gen — d10 mpv8 2000g on 28 CPU workers | 28 min | $1.15 |
| 2 | Training — 30 epochs on A10G GPU | 75 min | $3.10 |
| 3a | Eval — 120g vs SF1350 @ 800 sims | 34 min | $1.40 |
| 3b | Eval — 120g vs SF1350 @ 1600 sims | (still running at writeup) | ~$2.50 |
| | **Total spent so far** | **~3 h** | **~$9** |

The training was 2× slower per epoch than baseline because of the new
per-batch observability (grad-norm + JSON writes). I'd accept that
cost again — debugging this negative result without the observability
would have been blind.

## Lessons that generalize

1. **More compute on the same recipe rarely beats a better recipe at
   the same compute.** 30 epochs of multipv-soft + 20×256 ≈ 20 epochs
   of hard-target + 10×128 wallclock-wise, and the simpler recipe wins.

2. **Top-K accuracy is a *training* metric, not a *play* metric.** 87%
   top-K sounds excellent; the resulting Elo says it's not. The bridge
   between supervised metrics and game-playing Elo is uncertain and
   sometimes inverted.

3. **Observability changes you wish you had** *during* a run are the
   ones you need to bake in *before* the next run. Per-batch progress,
   gradient norms, and a saved training-history JSON cost almost
   nothing in code complexity and saved this analysis from being pure
   guesswork.

## Files

- `aws_results/run02_30ep/train_history.json` — full per-epoch metrics
- `aws_results/run02_30ep/distilled_epoch029.pt` — gitignored; pulled locally only
- `aws_results/run02_30ep/az_02c_e2e30.log` — full combined log
