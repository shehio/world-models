---
title: "What's next"
subtitle: "distill-then-RL — a distilled prior plus self-play, running on an EKS pod right now"
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
realistic path: **distill-then-RL** — warm-start a strong prior, then
self-play. That prior-plus-self-play shape is how Leela Chess Zero
reached grandmaster level on volunteer hardware (Lc0 built its prior
from self-play; we build ours by distilling Stockfish).

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

## Self-Play Postmortem — Six Attempts, No Progress {#selfplay-postmortem}

Self-play *should* be the next +200 to +500 Elo on top of the
distilled prior (self-play on a strong prior is how Lc0 reached engine
strength — though Lc0's prior came from self-play, not a Stockfish distill).
We've launched the loop six times since 2026-05-25 and not yet gotten
past iteration 1. Two distinct failure modes, both fixed in code, but
the *infrastructure* turned out to be the binding constraint.

### Failure mode #1 — catastrophic forgetting at LR=1e-3 / 1e-4

The first attempts trained on top of the distilled prior with the
same Adam LR=1e-3 used during supervised distillation. Self-play
generates ~30× less data per iter than supervised training, so the
gradient updates were 30× too large relative to the new signal —
the network *forgot the prior* on the first iter:

| Run | LR | Iters | Best sims=800 Elo @ UCI=1,800 |
|---|---:|---:|---:|
| selfplay v3 (`20260525T1331Z`) | **1e-4** | 0 → 3 | iter 0 = **1,643** [1561, 1712], iter 2 = 1,704 [1630, 1771] |

The prior (d10 ep 15) at sims=800 sits around 2,000 Elo. Iter 0 came
in at 1,643. That's **~−350 Elo from the prior in 200 SGD steps** —
the canonical catastrophic-forgetting signature.

Fix: commit
[`5c7cc65`](https://github.com/shehio/world-models/commit/5c7cc65)
dropped the self-play default to LR=1e-5 (100× smaller). Commit
[`ddbb6d7`](https://github.com/shehio/world-models/commit/ddbb6d7)
confirmed even LR=1e-4 was still too aggressive and reverted to 1e-5
from a clean d10 ep 15 seed.

### Failure mode #2 — spot evictions kill iter 0 before it finishes

With LR=1e-5 the network *stays at* the prior's strength on iter 0
(no progress, no regression — exactly as expected for one iter at a
gentle LR). But every run we've launched on spot has been **evicted
before iter 1 finishes**:

| Run | LR | Iters reached | Wallclock when killed | Iter 0 eval (sims=800, UCI=1,800) |
|---|---:|---:|---:|---:|
| v2 (`20260525T0453Z`) | 1e-5 | 0 | 159 min (11% of 24h budget) | 1,996 [1924, 2087] |
| (`20260525T0948Z`) | 1e-5 | 0 → 1 | 164 min | 1,885 [1819, 1958] |
| (`20260525T1948Z`) | 1e-5 | 0 | ≤ 60 min | 2,034 [1960, 2132] |
| (`20260526T0548Z`) | 1e-5 | 0 | ≤ 60 min | (no eval) |
| v4 (`20260526T0805Z`) | 1e-5 | 0 | 88 min (6% of budget) | (no eval) |

One iter of 32 games × 16 workers takes **~90 min** on a g6e.8xlarge.
The us-east-1 spot eviction window for g6e is hitting at < 90 min in
the AZs we've tried, so iter 1 never starts. The runs that did finish
iter 0 produced an iter-0 ckpt that's *the prior with one SGD pass on
800 fresh self-play positions* — nothing to learn from.

Why spot at all: the OD G/VT vCPU quota in us-east-1 was saturated by
the deep-eval boxes (autoeval daemon fires per-ckpt evals on the same
32-vCPU quota), so the only way to launch a long-running self-play
trainer was to accept spot eviction risk. Commit
[`266f082`](https://github.com/shehio/world-models/commit/266f082)
made that trade-off explicit.

### What needs to change for self-play to actually run

In rough order of "would fix it":

1. **Move self-play off spot.** $/h delta between g6e.8xlarge OD and
   spot is ~3×; iter 1 needs ~90 min of unbroken wallclock; the deep-
   eval daemon hogging the OD quota only fires while the laptop is
   on. A clean fix is one OD g6e.8xlarge in eu-central-1 (parallel to
   R1 v2), 24h budget, no spot.
2. **Shorten the iter wall.** 90 min/iter × budget needs to be longer
   than the AZ's typical spot lifetime, or shorten the iter. Drop
   `games-per-worker` from 2 to 1 (halves iter wall to ~45 min) and
   the loop fits inside more AZs' eviction windows.
3. **Replay buffer across runs.** Each spot eviction drops the
   self-play games on the floor (state.pt is uploaded but starting
   from a fresh trainer + fresh buffer wastes ~10% of the budget on
   warmup). Add a `--resume` path that pulls `state.pt` from S3 and
   continues from the last iter — would let us amortize evictions
   instead of restart from scratch each time.

The algorithm is *not* the bottleneck right now. It's the launcher
making a bet on spot stability that doesn't hold.

### Update 2026-05-29 — self-play ran on OD, and the verdict flipped {#selfplay-od-verdict}

The postmortem above blamed infrastructure (spot eviction). Moving to OD
removed that — and self-play then *ran to completion*, which surfaced the real
result:

- **Ungated (attempt #7, OD g6.4xlarge, ~10 iters).** The iter-10 checkpoint,
  evaluated on the same harness as the teacher (**~2,101** [2,020, 2,224] @
  sims=800), came in at **~1,730** [1,656, 1,797] — a confirmed **~370-Elo
  regression**. When the loop actually runs, the strong prior *degrades*: the
  MCTS self-play targets aren't better than what a ~2,100 net already plays,
  so the low-LR updates walk it off the supervised optimum.
- **Gated (MVP, 2026-05-29).** Added AlphaGo-Zero arena gating (promote only
  if a candidate beats the champion ≥55% over 60 games), a KL-anchor to the
  teacher, a 50-shard replay window, and an in-loop Stockfish yardstick.
  Gating *works* — the champion floors at the teacher's ~2,101, no regression —
  but candidates keep scoring **below 55%** vs the teacher (0.542, then 0.408
  → rejected), so nothing promotes. **Self-play holds the teacher; it does not
  climb.**

So the bottleneck was never the algorithm *or* the spot launcher — it's that
**the self-play signal is too weak to improve a ~2,100 net at one-GPU data
scale** (~1.3k recorded samples/iter). This is the strongest confirmation yet
of the project thesis: distillation ≫ self-play at fixed compute. Full
write-up in
[EXPERIMENTS_LOG.md](https://github.com/shehio/world-models/blob/main/EXPERIMENTS_LOG.md#selfplay-muzero-negative).

(Engineering footnote: one gated relaunch was lost to a mistaken
"optimization" that moved the gate match onto CPU workers — ~2-3× slower for
the 20×256 net than the single-threaded GPU gate. See
[failure-modes #18](/failures/).)

### Update 2026-05-30 — the gated run finished, and the MuZero sibling was a bug {#selfplay-gated-final}

Two follow-ups closed the loop on the verdict above.

**The gated run ran to its full 24h budget** (`20260529T0533Z`, g5.4xlarge A10G,
8 iters, on the corrected single-GPU gate). Over a full-length run the champion
promoted **exactly once**:

| gate @ iter | candidate vs champion (60g, sims=200) | result |
|---|---|---|
| 1 | 0.408 (W7 D35 L18) | rejected |
| 3 | **0.600** (W20 D32 L8) | **promoted → `net_iter003`** |
| 5 | 0.467 (W17 D22 L21) | rejected |
| 7 | 0.500 (W17 D26 L17) | rejected |

The single promotion landed at **1,931** [1,862, 2,011] vs SF-1800 (100-game
eval, 53W/30D/17L) — ~170 Elo *below* the teacher's ~2,101 [2,020, 2,224], with
the confidence intervals essentially non-overlapping (a *real* gap, not noise),
and every later candidate scored ≤ 0.50 against it. So with gating the
champion **floors at the teacher and never climbs**: one promotion shaped like
noise, not a trajectory. Same verdict as the MVP above, now with a full run
behind it.

**Go says the same thing, more cleanly.** The 9×9 self-play run (`20260526T1947Z`,
seeded from the 8×128 KataGo-parity teacher, 103 ungated iters) ended at **parity**
with its teacher — head-to-head **21–19 over 40 games, Elo gap +17, 95% CI
[−89, +124]**: statistically indistinguishable. Go neither beat the teacher nor
regressed. Across both games, on correct search: **distillation ≥ self-play at
fixed compute** — "≫" for chess, "≈" for 9×9 Go.

**Why "correct search" matters here.** The [MuZero leg](/vs-muzero/) — a separate,
*learned-dynamics* experiment — had collapsed to 0/30 vs Stockfish. That turned
out to be an **inverted value-sign in its MCTS child selection**: it ranked
children by the child's own-side-to-move value without negating to the parent's
POV, so the search preferred the *opponent's* best reply, and more simulations
made it worse. Fixed, the MuZero distill-init jumps from random to ~1,700 (still
short of the teacher). Crucially the bug was **confined to the MuZero MCTS** — both
self-play searches above (`wm_chess.mcts` and `distill_go.mcts`) negate to the
parent POV correctly, so the chess and Go results here stand. A clean reminder
that a "negative result" is only as trustworthy as the harness underneath it.

## What Originally Was Running Here (preserved for context)

Before the self-play postmortem above, this page described the
expected single 12h spot run. The job manifest, entrypoint and
orchestrator are all still in the repo and still correct — only the
"works in practice" claim needs the postmortem.

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

## d15-46M Trainers — Killed Early, Followed By Focused Evals {#d15-trainers-killed}

The data ablation said +199 Elo from a 6× larger dataset *with a weaker
teacher* (d10). The natural next step was to combine the strongest
teacher (d15) with the same data scale, which became the d15 46M
2×2 ablation: R1/R2 (network capacity) × v1/v2 (LR schedule). Full
results table on the [experiments page](/experiments/#d15-46m).

**Update 2026-05-26 17:15 UTC** — both cosine trainers (R1 v2 + R2 v2)
were killed early, before their nominal epoch budgets. The peaks
landed early (R1 v2 ep 7 = 2,209; R2 v2 ep 4 = 2,285), late-epoch
trajectories at sims=800 were not trending up, and the autoeval daemon
was offline anyway (operator laptop closed). Three follow-on facts
that closed the runs:

- **R1 v2** — eu-central-1 OD g6e.8xlarge `i-0a6c44e043b2d241e`.
  Killed at ep 15 of 40. Would have cost ~$320 to finish on OD,
  with low probability of beating ep 7's 2,209.
- **R2 v2** — ap-northeast-1 spot g6e.8xlarge `i-008f0d7d3a15974b9`.
  Killed at ep 22 of 30. Would have cost ~$5 to finish on spot;
  killed for *information value* — ep 4 sims=4,000 was already the
  best deep-eval'd ckpt and later epochs were noise on sims=800.
- **R2 v1 + R1 v1** were already complete and idle.

All trained checkpoints remain on S3 at
`s3://wm-chess-library-594561963943/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/`.
Nothing is lost; the runs can resume from any saved epoch if a future
hypothesis justifies it.

**Update 2026-05-26 23:06 UTC — the 2,300 line is broken.** The
"ep 14 sims=800 was the run's highest but never got a deep-eval" lead
turned into the project's strongest measurement: **R2 v2 ep 14 at
sims=4,000 vs UCI=1,800 = 2,301 Elo** [2,190, 2,601], 95% CI lower
bound strictly above the prior peak (d10 ep 15 = 2,189). One last
in-flight eval (R2 v2 ep 4 at sims=8,000 vs UCI=2,000) will give the
tightest single CI of any chess measurement to date — landing within
minutes. The two cheap follow-up evals were:

| Eval | Instance | What it answers |
|---|---|---|
| sims=8,000 vs UCI=2,000 on **R2 v2 ep 4** | us-east-1 g6.4xlarge OD `i-04f84755196ee0163` (~$5) | Does deeper search on the top ckpt give a tighter CI? UCI=2,000 anchor keeps score near 0.5. _In final ~10 min as of 2026-05-26 22:30 UTC._ |
| sims=4,000 vs UCI=1,800 on **R2 v2 ep 14** | us-east-1 g6.2xlarge OD `i-0b24ea8a7fdd7d149` (~$5) | R2 v2 ep 14 had the highest sims=800 of the run (2,055) but was never deep-eval'd. **Landed 2026-05-26 23:06 UTC: 2,301 Elo [2,190, 2,601] — first project measurement with a 95% lower CI bound above 2,300.** |

Per-ckpt sims=800 evals normally fire via the
[autoeval daemon](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-autoeval-daemon.sh)
when the operator laptop is on; ckpts crossing 1,940 trigger a
sims=4,000 deep-read via the
[`wm-deep-eval-daemon`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-deep-eval-daemon.sh).
With the laptop currently offline, the two evals above were launched
as one-shot scripts.

## The 2,500 Ceiling — What's in the Way and Why

**Verdict: with cosine LR, distillation has *not* saturated yet.** The
2026-05-26 cosine numbers (R1 v2 ep 7 = 2,209 [2115, 2389]; R2 v2
ep 4 = 2,285 [2177, 2554]) both clear the prior 2,189 ceiling, and
R1 v2 has 25 more epochs left. The constant-LR write-up that said
"pure SL has saturated" was wrong about the recipe; the teacher
itself (~2,500 Elo) is still the asymptotic upper bound, but we now
have evidence we're climbing toward it, not stuck below it.

### Levers ranked by Elo per dollar

| # | Lever | Estimated Δ Elo | Effort | Status |
|---|---|---|---|---|
| 1 | **Self-play RL** on top of distilled prior (self-play on a prior, à la Lc0/AlphaZero) | +200 to +500+ | Weeks of GPU + careful LR tuning | Ran on OD + gated (2026-05-29): **regresses ungated (→~1,730), holds but does NOT climb gated (~2,101)**. Signal too weak at 1-GPU data scale — see the [verdict update](#selfplay-od-verdict). |
| 2 | **Higher eval sim count** (sims=16,000+) | +100 to +200 | $0 retraining, ~4× per-game time at eval | Untested above sims=4,000 |
| 3 | **Finish R1 v2 (cosine d15)** | +0 to +100 | Currently 15/40 epochs; ~5 more days wallclock | In flight. Best so far = 2,209 at ep 7; could climb further. |
| 4 | **Re-eval R2 v2 ep 4 at 400-game count** | tightens CI (currently ±200 Elo) | One ~4h eval run (~$5) | Open. 2,285 point estimate is the project high but the CI is too wide to publish confidently. |
| 5 | **Sharper teacher targets** (multipv=1 hard, or T=0.3 soft) | +50 to +150 | One retraining run (~$80) | Tested at small scale; not at full data scale. |
| 6 | More data (100M+ positions) | Probably +0 to +50 (saturating) | New 200K+ game datagen + 60h training (~$300) | The 30M→46M step at constant LR gave 0 Elo, but the cosine d15 result suggests we hadn't actually used the 46M well — re-evaluate after R1 v2 finishes. |
| 7 | Bigger net (60×384, ~100M params) | Probably +0 to +50 | Major training cost (~$400) | Capacity rejected at 5M; partially re-tested at 46M. |
| 8 | Stronger teacher (Stockfish d20+) | Uncertain (+30 to +100 maybe) | New datagen at higher depth (~$500 — d depth doubles datagen cost) | Untested. |

**Lever #1 (self-play) is still the only one in this list that can
plausibly hit 2,500.** Everything else is incremental. Self-play
replaces "match the teacher's policy" with "find moves the teacher
missed via search, then train on those" — which is *also* what carried
Leela to ~3,600 over years of community self-play.

**Update 2026-05-29:** this ranking predates *actually running* self-play. On
OD it ran to completion and **regressed** (→~1,730); gated, it **holds** at
~2,101 but doesn't climb (see the [verdict update](#selfplay-od-verdict)). At
one-GPU data scale, self-play is no longer the clear path to 2,500 — the
distillation levers (cosine LR, deeper eval-side search) remain the reliable
ones. The honest current ceiling is the distilled **~2,301**.

### Order of operations

1. **Re-launch self-play on OD, not spot.** Single g6e.8xlarge in
   eu-central-1 (parallel to R1 v2), LR=1e-5, 24h budget, OD pricing
   (~$8 vs spot's ~$3). Iter 1 needs ~90 min of unbroken wallclock; OD
   gives us that. If it doesn't regress *or* improve, the distill
   ceiling is the project ceiling and we accept ~2,250 as the
   answer. If it improves cleanly, scale up.
2. **Restart the autoeval daemon.** As of 2026-05-26 16:00 UTC the
   laptop is offline, so R1 v2 ckpts ≥ ep 13 and R2 v2 ckpts ≥ ep 18
   are still unevaluated. The training continues without it; eval just
   stalls.
3. Re-eval R2 v2 ep 4 at 400 games to tighten its CI before promoting
   2,285 to the headline number.
4. Finish R1 v2 + R2 v2 (both in flight). Once they peak, run an
   eval-sims sweep on the top ckpt across sims = {800, 2,000, 4,000,
   8,000, 16,000, 32,000}.

### What we're NOT going to do (and why)

- **More data alone (constant LR).** Falsified by the constant-LR
  arm of R1 and R2. Worth re-trying at cosine LR once R1 v2 finishes.
- **Tabula-rasa AlphaZero from scratch on one GPU.** d15 ep 20 (or
  d10 ep 15) is a much better starting prior than random; the
  distill-then-RL recipe was always the realistic path.
- **Bigger network at constant LR.** Pure capacity was already
  rejected. Bigger network *with* cosine is a possible follow-up
  after R1 v2 finishes, but the 40×256 / 20×256 tie at constant LR
  suggests it won't move the needle much either way.

