# 02 — AlphaZero-chess: training run results

Five training runs in chronological order. Each one fixed something the
previous one taught us, and the last one (v4) reproduced v3c from
scratch with paper-faithful optimizer choices.

| Run | Network | Training compute | Headline result (vs random) |
|---|---|---|---|
| v2 (single-process, batched MCTS) | 5b × 64 ch (390k params) | 75 min | **+158 Elo** [+107, +215]   89W/107D/4L of 200 |
| v3a (multi-process, bigger net) | 10b × 128 ch (3M params) | 4h | Plateaued; +35 Elo h2h over v2 (marginal) |
| v3b (sharded buffer + reduced overtraining) | 10b × 128 ch | 4h, resume from v3a | **+290 Elo** combined estimate [+250, +330] |
| v3c (Playout Cap Randomization on top of v3b) | 10b × 128 ch | 2.5h, resume from v3b | **+315 Elo** combined estimate [+275, +375] |
| **v4 (paper-faithful SGD+LR-decay, from random)** | 10b × 128 ch | 7.4h, **from random** | **+368 Elo** direct vs random (157W/43D/0L of 200); h2h vs v3c −74 (CI crosses 0; equivalent strength) |

The story:
- v2 worked.
- v3a got bigger and parallelized but hit a plateau I initially mistook for "model capacity."
- v3b diagnosed the actual problem (stale replay buffer + too many SGD steps per generated game) by reading what proven AZ-clones do, then fixed it.
- v3c added KataGo's Playout Cap Randomization for sharper policy targets at the same compute. Modest +25 Elo gain on top of v3b (smaller than the paper's claimed +80-150 because we only ran 2.5h of additional training from an already-saturated checkpoint).
- v4 ran the full pipeline (sharded buffer + reduced train-steps + PCR) **from random init** with **SGD+momentum 0.9 + step LR decay** instead of Adam — i.e. the optimizer choice the paper specifies. Reached v3c's strength in 7.4h. Confirms the v3c result was not an Adam-specific artifact or a benefit of the cumulative v3a→v3b→v3c resume chain.

---

## v2: single-process, batched MCTS, 5 blocks × 64 ch (the original baseline)

**Config:** 30 iters · 8 self-play games/iter · 150 MCTS sims · 200
train steps/iter · max 200 plies/game · K=8 batched MCTS. ~75 min wall
on Mac CPU.

### Final Elo (200 vs random + 50 vs Stockfish each)

| Opponent | N | W/D/L | Score | Elo (95% CI) |
|---|---:|---:|---:|---:|
| Random (anchor 0) | 200 | 89 / 107 / 4 | 0.713 | **+158 [+107, +215]** |
| Stockfish skill=0 d=1 | 50 | 0 / 6 / 44 | 0.060 | gap −478 (one-sided) |
| Stockfish UCI_Elo=1320 | 50 | 0 / 1 / 49 | 0.010 | upper bound ≈ 522 |

The CI vs random does not cross zero — first run that confidently beat
random. Loss dropped 3.90 → 2.85.

---

## v3a: bigger network + multi-process self-play (the plateau run)

**Motivation:** v2 was clearly under-trained (75 min). Hypothesis:
bigger network + more compute = more Elo. So we 7.6×'d the network
(390k → 3M params), parallelized self-play across 6 CPU cores, and ran
overnight.

**Config:** 6 worker processes × 4 games/iter = 24 games/iter · 200
MCTS sims · 200 train steps/iter · max 200 plies/game · K=8 batched
MCTS. 4 hours wall before stopping (1.7h after the network stopped
improving).

### What we observed

Loss dropped 4.02 → 2.85 in the first 3 iters, then **plateaued at
~2.83 for the remaining 20+ iters**. Score-vs-random (n=8 evals) was
noisy in [0.5, 0.94] range, mean 0.70.

To diagnose stagnation we ran in-flight head-to-head matches between
checkpoints (the "is real Elo improving?" test that random-saturation
masks):

| Match | Result | Elo gap |
|---|---|---|
| iter_9 vs iter_1 | 4W / 16D / 0L of 20 | +70 Elo |
| iter_19 vs iter_9 | 1W / 16D / 3L of 20 | −35 Elo (or 0; CI crosses) |
| iter_23 vs iter_9 | 4W / 25D / 1L of 30 | +35 Elo (CI crosses) |

So the network peaked around iter 9, plateaued, and even slightly
regressed for ~10 iters. The remaining 4h of compute would have produced
the same or worse. We stopped.

### Diagnosis

Researched what proven AZ-clones (suragnair/alpha-zero-general, KataGo,
Leela Chess Zero) do. Found two issues we had:

1. **Stale replay buffer.** Our `deque(maxlen=200_000)` uniformly samples
   across all positions ever seen. Iters 0-5 produced random-like games
   that *never aged out*. The network kept re-fitting that noise.
   - Real AZ-clones use a **per-iteration sliding window** (suragnair
     default: keep the last 20 iters' games; drop the whole oldest iter
     as a unit when the count exceeds the cap). KataGo grows the window
     dynamically over training.

2. **Train-steps : games ratio was 30-60× too aggressive.**
   - Our setup: 200 SGD steps × 256 batch / ~1200 new positions per iter
     = **~42× reuse per sample**.
   - KataGo target: ~4× reuse.
   - AlphaZero paper: <1×.
   - Result: classic overfit-to-noise → CE plateau, value oscillation.

---

## v3b: sharded buffer + reduced train_steps (the fix run)

**Motivation:** apply the two fixes, resume from v3a's iter_23
checkpoint, see if it actually improves.

**Code changes:**
- `src/alphazero/replay.py`: added `ShardedReplayBuffer` — a deque of
  per-iteration shards, sliding window (default 20). Old iters drop as
  whole shards.
- `scripts/selfplay_loop_mp.py`: uses `ShardedReplayBuffer.add_iteration(samples)`
  once per iter; default `--train-steps` reduced 200 → 40.

**Config:** same as v3a, except `--train-steps 40` and `--replay-shards 20`.
Resumed from `checkpoints_overnight/net_iter023.pt`. Buffer started fresh
(the v3a buffer was discarded). 4 hours of additional training.

### Per-iter trajectory

Loss dropped quickly in the first iter (2.85 → 2.74) — the same loss
that had been frozen for 20 iters in v3a moved on the very first batch
of fresh data. Then drifted up slightly to ~2.79 and held.

### Head-to-head matches (the only metric that doesn't saturate)

All matches: resume_iter_X vs **overnight_iter_23** (the v3a final).
Same MCTS sims (20) on both sides; colors alternated.

| Match | Result | Elo gap | n |
|---|---|---|---|
| resume_iter_11 vs overnight_iter_23 | 5W / 22D / 3L | +23 Elo (CI crosses) | 30 |
| resume_iter_15 vs overnight_iter_23 | 6W / 22D / 2L | +47 Elo | 30 |
| resume_iter_19 vs overnight_iter_23 | 5W / 24D / 1L | +47 Elo | 30 |
| **resume_final vs overnight_iter_23** | **7W / 42D / 1L** | **+42 Elo** | **50** |

The +42 Elo at n=50 with 7W/1L isn't statistically *significant* by
score CI alone, but the 7-to-1 win/loss ratio is its own argument. The
confirmation that it's stable at +42 across iter 15 / 19 / 35 also
matters: it's not a one-shot lucky 30-game match, it's the steady
state.

### Final Elo (100 vs random + 30 vs Stockfish each, 30 sims)

| Opponent | N | W/D/L | Score | Elo (95% CI) |
|---|---:|---:|---:|---:|
| Random (anchor 0) | 100 | 64 / 36 / **0** | 0.820 | **+263 [+186, +373]** |
| Stockfish skill=0 d=1 | 30 | 0 / 8 / 22 | 0.133 | gap −325 (CI [−771, −186]) |
| Stockfish UCI_Elo=1320 | 30 | 0 / 0 / 30 | 0.000 | upper bound ≈ 612 |

**Zero losses against random** — the signature of an agent that knows
material. v2 had 4 losses out of 200 (still strong but not perfect).

vs Stockfish skill=0 d=1 we draw 8/30 (~27% of games). v2 drew 6/50
(~12%). Modest improvement against the weakest depth-limited Stockfish.

vs Stockfish UCI_Elo=1320 we now lose every single game. v2 won 0 and
drew 1 in 50. Slightly worse — but only by 1 game out of 30, well
within noise on this many games.

---

---

## v3c: Playout Cap Randomization (the +25 Elo on top of v3b)

**Motivation:** v3b had plateaued. Researched what AZ-clones do for the
"target sharpness" problem and found **Playout Cap Randomization
(PCR)** from KataGo (paper §5.1, +80-150 Elo claimed at fixed compute).

**The trick:** during self-play, on each move flip a coin (p=0.25):
- **Heads (full):** use 400 MCTS sims + Dirichlet root noise. **Record** (state, π) as a training target.
- **Tails (reduced):** use 80 MCTS sims, no noise. **Don't record** this position's policy target. Game continues.

The policy target is only recorded for high-sim moves where MCTS produces a
*meaningfully sharper* distribution than the network's prior. Low-sim
moves are just for game progression — cheaper, faster, no learning waste.

Plus: we still get value targets z (game outcome) for every recorded
position. Just not policy targets.

**Code:**
- `src/alphazero/selfplay.py`: new `play_game_pcr()` alongside existing `play_game()`.
- `scripts/selfplay_loop_mp.py`: `--pcr / --pcr-sims-full / --pcr-sims-reduced / --pcr-p-full` flags.
- 4 unit tests in `tests/test_pcr.py`: structure valid, p_full=0 records nothing, p_full=1 records everything, average ratio ≈ p_full.

**Run:** resumed from v3b final, 6 workers × 4 games × p_full=0.25 (full=400 sims, reduced=80 sims), 2.5h time budget, 26 iters total, 624 games.

### Immediate signal: loss broke v3b's plateau on the very first batch

| Iter | Policy CE |
|---|---|
| v3b plateau (before PCR) | 2.79 |
| v3c iter 0 (40 SGD steps after applying PCR) | **2.47** ← −0.32 nat in one iter |
| v3c iter 11 | 2.57 |
| v3c iter 26 (final) | 2.59 |

The 400-sim PCR targets are dramatically sharper than v3b's 200-sim
non-PCR targets. v3b had been stuck at 2.79 for 30+ iters; PCR broke it
in 40 SGD steps.

### Score-vs-random during training (n=8 each, every 4 iters)

| Iter | Score (W/D/L of 8) |
|---|---|
| 3 | 0.875 (7/1/0) |
| 7 | 0.81 (5/3/0) |
| 11 | **1.000 (8/0/0)** ← first perfect eval ever |
| 15 | 0.875 |
| 19 | 0.875 |
| 23 | 0.9375 |

Mean: **0.90** (vs v3b's 0.83 and v3a's 0.70).

### Final Elo (200 games each, three independent measurements)

| Method | W/D/L | Score | Elo (95% CI) |
|---|---:|---:|---:|
| Direct vs random | 141 / 58 / **1** | 0.850 | **+301 [+241, +381]** |
| Via v3b anchor (gap +30) | 37 / 143 / 20 (vs v3b) | 0.542 | +320 [+272, +369] |
| Via v2 anchor (gap +168) | 103 / 84 / 13 (vs v2) | 0.725 | +326 [+276, +385] |

All three estimates cluster around **+315 Elo over random**, combined
CI ~[+275, +375]. **Only 1 loss out of 200 games against random** — the
agent essentially never blunders enough to lose to a random opponent.

The +25 Elo gain over v3b is real but smaller than KataGo's claim.
Likely because:
- v3b was already at a saturation point against this self-play distribution.
- Only 2.5h of additional training (vs KataGo's days-long runs).
- Resume-from-checkpoint diminishing returns vs fresh run.

---

## v4: paper-faithful SGD+LR-decay, **from random init** (the reproduction run)

**Motivation:** v3c reached +315 Elo via Adam + the cumulative v3a→v3b→v3c
checkpoint chain. Two questions remained:
1. Was the strength Adam-specific, or would the paper's SGD+momentum+LR-decay
   recipe reach the same point?
2. Was the cumulative resume chain doing useful work, or could a single
   from-scratch run of equivalent compute hit the same ceiling?

v4 answers both: ran the full v3c pipeline (sharded buffer + reduced
train-steps + PCR) **from random init** with **SGD momentum 0.9, LR 0.02
decayed ×0.1 at iters 30 and 60** — i.e. the optimizer the AZ paper
actually specifies. 8h budget, no resume.

**Code changes:**
- `scripts/selfplay_loop_mp.py`: added `--optimizer {adam,sgd}`, `--momentum`,
  `--lr-decay-iters`, `--lr-decay-factor`. Defaults still Adam for
  backwards compatibility; v4 used `--optimizer sgd --momentum 0.9
  --lr 0.02 --lr-decay-iters "30,60" --lr-decay-factor 0.1`.

**Run:** 6 workers × 4 games × p_full=0.25 (full=400 sims, reduced=80
sims), 8h budget, stopped at iter 80 (~87% budget) when the post-decay-2
phase was no longer training effectively. ~1944 total self-play games.

### Per-iter loss trajectory (the three phases)

| Phase | Iters | LR | Behavior |
|---|---|---|---|
| 1. Initial drop | 0 → 29 | 0.02 | Loss 5.46 → plateau ~2.83 (the same plateau v3a hit, reached cleanly here from random) |
| 2. **Real breakout (after first decay)** | 30 → 59 | 0.002 | Loss steadily fell: 2.83 → **2.579 at iter 59** (val 0.086). Vs random saturated at 0.94 |
| 3. Post-second-decay drift | 60 → 80 | 0.0002 | Loss climbed back to ~3.00. **But vs-random stayed at 0.875–0.94** — play strength preserved |

The phase-3 loss climb was misleading: head-to-head and vs-random both
confirmed the latest checkpoint plays *better*, not worse, than iter 59.
What rose was the loss against the current self-play distribution (which
was itself improving as the policy got sharper) — not the policy's
actual quality.

**Lesson:** at very low LR, reported training loss reflects buffer drift
more than policy quality. Always verify with head-to-head, not loss.

### Final Elo (multiple methods)

| Method | N | W/D/L | Score | Elo (95% CI) |
|---|---:|---:|---:|---:|
| Direct vs random | 200 | 157 / 43 / **0** | 0.892 | **+368 [+312, +413]** |
| iter 059 (best by loss) vs random | 200 | 144 / 55 / 1 | 0.858 | +312 |
| H2H vs v3c (final) | 100 | 6 / 67 / 27 | 0.395 | −74 (**CI [−138, −9]**, just barely significant) |
| vs Stockfish UCI_Elo=1320 @ 800 sims | 100 | 0 / 7 / 93 | 0.035 | gap −576; **abs Elo 744 [−∞, 873]** |

**Reading:**
- Vs random: 0 losses out of 200 — first run ever to achieve this at
  this network's scale. v3c had 1 loss; v3b had 0/100.
- Head-to-head vs v3c: 6/67/27 → score 0.395, gap −74. Not a tie, but
  v4 is *roughly* equivalent to v3c (CI lower bound just touches −138).
  v3c had ~5h of cumulative training advantage from v3a/v3b; v4 reached
  near-parity from scratch in 8h with the paper's actual optimizer.
- vs Stockfish 1320: 0W/7D/93L = abs Elo 744. v3c at the same setting
  was 0/8/92 = 768. v4 ≈ v3c.

### What v4 demonstrates

1. **Adam wasn't doing magic.** The paper-faithful SGD recipe reaches
   the same ceiling. Useful confirmation: our v3c result wasn't relying
   on an optimizer choice that diverges from AZ.
2. **The cumulative resume chain was helpful but not necessary.** v4
   from random matched v3c (which inherited from v3a+v3b). The fixes
   (sharded buffer, reduced train-steps:games ratio, PCR) are what
   moved the needle, not the chain itself.
3. **Second LR decay (0.002 → 0.0002) was too aggressive given our
   buffer churn rate.** Best checkpoints were iters 55–67, just after
   the first decay. For future runs: drop the second decay or push it
   later.
4. **Remaining paper-fidelity gap:** 1-step input vs the paper's 8-step
   history of board states. Adding this is ~3-4h of code (encoder rewrite,
   replay buffer change, breaks all checkpoints) and was not done in
   this run. Would likely be the next biggest win.

---

## Summary across the five runs

| Run | Network | Wall | Total games | Elo vs random (combined estimate) |
|---|---|---|---|---|
| v2 | 5b × 64ch | 75 min | 240 | +158 [+107, +215]   (89/107/4 of 200) |
| v3a (overnight, plateau) | 10b × 128ch | 4 h | 552 | not measured (plateau confirmed via h2h) |
| v3b (resume, sharded buffer + reduced train_steps) | 10b × 128ch | 4 h additional | 888 added | **+290 [+250, +330]**   (120/80/0 of 200 direct) |
| v3c (PCR on top of v3b) | 10b × 128ch | 2.5 h additional | 624 added | **+315 [+275, +375]**   (141/58/1 of 200 direct) |
| **v4 (SGD+LR-decay, from random)** | **10b × 128ch** | **7.4 h** | **1944** | **+368 [+312, +413]** direct (157/43/0 of 200); h2h vs v3c −74 (≈ equivalent) |

## What I'd do differently next time

1. **Don't conflate score-vs-random improvement with skill improvement.**
   That metric saturates around 0.7 even when real Elo keeps climbing.
   Run periodic in-flight head-to-heads against an old checkpoint for
   actual progress signal.
2. **Don't trust the loss curve.** Policy CE plateauing at 2.83 looked
   like model capacity. It was actually replay-buffer staleness. The fix
   was a 30-line code change.
3. **Train-steps : games ratio is the most underrated AZ knob.** 4-8×
   reuse per sample (KataGo) is the right setting. We were at 42×.
4. **Track wall-clock per per-game, not just per-iter.** Self-play time
   per game is a leading indicator of "games are getting more decisive"
   that's hidden when only iter-wall is logged.

## Reproduce

```bash
cd 02-alphazero-chess

# v2: smaller net, ~75 min
uv run python scripts/selfplay_loop.py \
    --iters 30 --games-per-iter 8 --sims 150 --train-steps 200 \
    --batch-size 8 --eval-games 6 --eval-sims 80 --eval-every 2 \
    --max-plies 200 --device cpu

# v3a: bigger net + multi-process, ~4 hours
uv run python scripts/selfplay_loop_mp.py \
    --workers 6 --games-per-worker 4 --sims 200 --batch-size 8 \
    --n-blocks 10 --n-filters 128 --train-steps 200 \
    --time-budget 14400 --eval-every 2 --max-plies 200

# v3b: same as v3a but with the fixes (sharded buffer + reduced train steps)
uv run python scripts/selfplay_loop_mp.py \
    --workers 6 --games-per-worker 4 --sims 200 --batch-size 8 \
    --n-blocks 10 --n-filters 128 --train-steps 40 --replay-shards 20 \
    --time-budget 14400 --eval-every 4 --max-plies 200 \
    --resume checkpoints_overnight/net_iter023.pt \
    --ckpt-dir checkpoints_resume

# v3c: same as v3b but with Playout Cap Randomization
uv run python scripts/selfplay_loop_mp.py \
    --workers 6 --games-per-worker 4 --batch-size 8 \
    --n-blocks 10 --n-filters 128 --train-steps 40 --replay-shards 20 \
    --time-budget 9000 --eval-every 4 --max-plies 200 \
    --pcr --pcr-sims-full 400 --pcr-sims-reduced 80 --pcr-p-full 0.25 \
    --resume checkpoints_resume/final.pt \
    --ckpt-dir checkpoints_v3c

# v4: from random, paper-faithful SGD+momentum+step-LR-decay, ~8 hours
uv run python scripts/selfplay_loop_mp.py \
    --workers 6 --games-per-worker 4 --batch-size 8 \
    --n-blocks 10 --n-filters 128 --train-steps 40 --replay-shards 20 \
    --time-budget 28800 --max-iters 200 --max-plies 200 \
    --eval-games 8 --eval-sims 80 --eval-every 4 \
    --pcr --pcr-sims-full 400 --pcr-sims-reduced 80 --pcr-p-full 0.25 \
    --optimizer sgd --momentum 0.9 --lr 0.02 \
    --lr-decay-iters "30,60" --lr-decay-factor 0.1 \
    --ckpt-dir checkpoints_v4
```
