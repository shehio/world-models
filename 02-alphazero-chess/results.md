# 02 — AlphaZero-chess: training run results

Three training runs in chronological order. Each one fixed something the
previous one taught us.

| Run | Network | Training compute | Headline result |
|---|---|---|---|
| v2 (single-process, batched MCTS) | 5 blocks × 64 ch (390k params) | 75 min | **+158 Elo vs random** [95% CI: +107, +215] |
| v3a (multi-process, bigger net) | 10 blocks × 128 ch (3M params) | 4 hours | Plateaued; +35 Elo head-to-head over v2 baseline (statistically marginal) |
| v3b (sharded buffer + reduced overtraining) | 10 blocks × 128 ch (3M params) | 4 hours, resumed from v3a | **+42 Elo head-to-head over v3a** with 7W/42D/1L of 50; absolute Elo see below |

The story: v2 worked. v3a got bigger and parallelized but hit a plateau
that I initially mistook for "model capacity." v3b diagnosed the actual
problem (stale replay buffer + too many SGD steps per generated game)
by reading what proven AZ-clones do, then fixed it.

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

## Summary across the three runs

| Run | Network | Wall | Total games | Elo vs random |
|---|---|---|---|---|
| v2 | 5b × 64ch | 75 min | 240 | +158 [+107, +215]   (89W/107D/4L of 200) |
| v3a (overnight, plateau) | 10b × 128ch | 4 h | 552 | not measured (plateau confirmed via h2h) |
| v3b (resume, fixed) | 10b × 128ch | 4 h additional | 888 added | **+263 [+186, +373]   (64W/36D/0L of 100)** |

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

# Final Elo on the v3b checkpoint
uv run python scripts/elo.py \
    --ckpt checkpoints_resume/final.pt \
    --games-vs-random 200 --games-vs-stockfish 50 \
    --sims 50 --batch-size 8 --n-blocks 10 --n-filters 128 \
    --max-plies 200 --device cpu
```
