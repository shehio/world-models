---
title: "Elo bisect"
subtitle: "binary-search stockfish UCI_Elo until you bracket the agent"
next: "/alphazero/"
---

## the motivation

A single eval at `UCI = 1350` reports "agent absolute Elo: 1807 [1704,
2034]" — the implied Elo CI is 330 points wide because the score
(0.93) is far from 0.5. The CI is tightest exactly when the agent is
*well-matched* against the opponent (score ≈ 0.5).

So: probe Stockfish at successively-stronger settings until a probe
lands in `[0.45, 0.55]`, then report that Elo. That's
[`experiments/distill-soft/scripts/elo_bisect.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/scripts/elo_bisect.py).

## the algorithm

Binary search over `UCI_Elo` in `[1350, 2800]`:

1. Probe at the midpoint with N games.
2. If score ≥ 0.5: agent is stronger than `mid` → raise `lo`.
3. If score < 0.5: agent is weaker than `mid` → lower `hi`.
4. Stop when bracket ≤ 100 Elo, **or** when a probe lands in
   `[0.45, 0.55]` (the score range where the implied Elo conversion
   is most accurate).

`log2(2800 - 1350) ≈ 10` rounds gives ~50-Elo precision, but in
practice 3–4 probes is enough because the in-bracket stopping rule
fires quickly.

## the math

The Elo formula converts a score into an Elo gap:

`gap = -400 · log10(1/score - 1)`

At extreme scores (0 or 1) this blows up, so the implementation
clamps to `[1e-3, 1 - 1e-3]` to keep math finite when the agent
sweeps or gets swept.

The agent's estimated Elo is `mid + gap` from the most recent probe.

## usage

```bash
python scripts/elo_bisect.py \
    --ckpt /work-tmp/distilled_epoch019.pt \
    --workers 8 --games-per-probe 40 \
    --sims 800 --n-blocks 20 --n-filters 256 \
    --out /work-tmp/bisect.json
```

Three to four probes converge to ±50 Elo precision in roughly 30–40
minutes on a `g6.4xlarge`. The JSON trace at `--out` lets you plot the
probe history.

## why this isn't already in the auto-eval daemon

The routine [eval](/eval/) runs at fixed anchors (1350 + 1800) so all
checkpoints are comparable on a single scale. Bisect is for when you
want the *single best estimate* of one specific checkpoint — typically
for a milestone like "the final d15 checkpoint" or "the self-play
loop's iter-N model."

Tested with 13 unit tests covering the Elo math, bracket halving,
stop conditions, and end-to-end convergence on a fake agent at 1,400 /
1,800 / 2,400 Elo. See
[`tests/test_elo_bisect.py`](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/tests/test_elo_bisect.py).
