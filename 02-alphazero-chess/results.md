# 02 — AlphaZero-chess: training run results

> Reproduce: `uv run python scripts/selfplay_loop.py ...` then `uv run python scripts/elo.py ...`. Commands at the bottom.

## Training run

- **Hardware:** Apple Silicon, CPU inference (3× faster than MPS at batch-1 for our 390k-param net).
- **Config:** 8 iters · 6 self-play games/iter · 25 MCTS sims · 80 train steps/iter · max 120 plies/game.
- **Total compute:** ~5 min self-play + ~2 min training + ~3 min eval = **~10 min wall**.
- **Final policy/value loss:** **2.42 / 0.005** (started at 4.18 / 0.025).

## Per-iteration trajectory

| Iter | Total loss | Policy CE | Value MSE | Score vs random (n=8) |
|----:|----:|----:|----:|---:|
| 0 | 4.178 | 4.153 | 0.025 | 0.500 (2W/4D/2L) |
| 1 | 2.818 | 2.814 | 0.004 | 0.375 (1W/4D/3L) |
| 2 | 2.616 | 2.614 | 0.002 | 0.438 (1W/5D/2L) |
| 3 | 2.529 | 2.527 | 0.001 | 0.500 (0W/8D/0L) |
| 4 | 2.498 | 2.490 | 0.008 | 0.500 (0W/8D/0L) |
| 5 | 2.481 | 2.468 | 0.013 | 0.625 (2W/6D/0L) |
| 6 | 2.443 | 2.439 | 0.003 | 0.438 (0W/7D/1L) |
| 7 | 2.417 | 2.411 | 0.005 | 0.438 (1W/5D/2L) |

Loss is monotonically decreasing — the network is learning. The
in-loop score-vs-random is too noisy to read on n=8 with max_plies=120
(most games hit the cap and draw).

## Elo on the final checkpoint (`net_iter007.pt`)

100 games vs random + 30 vs Stockfish at two strengths, max_plies=250, agent uses 50 MCTS sims.

| Opponent | N | W/D/L | Score | Elo (95% CI) |
|---|---:|---:|---:|---:|
| Random (anchor 0) | 100 | 16/75/9 | 0.535 | **+24 [−44, +95]** |
| Stockfish skill=0 depth=1 | 30 | 0/8/22 | 0.133 | gap −325 [−771, −186] |
| Stockfish UCI_Elo=1320 | 30 | 0/3/27 | 0.050 | **upper bound ≈ 990** |

### How to read this

- **vs random.** 16 wins to 9 losses (with 75 draws) over 100 games. The
  point estimate puts the agent at +24 Elo above random, but the 95% CI
  spans 0 — at this sample size and training budget we cannot reject
  "no better than random." There IS a small positive signal (W>L), but
  it would need ~500 games to be confidently above zero.
- **vs Stockfish skill=0 d=1.** Lost 22, drew 8, won 0. Even Stockfish
  at depth=1 with skill=0 is enough material-evaluation to crush this
  agent.
- **vs Stockfish UCI_Elo=1320 (the lowest the engine will allow).**
  Lost 27, drew 3, won 0. Score = 0.05 implies the agent is at *most*
  ~990 Elo — a one-sided upper bound. Could be much lower.

## Why so weak?

This was a **10-minute training run on a Mac**, not a real AlphaZero
training. To put it in context:

- DeepMind's AlphaZero ran 700,000 training steps with **5,000 TPUs**
  for self-play and 64 TPUs for training. Ours: 640 training steps,
  one CPU.
- Self-play volume: 48 games here vs ~44 million for the published
  AlphaZero run.
- MCTS sims/move: 25 here vs 800 in the paper.

The point of this exercise was to verify the **loop works end-to-end**
(it does — loss decreases monotonically), not to reproduce paper-level
strength. To meaningfully improve from here, the highest-leverage
changes would be:

1. **Batched MCTS** with virtual loss — would unblock 100–200 sims at
   the same wall-clock budget. Likely worth +200 Elo.
2. **More self-play volume.** 50–100× more games. Likely worth another
   +200–400 Elo.
3. **Larger replay window per training step.** Currently we sample
   uniformly from a 100k buffer; AlphaZero used a windowed recent
   buffer with more train steps per game.

## Reproduce

```bash
cd 02-alphazero-chess

# 8 iterations of self-play + train, ~10 min
uv run python scripts/selfplay_loop.py \
    --iters 8 --games-per-iter 6 --sims 25 --train-steps 80 \
    --eval-games 8 --eval-sims 25 --max-plies 120 --device cpu

# Elo on final checkpoint, ~10 min for 160 total games
uv run python scripts/elo.py \
    --ckpt checkpoints/net_iter007.pt \
    --games-vs-random 100 --games-vs-stockfish 30 \
    --sims 50 --max-plies 250 --device cpu
```
