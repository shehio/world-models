# 02 — AlphaZero-chess: training run results

> Reproduce with the commands at the bottom. Re-run on a fresh checkpoint to update.

## What changed in this run vs the v1 run

- **Batched MCTS** (run_mcts_batched, K=8 parallel descents per network call)
  added since v1. Wall-clock per simulation is ~1.9× faster, so we
  raised sims/move from 25 → 150 (training) and 25 → 80 (in-loop eval) /
  100 (final Elo eval).
- **More iters** (8 → 30) and **more games per iter** (6 → 8).
- **Longer training** per iter (80 → 200 train steps).
- **Larger Elo eval sample** (100 → 200 games vs random).

## Training run

- **Hardware:** Apple Silicon, CPU inference (3× faster than MPS at batch-1; with K=8 batches MPS gets closer but CPU is still simpler and competitive).
- **Config:** 30 iters · 8 self-play games/iter · 150 MCTS sims · 200 train steps/iter · max 200 plies/game · K=8 batched MCTS.
- **Total compute:** ~75 min wall (self-play dominates).
- **Final loss:** 2.85 (started at 3.90).

## Per-iteration trajectory

Loss + score-vs-random (n=6 each, evaluated every 2 iters).

| Iter | Total loss | Policy CE | Value MSE | Score vs random (n=6) |
|----:|----:|----:|----:|---:|
| 0  | 3.900 | 3.871 | 0.029 | — |
| 1  | 3.135 | 3.120 | 0.015 | 0.500 (0/6/0) |
| 2  | 2.987 | 2.974 | 0.014 | — |
| 3  | 2.906 | 2.893 | 0.013 | 0.667 (2/4/0) |
| 4  | 2.871 | 2.854 | 0.017 | — |
| 5  | 2.833 | 2.817 | 0.016 | 0.583 (1/5/0) |
| 6  | 2.832 | 2.812 | 0.020 | — |
| 7  | 2.826 | 2.803 | 0.023 | 0.667 (2/4/0) |
| 8  | 2.827 | 2.801 | 0.026 | — |
| 9  | 2.820 | 2.786 | 0.035 | 0.750 (3/3/0) |
| 10 | 2.804 | 2.780 | 0.024 | — |
| 11 | 2.832 | 2.804 | 0.028 | **0.917 (5/1/0)** |
| 12 | 2.816 | 2.790 | 0.026 | — |
| 13 | 2.812 | 2.780 | 0.031 | 0.750 (3/3/0) |
| 15 | 2.817 | 2.785 | 0.031 | 0.667 (2/4/0) |
| 17 | 2.828 | 2.795 | 0.033 | 0.583 (2/3/1) |
| 19 | 2.830 | 2.792 | 0.038 | 0.500 (1/4/1) |
| 21 | 2.813 | 2.780 | 0.033 | 0.500 (0/6/0) |
| 23 | 2.844 | 2.807 | 0.037 | 0.833 (4/2/0) |
| 25 | 2.829 | 2.792 | 0.037 | 0.583 (1/5/0) |
| 27 | 2.844 | 2.803 | 0.041 | 0.750 (3/3/0) |
| 29 | 2.849 | 2.807 | 0.042 | 0.833 (4/2/0) |

Observations:

- **Loss drops fast then plateaus.** 3.90 → 2.83 in 5 iters; ~2.83 thereafter.
  Most of the policy-CE drop is the network learning that *most* moves are
  illegal — the legal moves account for ~30 / 4672 entries, so a uniform
  prior gets log(4672) ≈ 8.45 nats CE per sample, while a legal-only
  uniform gets log(30) ≈ 3.4 nats. Going from 3.9 → 2.83 means the network
  is also learning something *within* the legal moves — but slowly.
- **Score-vs-random is noisy at n=6** (95% CI on a single 0.5 score is
  about ±0.36). Trends are not interpretable until the final 200-game eval.
- **Peak in-loop score was iter 11 (5W/1D/0L vs random).** Subsequent
  iters fluctuate around 0.6–0.83.

## Elo on net_iter029.pt

200 games vs random + 50 vs Stockfish at two strengths. Agent uses 100
MCTS sims with K=8 batching. max_plies=250.

| Opponent | N | W/D/L | Score | Elo (95% CI) |
|---|---:|---:|---:|---:|
| Random (anchor 0) | 200 | 89 / 107 / 4 | 0.713 | **+158 [+107, +215]** |
| Stockfish skill=0 d=1 (no anchor) | 50 | 0 / 6 / 44 | 0.060 | gap −478 (one-sided) |
| Stockfish UCI_Elo=1320 (anchor 1320) | 50 | 0 / 1 / 49 | 0.010 | upper bound ≈ 522 |

### How to read this

- **vs random.** 89 wins, 107 draws, only **4 losses** out of 200 games
  (1 of 50 if you only count games-where-someone-won, the agent won 96%
  of decisive games). The 95% CI is **strictly above 0** — we can now
  reject "the agent is no better than random" with confidence. Compared
  to the v1 run (+24 Elo, CI crossing 0), this is a 6.6× improvement
  driven by the larger sims/move that batched MCTS unblocked.
- **vs Stockfish skill=0 d=1.** 6 draws, 44 losses. The agent can
  occasionally hold a draw — usually by triggering 50-move/insufficient-
  material rules in a position Stockfish hasn't yet converted — but
  loses every decisive game.
- **vs Stockfish UCI_Elo=1320.** 1 draw, 49 losses. Stockfish at its
  weakest configurable strength still crushes the agent. Gives an
  approximate upper bound: agent's Elo is well below 1000.

## Reproduce

```bash
cd 02-alphazero-chess

# 30 iterations of self-play + train, ~75 min on Mac CPU
uv run python scripts/selfplay_loop.py \
    --iters 30 --games-per-iter 8 --sims 150 --train-steps 200 \
    --batch-size 8 --eval-games 6 --eval-sims 80 --eval-every 2 \
    --max-plies 200 --device cpu

# Elo on final checkpoint, 200 + 50 + 50 games, ~25 min
uv run python scripts/elo.py \
    --ckpt checkpoints/net_iter029.pt \
    --games-vs-random 200 --games-vs-stockfish 50 \
    --sims 100 --batch-size 8 --max-plies 250 --device cpu
```

## Caveats

- 30 iterations is still a tiny training budget. DeepMind's AlphaZero
  ran 700,000 *training steps* (we did 200×30=6,000) with 5,000 TPUs of
  self-play (we did 240 games on one CPU).
- Loss plateau at 2.83 likely indicates the network has reached the
  ceiling of what it can extract from this training distribution at this
  size. Bigger network, more sims, more games would push further.
- Score-vs-random above 0.85 starts to indicate "the agent is materially
  aware" — it's not just random, it's making choices that beat random
  more often than not. Below ~0.65 with n=6 we can't distinguish from
  random.
