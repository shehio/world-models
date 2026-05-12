# 02b — AlphaZero-architecture, supervised-distilled from Stockfish

> **Important: this is NOT a faithful AlphaZero implementation.** AZ's
> defining feature is *tabula rasa* learning from self-play with no
> external knowledge. This project deliberately violates that to test a
> different question: **given good targets, how strong can our same
> network architecture get?**
>
> The "real" AlphaZero implementation is in [`../02-alphazero-chess/`](../02-alphazero-chess/).
> This 02b is an experimental sibling on a separate branch
> (`stockfish-distill`) so it doesn't muddy the AZ main line.

## What this does

1. **Generate data**: Stockfish (any reasonable strength) plays itself
   for thousands of games. Save (board_position, move_played, game_outcome)
   tuples to disk.
2. **Train**: same network architecture as 02 (`AlphaZeroNet`, 10 ResNet
   blocks × 128 channels). Loss is **cross-entropy on Stockfish's actual
   move** (one-hot policy target, no MCTS-derived π) plus **MSE on game
   outcome** (value target). Standard supervised learning.
3. **Eval**: same as 02 — MCTS over the trained network at decision
   time, vs Stockfish UCI_Elo=1320.

## Why distillation works (and why it's not AZ)

- AlphaZero's policy iteration loop is *slow* because the network has
  to discover everything from scratch — it spends thousands of games
  learning that captures matter, knights jump, etc.
- Stockfish, given just a few thousand of its own games, can teach all
  of that directly through its move choices.
- Lc0's earliest networks were bootstrapped from human master games for
  exactly this reason. KataGo did similar bootstrapping for Go.
- This is **distillation**, not RL. The agent is not exploring; it's
  imitating. The ceiling is roughly the strength of the teacher — you
  can't be much stronger than what you imitate.

## Goal of this experiment

Win **1 game out of 100** against Stockfish UCI_Elo=1320 with the same
network architecture as our 02-AZ run. If yes: confirms our v3c
architecture can play stronger chess; the bottleneck was self-play data
quality, not the network. If no: the network itself is the limit.

## Layout

```
src/azdistill/
    board.py, network.py, mcts.py, arena.py, config.py  — copied from 02 (read-only)
    stockfish_data.py                                    — game generation
    train_supervised.py                                  — supervised training step
scripts/
    generate_data.py                                     — Stockfish self-play data
    train.py                                             — train on the data
    eval.py                                              — vs Stockfish 1320 at 800 sims
tests/
    test_data_generation.py
    test_train.py
data/   <- generated game tuples (gitignored)
```

## Results (vs Stockfish UCI_Elo=1320, 100 games at 800 sims)

| Run | Teacher | Training data | Train top-1 | W / D / L | Score | Absolute Elo |
|---|---|---|---|---|---|---|
| Baseline v3c (pure self-play, no distill) | n/a | self-play | n/a | 0 / 8 / 92 | 0.04 | ~768 |
| **d6 distilled** | Stockfish depth=6 (~1900 Elo) | 500 games / 64k positions | 91.6% | **2 / 31 / 67** | 0.175 | **~1051 [939, 1129]** |
| **d10 distilled** | Stockfish depth=10 (~2200 Elo) | 2000 games / 250k positions | 65.6% | **19 / 25 / 56** | 0.315 | **~1185 [1104, 1254]** |

The d10 result is **+417 Elo over v3c** with the same network architecture
and the same eval (800 sims). Confirms our network capacity was never the
ceiling — pure self-play just couldn't generate strong-enough training
targets in the compute budget we had.

Note: d10 has *lower* train top-1 accuracy than d6 (65.6% vs 91.6%) because
depth=10 Stockfish moves are harder to memorize. Lower memorization, but
the moves it learns to predict are *stronger* — better generalization,
better play.

## Run

```bash
cd 02b-alphazero-stockfish-distill
uv sync

# Tests (~30s)
uv run python tests/test_data_generation.py
uv run python tests/test_train.py

# Phase 1: generate Stockfish self-play data
# d6 (fast): ~20 sec for 500 games at depth=6
uv run python scripts/generate_data.py --n-games 500  --workers 6 --depth 6  --output data/stockfish_d6_500g.npz
# d10 (stronger): ~7 min for 2000 games at depth=10
uv run python scripts/generate_data.py --n-games 2000 --workers 6 --depth 10 --output data/stockfish_d10_2000g.npz

# Phase 2: supervised training (MPS strongly recommended — ~18× faster than CPU)
# d6: ~15 min for 30 epochs
uv run python scripts/train.py --data data/stockfish_d6_500g.npz   --epochs 30 --device mps --ckpt-dir checkpoints
# d10: ~40 min for 20 epochs
uv run python scripts/train.py --data data/stockfish_d10_2000g.npz --epochs 20 --device mps --ckpt-dir checkpoints_d10

# Phase 3: eval vs Stockfish 1320 (~50 min, 100 games at 800 sims, 5 parallel workers)
uv run python scripts/eval.py --ckpt checkpoints_d10/distilled_epoch019.pt --workers 5 --games-per-worker 20 \
    --sims 800 --stockfish-elo 1320
```
