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

## Run

```bash
cd 02b-alphazero-stockfish-distill
uv sync

# Phase 1: generate Stockfish self-play data (~30-60 min wall)
uv run python scripts/generate_data.py --n-games 1000 --workers 6 --depth 8 --random-opening-plies 6

# Phase 2: supervised training (~30-60 min)
uv run python scripts/train.py --epochs 30 --batch-size 256

# Phase 3: eval vs Stockfish 1320 (~25 min, 100 games at 800 sims)
uv run python scripts/eval.py --games 100 --sims 800 --stockfish-elo 1320
```
