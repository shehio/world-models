# 02 — AlphaZero on chess

A small-scale, learning-first AlphaZero implementation. Self-play,
PUCT-MCTS, ResNet policy/value head.

## What AlphaZero is

One neural network `f_θ(s) → (π_θ, v_θ)` is trained to imitate the
**output of MCTS**, not the env's reward directly. At every move, MCTS
runs `N` simulations from the root, each descending the tree using the
PUCT formula. The visit-count distribution at the root becomes a
*better* policy than `π_θ`. That improved policy is the training target.
Self-play games provide the value target (game outcome). Repeat — the
network gets better, MCTS uses a better prior, MCTS gets stronger, the
network imitates a stronger target. Policy iteration.

## How the pieces fit together

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     OUTER LOOP (per iteration)                   │
  │                                                                  │
  │   ┌─────────────────────┐         ┌────────────────────────┐     │
  │   │  G self-play games  │ ──────▶ │  replay buffer of      │     │
  │   │  (MCTS at every     │         │   (state, π, z) tuples │     │
  │   │   move)             │         └──────────┬─────────────┘     │
  │   └─────────────────────┘                    │                   │
  │             ▲                                ▼                   │
  │             │                  ┌────────────────────────┐        │
  │             │                  │  train_steps:           │       │
  │   ┌─────────┴───────────┐      │  sample batch from      │       │
  │   │ updated network θ'  │ ◀────│  buffer, minimize       │       │
  │   └─────────────────────┘      │  CE(π,π_θ) + MSE(z,v_θ) │       │
  │                                └─────────────────────────┘       │
  │                                                                  │
  │   periodic eval: arena vs random / vs Stockfish / save ckpt      │
  └──────────────────────────────────────────────────────────────────┘
```

### One self-play move (the inner loop of "G self-play games")

```
   board (chess.Board)
        │
        ▼
   ┌─────────────────────────────────────────────────────────┐
   │            run_mcts(board, network, N sims)             │
   │                                                         │
   │   create root from network forward pass                 │
   │   add Dirichlet(α) noise to root priors                 │
   │                                                         │
   │   for _ in range(N):                                    │
   │     ┌───────────── SELECT ─────────────┐                │
   │     │ descend by PUCT until a leaf:    │                │
   │     │   score(child) =                 │                │
   │     │     -Q(child)                    │                │
   │     │     + c · P(child) · √N(p)       │                │
   │     │              ──────────          │                │
   │     │              (1 + N(child))      │                │
   │     └──────────────────┬───────────────┘                │
   │                        ▼                                │
   │     ┌───────────── EVALUATE ───────────┐                │
   │     │ if terminal: v = ±1 / 0          │                │
   │     │ else: expand leaf via network    │                │
   │     │       v = value head output      │                │
   │     └──────────────────┬───────────────┘                │
   │                        ▼                                │
   │     ┌───────────── BACKUP ─────────────┐                │
   │     │ walk path leaf → root:           │                │
   │     │   each node += v;  v = -v        │                │
   │     │ (sign flips: adjacent plies have │                │
   │     │  opposite players to move)       │                │
   │     └──────────────────────────────────┘                │
   │                                                         │
   │   return visit counts at root                           │
   └────────────────────┬────────────────────────────────────┘
                        ▼
   π = visit_counts / Σ visit_counts          ← TRAINING TARGET (policy)
   move = sample(π) if ply<30 else argmax(π)  ← actually played move
   board.push(move)
```

### How the network sees a position

```
   chess.Board
       │
       ▼  encode_board (board.py)
   ┌──────────────────────────────────────────────────┐
   │  19 input planes, 8x8 each (always white's POV)  │
   │                                                  │
   │   0..5   white pieces (P, N, B, R, Q, K)         │
   │   6..11  black pieces (P, N, B, R, Q, K)         │
   │   12     side-to-move (1=white, 0=black)         │
   │   13..16 castling rights (W-K, W-Q, B-K, B-Q)    │
   │   17     en-passant target square (one-hot)      │
   │   18     halfmove clock / 100                    │
   └──────────────────────────────────────────────────┘
       │
       ▼  AlphaZeroNet (network.py)
   ┌──────────────────────────────────────────────────┐
   │  Conv(64, 3x3) + BN + ReLU                       │
   │  ResBlock × 5                                    │
   ├──────────────────────────┬───────────────────────┤
   │  policy head             │  value head           │
   │  Conv(73, 1x1)           │  Conv(1, 1x1) + BN    │
   │  flatten → 4672 logits   │  Linear(64,64) + ReLU │
   │                          │  Linear(64,1) + tanh  │
   │                          │  → scalar in [-1, 1]  │
   └──────────────────────────┴───────────────────────┘
```

### How a move maps to one of 4672 indices

```
   from-square (0..63) × move-plane (0..72) → 4672 actions

   ┌─────────────────────────────────────────────┐
   │ planes 0..55  : QUEEN-LIKE                  │
   │   8 directions (N, NE, E, SE, S, SW, W, NW) │
   │   × 7 distances (1..7)                      │
   │   = 56                                      │
   │   covers king moves, sliding pieces, pawn   │
   │   pushes/captures, queen-promotions         │
   ├─────────────────────────────────────────────┤
   │ planes 56..63 : KNIGHT (8 jumps)            │
   ├─────────────────────────────────────────────┤
   │ planes 64..72 : UNDERPROMOTION              │
   │   3 pieces (N, B, R) × 3 directions         │
   │   (capture-left, push, capture-right) = 9   │
   └─────────────────────────────────────────────┘

   action_idx = plane * 64 + from_square
```

## Why the value sign flips on backup

Each node stores `value_sum` from the POV of the player **whose turn it
is at that node**. White-to-move nodes track white's expected return;
black-to-move nodes track black's. They alternate every ply.

When we get a value `v` at a leaf and propagate it upward:

```
              parent (white to move)
                    ▲   add: -v_leaf
                    │
              child (black to move)
                    ▲   add: +v_leaf
                    │
              leaf (white to move)   v_leaf = network value at leaf
```

For PUCT selection the parent wants Q from its own POV, so it negates
the child's stored Q:

```
   Q_at_parent_pov = -value_sum(child) / visit_count(child)
```

This is the same negamax trick chess engines have used since the 1950s,
just dressed up with a tree search.

## Design choices

| Choice | Value | Why |
|---|---|---|
| Board encoding | 19 planes (piece × color, side, castling, EP, no-progress) | Simplified AlphaZero; no 8-step history |
| Orientation | Always white's POV; side-to-move plane | Avoids a class of mirroring bugs around castling/EP |
| Move encoding | 8×8×73 = 4672 | AlphaZero scheme; queen-promo encoded as queen-like |
| Network | 5 ResNet blocks × 64 ch (~390k params) | Learning-first; trains in minutes on Mac CPU |
| Inference device | CPU | Batch-1 MCTS calls are 3× faster on CPU than MPS for this net size |
| MCTS sims | 30 train, 50 eval | Tight; raise once the loop is verified |
| PUCT c | 1.5 | AZ used 1.25–4.0; 1.5 is mid-range |
| Dirichlet | α=0.3, ε=0.25 at root | Standard AZ values |
| Temperature | τ=1 first 30 plies, τ=0 after | Standard AZ |
| Loss | soft-target CE on policy + MSE on value | Adam, lr=1e-3, weight_decay=1e-4 |

## Layout

```
src/alphazero/
    board.py        chess.Board ↔ planes; move ↔ 4672-index; legal mask
    network.py      ResNet trunk + policy head + value head
    mcts.py         PUCT MCTS, Dirichlet root noise, temperature select
    selfplay.py     plays a game, yields (state, π, z) per ply
    replay.py       fixed-size deque, samples to torch tensors
    train.py        one SGD step: CE(π) + MSE(z)
    arena.py        head-to-head matches, random/network/stockfish policies
    config.py       hyperparameters
scripts/
    selfplay_loop.py    iters of self-play → train → eval
    eval_vs_random.py   honest eval against random
    elo.py              compute Elo vs random / Stockfish anchors
tests/
    test_board.py       move encoding round-trip
    test_network.py     forward pass shapes
    test_mcts.py        sim count, prior masking, terminal handling
    test_selfplay.py    one-game smoke
    test_arena.py       match runner with two random policies
```

## Run

```bash
cd 02-alphazero-chess
uv sync

# Tests (~5s)
uv run python tests/test_board.py
uv run python tests/test_network.py
uv run python tests/test_mcts.py
uv run python tests/test_selfplay.py
uv run python tests/test_arena.py

# Training loop
uv run python scripts/selfplay_loop.py \
    --iters 10 --games-per-iter 6 --sims 30 --train-steps 100 \
    --eval-games 10 --device cpu

# Eval (after training)
uv run python scripts/eval_vs_random.py --ckpt checkpoints/net_iter009.pt --games 50

# Elo (anchors random=0; uses Stockfish if on PATH)
uv run python scripts/elo.py --ckpt checkpoints/net_iter009.pt --games-vs-random 100
```

## Results

After a **10-minute training run** (8 iters × 6 games × 25 sims × 80
train steps), the agent's Elo over **100 games vs random** is
**+24 Elo [95% CI: −44, +95]** — within noise of random. Loss decreased
monotonically from 4.18 to 2.42, so the loop works; the agent is just
under-trained. Full breakdown including Stockfish baselines: see
[results.md](./results.md).

| Opponent | N | W/D/L | Score | Elo |
|---|---:|---:|---:|---:|
| Random (anchor 0) | 100 | 16/75/9 | 0.535 | **+24 [−44, +95]** |
| Stockfish skill=0 d=1 | 30 | 0/8/22 | 0.133 | gap −325 |
| Stockfish UCI_Elo=1320 | 30 | 0/3/27 | 0.050 | upper bound ≈ 990 |

## Open questions / decisions deferred

- **MCTS batching.** Single-leaf inference is the bottleneck. Real AZ
  parallelizes via virtual loss + batched leaf eval. Worth doing if you
  want >50 sims at reasonable speed.
- **Symmetry augmentation.** Chess has no rotational symmetry — no flips.
- **Resignation threshold.** Skip for v1; play to termination.
- **8-step history.** Without it, threefold repetition is invisible to
  the network. Add later if draws-by-repetition become a problem.
