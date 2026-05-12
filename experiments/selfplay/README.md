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

### Batched MCTS: same algorithm, K leaves per network call

The diagram above is the **sequential** version (`run_mcts`): one
descent → one network call → one backup, looped N times. Each network
call is a batch-of-1, and each batch-of-1 wastes per-call overhead.

The **batched** version (`run_mcts_batched`) runs the same algorithm but
collects K leaves before the network call:

```
   sequential MCTS (run_mcts):                batched MCTS (run_mcts_batched):

     for _ in range(50):                        for _ in range(50 / K):
       descend ──> leaf                           descend K times in parallel:
       ┌───────────────┐                            leaf₁ ─┐
       │ network(leaf) │  ◄── 1 board                leaf₂ ─┤
       └───────┬───────┘                              ...   ├── ┌─────────────────┐
               ▼                                     leafₖ ─┘   │ network([leaf₁  │
       backup leaf → root                                       │          ...    │  ◄── K boards
                                                                │          leafₖ])│      one call
   total network calls: 50                                      └────────┬────────┘
   total wall: ~50 × (per-call overhead +                                ▼
                       per-board work)                          backup K paths

                                                  total network calls: 50/K
                                                  total wall: (50/K) × (per-call overhead +
                                                                         K × per-board work)
                                                  → faster when per-call overhead dominates
                                                  → measured: K=8 gives ~1.9× speedup
```

### Virtual loss: how K parallel descents avoid all picking the same path

If we just descended K times naively, every descent would pick the same
sequence of moves (same priors, same Q values, same PUCT scores).
We'd hit one leaf K times instead of K different leaves — no benefit.

**Virtual loss** is a tiny lie we tell ourselves during the batch:
while a descent is still in flight (not yet backed up) passing through
a child node, we *pretend* that descent already returned a "+1 from the
child's POV" reward. The child's Q goes up; the parent's negated view
of the child goes down; the next descent in the batch sees a less
attractive child and picks something else.

```
   K=4 parallel descents through a shared tree:

                     root
                    / | | \                  PUCT scores at root before
                   /  | |  \                 any in-flight descents:
                  /   | |   \
                a    b    c   d                  a: 0.40   ← highest
              0.40 0.35 0.30 0.20

   descent 1 picks a.  child a now has VL=1.
   PUCT scores re-evaluated:
                   a: -1.0+U   ← virtual loss made it less attractive
                   b: 0.35     ← now highest
                   c: 0.30
                   d: 0.20
   descent 2 picks b. child b now has VL=1. ... and so on.

   After all 4 descents are backed up:
     each child's VL is decremented from 1 back to 0.
     each child's visit_count and value_sum reflect the REAL value
     returned by the network, not the +1 lie.
```

```
   Effective PUCT during a batch (each child has REAL N, W and pending VL):

      Q(child, child's POV) = (W + VL) / (N + VL)   ← lie pulls Q up
      Q(child, parent's POV) = -Q(child, child's POV) ← so parent sees lower Q

   At VL=0 (no in-flight descents): identical to sequential MCTS.
   At VL=1, no real visits yet:
      Q(child, parent's POV) = -1
      → maximally unattractive, parent picks a sibling.
```

The lie is exactly cancelled when the descent backs up: we apply the
real value AND decrement virtual_loss. By the time MCTS returns,
every node has `virtual_loss == 0` again. The unit tests verify this.

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
| MCTS sims | 150 train, 80 eval | Up from 30/50 once batched MCTS unblocked the speedup |
| MCTS batch K | 8 parallel descents per network call | ~1.9× wall-clock speedup on Mac CPU at this net size |
| PUCT c | 1.5 | AZ used 1.25–4.0; 1.5 is mid-range |
| Dirichlet | α=0.3, ε=0.25 at root | Standard AZ values |
| Temperature | τ=1 first 30 plies, τ=0 after | Standard AZ |
| Loss | soft-target CE on policy + MSE on value | Adam, lr=1e-3, weight_decay=1e-4 |

## Layout

```
src/alphazero/
    board.py        chess.Board ↔ planes; move ↔ 4672-index; legal mask
    network.py      ResNet trunk + policy head + value head
    mcts.py         PUCT MCTS — `run_mcts` (sequential reference) + `run_mcts_batched` (K leaves per network call, with virtual loss)
    selfplay.py     plays a game, yields (state, π, z) per ply
    replay.py       fixed-size deque, samples to torch tensors
    train.py        one SGD step: CE(π) + MSE(z)
    arena.py        head-to-head matches, random/network/stockfish policies
    config.py       hyperparameters
scripts/
    selfplay_loop.py        single-process self-play loop (v1/v2)
    selfplay_loop_mp.py     multi-process self-play (v3a/b/c, v4); supports --pcr, --optimizer sgd, --lr-decay-iters
    eval_vs_random.py       sequential eval against random
    eval_vs_random_mp.py    parallel (5-worker) eval against random
    eval_high_sims.py       parallel eval vs Stockfish at high sim counts
    h2h_mp.py               parallel head-to-head between two checkpoints
    elo.py                  compute Elo vs random / Stockfish anchors
tests/
    test_board.py        move encoding round-trip
    test_network.py      forward pass shapes
    test_mcts.py         sim count, prior masking, terminal handling
    test_mcts_batched.py virtual_loss invariants, equivalence at K=1, speedup
    test_selfplay.py     one-game smoke
    test_arena.py        match runner with two random policies
```

## Run

```bash
cd 02-alphazero-chess
uv sync

# Tests (~10s, all 28 pass)
uv run python tests/test_board.py
uv run python tests/test_network.py
uv run python tests/test_mcts.py
uv run python tests/test_mcts_batched.py
uv run python tests/test_selfplay.py
uv run python tests/test_arena.py

# Training loop with batched MCTS (--batch-size 8)
uv run python scripts/selfplay_loop.py \
    --iters 30 --games-per-iter 8 --sims 150 --train-steps 200 \
    --batch-size 8 --eval-games 6 --eval-sims 80 --eval-every 2 \
    --max-plies 200 --device cpu

# Eval (after training)
uv run python scripts/eval_vs_random.py --ckpt checkpoints/net_iter029.pt --games 50

# Elo (anchors random=0; uses Stockfish if on PATH)
uv run python scripts/elo.py --ckpt checkpoints/net_iter029.pt \
    --games-vs-random 200 --games-vs-stockfish 50 --batch-size 8
```

To use the **sequential** MCTS (the textbook PUCT version) instead of
batched — for instance, to read it as a reference — pass `--batch-size 1`.

## Results

Five runs, each one fixing what the previous one taught us. Headline
numbers vs random:

| Run | Network | Wall | W/D/L | Elo (95% CI) |
|---|---|---|---|---|
| v1 (no batching, 10 min) | 5b × 64ch | 10 min | 16 / 75 / 9 of 100 | +24 [−44, +95] |
| v2 (batched MCTS) | 5b × 64ch | 75 min | 89 / 107 / 4 of 200 | +158 [+107, +215] |
| v3b (multi-process + sharded buffer + reduced overtraining) | 10b × 128ch | 4h+4h | 120 / 80 / 0 of 200 | +290 [+250, +330] (combined) |
| v3c (Playout Cap Randomization on top of v3b) | 10b × 128ch | +2.5h | 141 / 58 / 1 of 200 | +315 [+275, +375] (combined) |
| **v4 (paper-faithful SGD+LR-decay, from random)** | **10b × 128ch** | **7.4h** | **157 / 43 / 0 of 200** | **+368 [+312, +413]** direct |

v3c achieves **only 1 loss in 200 games against random** and adds
**+25 Elo over v3b** via PCR — KataGo's trick of using cheap reduced-sim
moves for play and reserving expensive full-sim moves for training-target
generation. **v4** then reproduces the v3c result *from random init* using
the paper's actual optimizer (SGD momentum 0.9 + step LR decay) —
**zero losses out of 200** vs random; head-to-head vs v3c is roughly
even (6W/67D/27L of 100). vs Stockfish UCI_Elo=1320 at 800 sims: 0/7/93,
abs Elo 744 — within noise of v3c's 0/8/92 (768).

Full Elo breakdown, the v3a plateau diagnosis, the diagnostic
head-to-heads that revealed it, and v4's three training phases (the
breakout after first LR decay; the misleading loss climb at very low LR
after the second): see [results.md](./results.md).

## Open questions / decisions deferred

- **MCTS batching.** Single-leaf inference is the bottleneck. Real AZ
  parallelizes via virtual loss + batched leaf eval. Worth doing if you
  want >50 sims at reasonable speed.
- **Symmetry augmentation.** Chess has no rotational symmetry — no flips.
- **Resignation threshold.** Skip for v1; play to termination.
- **8-step history.** Without it, threefold repetition is invisible to
  the network. Add later if draws-by-repetition become a problem.
