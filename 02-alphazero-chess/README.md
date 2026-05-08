# 02 — AlphaZero on chess

A small-scale, learning-first AlphaZero implementation. Self-play,
PUCT-MCTS, ResNet policy/value head, all on Apple Silicon (MPS).

> **Status:** scaffold only. Modules are stubbed with docstrings
> describing what they'll do; nothing trains yet.

## What AlphaZero is, in one paragraph

A single neural network `f_θ(s) → (π, v)` is trained to imitate the
**output of MCTS**, not the env's reward directly. At every move, MCTS
runs `N` simulations from the root, each descending the tree using the
PUCT formula `U(s,a) = Q(s,a) + c · π_θ(a|s) · √ΣN / (1+N(s,a))`, and
the visit-count distribution at the root becomes a *better* policy than
`π_θ` itself. That improved policy is the training target. Self-play
games provide the value target (game outcome). Repeat. The network gets
better, MCTS uses a better prior, MCTS gets stronger, the network
imitates a stronger target — the well-known policy-iteration loop.

## Design choices for this implementation

| Choice | Value | Why |
|---|---|---|
| Board encoding | 8×8×N input planes (piece-type × color, side-to-move, castling, no-progress) | Matches AlphaZero structurally, simplified (no 8-step history initially) |
| Move encoding | 8×8×73 = 4672 logits | AlphaZero's move-plane scheme: 56 queen-like + 8 knight + 9 underpromotion |
| Network | ~5 ResNet blocks × 64 channels | Learning-first; small enough to train in hours on MPS |
| MCTS sims/move | 50 (training), 200 (eval) | Tight budget; raise once the loop is verified |
| PUCT c | 1.5 | AlphaZero used 1.25–4.0 depending on game; 1.5 is a reasonable mid |
| Dirichlet noise at root | α=0.3, ε=0.25 | Standard AZ values; α typically scales as 10/avg_legal_moves |
| Replay buffer | 100k positions | Modest; hold last few thousand games |
| Batch | 256 | MPS-friendly |
| Eval | vs random, vs Stockfish depth-1, vs prior checkpoint | Three monotonic sanity bars |

## Layout

```
src/alphazero/
    board.py        chess.Board ↔ input planes; move ↔ 4672-index
    network.py      ResNet trunk + policy head + value head
    mcts.py         PUCT MCTS with Dirichlet root noise
    selfplay.py     plays a game with MCTS, yields (state, π, z) per move
    replay.py       fixed-size buffer of training samples
    train.py        loss = CE(π_target, π_pred) + MSE(z, v_pred) + L2
    arena.py        match two policies / eval vs random / vs Stockfish
    config.py       single source of truth for hyperparameters
scripts/
    selfplay_loop.py    alternate selfplay → train → checkpoint
    eval_vs_random.py   honest eval on N games
    play_human.py       UCI-ish loop you can play against
tests/
    test_board.py       round-trip move ↔ index, plane shapes
    test_mcts.py        single-sim sanity, value back-prop signs
```

## Run plan (when implemented)

```bash
uv sync
uv run python -m alphazero.scripts.selfplay_loop --games 200 --sims 50
uv run python -m alphazero.scripts.eval_vs_random --games 100
```

## Open questions / decisions deferred

- **Symmetry augmentation.** AlphaZero used board flips for Go; chess
  has no such symmetry, so we won't.
- **Resignation threshold.** Skip for v1; play games to termination.
- **Temperature schedule.** τ=1 for first 30 plies (sampling), τ→0 after
  (greedy). Standard AZ.
- **Stockfish dependency.** Need `stockfish` binary on PATH for the
  arena eval; skip that bar if not installed.
