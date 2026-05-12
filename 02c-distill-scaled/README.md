# 02c — Scaled-up Stockfish distillation

> **Sibling of 02b.** Same idea (supervised distillation from Stockfish,
> not real AlphaZero RL), but every dial turned up. The question: how
> close can pure distillation get to the teacher when we stop pinching
> pennies on data, targets, and capacity?

## What changes vs 02b

| Dial | 02b (d10 run, 1185 Elo) | 02c |
|---|---|---|
| Network | 10 blocks × 128 ch (~3M params) | **20 blocks × 256 ch (~21M params)** |
| Stockfish teacher | depth=10 (~2200 Elo) | **depth=15 (~2500 Elo)** |
| Policy target | hard one-hot of SF's chosen move | **multipv=8 softmax over top-8 SF moves** |
| Games generated | 2,000 | **4,000** |
| Positions | ~250k | ~300–400k |
| Eval sims | 800 | 800 (baseline) + 1600 (stretch) |

The single most important change is **multipv soft targets**: instead
of teaching the student "Stockfish played e4 here," we teach it
"e4 ≈ 38%, Nf3 ≈ 28%, c4 ≈ 18%, ..." Each position now carries the
teacher's full ranking, not just its argmax.

## How multipv soft targets work

For each visited position we call
`engine.analyse(board, depth=D, multipv=K)`. This is ONE Stockfish
search; internally SF keeps K principal variations alongside its normal
alpha-beta, so it's only ~1.3-1.7× slower than `multipv=1`. The K
returned PVs each have a centipawn score from the side-to-move's POV.
We softmax those scores at temperature T (in pawns):

```
p_i ∝ exp(score_i_cp / (100 * T))
```

with mate scores clipped to ±1000 cp so a forced mate doesn't collapse
the distribution. T=1 pawn means a 50cp gap → roughly a 62/38 split
between the two moves; large enough to be informative, small enough not
to encourage the student to play obviously-worse moves.

The training loss is a standard soft-target cross-entropy:

```
loss = -Σ p_target_i * log_softmax(logits)_i
```

scattered through the K-sparse target into the 4672-action vector.

## Why pick a bigger network

A 10×128 net (3M params) is genuinely too small to absorb a strong
teacher's positional knowledge. The 02b d10 run plateaued at 65.6%
top-1 even when the teacher was much stronger than the student —
classic capacity-bound symptom. The paper-sized 20×256 (~21M params)
gives the student room to actually fit teacher behavior.

## Network architecture

A standard AlphaZero-style ResNet with two heads. All shapes assume
batch size `B`, `f = n_filters`, `R = n_res_blocks`. Default is `f=256`,
`R=20`.

```
        input board planes
        (B, 19, 8, 8)
              │
              ▼
        ┌───────────────────────┐
        │ STEM                  │
        │  Conv 3×3  19 → f     │  (padding=1, stride=1)
        │  BatchNorm2d(f)       │
        │  ReLU                 │
        └───────────────────────┘
              │  (B, f, 8, 8)
              ▼
        ┌───────────────────────┐
        │ TRUNK — R × ResBlock  │
        │                       │
        │   x ──┬──────────────┐│   each ResBlock:
        │      │              ││     Conv 3×3 f→f
        │      ▼              ││     BN → ReLU
        │   Conv3x3 → BN → ReLU││     Conv 3×3 f→f
        │      │              ││     BN
        │      ▼              ││     + x  (residual)
        │   Conv3x3 → BN      ││     ReLU
        │      │              ││
        │      +──────────────┘│
        │      ▼ ReLU          │
        │   x_out              │
        └───────────────────────┘
              │  (B, f, 8, 8)
              │
              ├─────────────────────────┐
              ▼                         ▼
   ┌─────────────────┐        ┌─────────────────┐
   │ POLICY HEAD     │        │ VALUE HEAD      │
   │  Conv 1×1 f→73  │        │  Conv 1×1 f→1   │
   │  (B,73,8,8)     │        │  BatchNorm      │
   │  flatten        │        │  ReLU           │
   │                 │        │  flatten (B,64) │
   │                 │        │  FC 64→64 → ReLU│
   │                 │        │  FC 64→1 → tanh │
   ▼                 ▼        ▼                 ▼
 logits (B, 4672)            value (B,) ∈ [-1, +1]
```

**Move encoding (4672 = 8 × 8 × 73):**
- planes 0..55: queen-like (8 directions × 7 distances)
- planes 56..63: 8 knight jumps
- planes 64..72: underpromotion (3 pieces × 3 directions)
- queen-promotion lives in the queen-like planes (pawn forward to last rank)
- index = `plane * 64 + from_square`

**Board encoding (19 planes, white's POV):**
- 0..5: white P, N, B, R, Q, K (one-hot per square)
- 6..11: black P, N, B, R, Q, K
- 12: side-to-move (all-ones plane if white to move)
- 13..16: castling rights (W-king, W-queen, B-king, B-queen)
- 17: en-passant target (one-hot)
- 18: halfmove clock / 100 (no-progress count)

**Param count by config**

| Config | n_blocks | n_filters | ~Params | Used by |
|---|---:|---:|---:|---|
| 02b small | 10 | 128 | ~3.0M | 02b d10 (1185 Elo) |
| 02c default | 20 | 256 | ~21M | 02c-30ep run (1086 Elo) |

Per ResBlock: `2 × (f²·9 + 2f) ≈ 2f² · 9` (dominant term), so doubling
filters quadruples each block. Doubling blocks doubles trunk cost
linearly. 20×256 vs 10×128 ≈ 8× params, ≈ 16× FLOPs per forward pass.

## Checkpoint indexing

A `.pt` file alone is **not self-describing** — to load it you need to
know `(n_blocks, n_filters)` and the data provenance. The library
path-scheme used for datasets (see "Dataset library" below) is mirrored
for checkpoints:

```
checkpoints/
  sf-<v>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/        ← which data produced these weights
    net-<R>x<F>/                                ← which arch (R blocks × F filters)
      <run-id>/                                 ← target-shape + run timestamp
        manifest.json          ← {arch, data_path, training_args, git_sha}
        epoch_NNN.pt           ← state_dict only
        latest.pt              ← symlink to newest epoch
        train_history.json
```

**Current status: not yet implemented.** Today's ckpt dirs across the
repo are free-form (`checkpoints_v3c/`, `checkpoints_v4/`,
`aws_results/run02_30ep/`, …) and contain no architecture sidecar, so
loading any of them requires the user to pass `--n-blocks/--n-filters`
explicitly. The scheme above is the planned fix; see the open question
at the end of this README.

## Layout

```
src/azdistill_scaled/
    board.py, network.py, mcts.py, arena.py, config.py  — copied from 02b
    stockfish_data.py    — NEW: multipv-aware data generation
    train_supervised.py  — NEW: soft-CE loss against multipv distribution
scripts/
    generate_data.py     — CLI wrapper for data gen
    train.py             — CLI wrapper for training
    eval.py              — 100 games vs Stockfish UCI_Elo=1320 (same as 02b)
    run_overnight.sh     — full pipeline: generate → train → eval
data/, checkpoints/      — gitignored outputs
```

## Dataset library

Data generation now supports a structured library indexed by Stockfish
metadata, so multiple data runs coexist without filename collisions and
can be cross-referenced from checkpoints:

```
data/library/
  sf-<version>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/
    data.npz            ← assembled tensors (same schema as before)
    games.pgn           ← all games in PGN form
    metadata.json       ← {sf_version, depth, multipv, T, host, timestamps, …}
    chunks/             ← per-worker checkpoint files (every chunk_size games)
      worker_NN_chunk_MMMM.npz
      worker_NN.pgn
```

Workers flush a chunk every `--chunk-size` games (default 50), so a
crash (spot reclamation, OOM, anything) loses at most that many games
per worker. `--finalize-only` re-assembles whatever chunks exist into
`data.npz` + `games.pgn`. The legacy `--output flat.npz` mode is still
supported for backward compatibility.

## NPZ schema

The dataset is one compressed NPZ with five arrays:

```
states            (N, 19, 8, 8)  float32   board planes (same encoder as 02)
moves             (N,)           int64     SF's actually-played move idx (top-1)
zs                (N,)           float32   game outcome from STM POV at that ply
multipv_indices   (N, K)         int64     top-K SF candidate move indices
multipv_logprobs  (N, K)         float32   softmax-logprobs over those K moves
K                 ()             int32     value of K used (record-keeping)
```

Padding for positions with <K legal moves (rare, only deep endgames):
`multipv_indices[i, j] = -1` and `multipv_logprobs[i, j] = -inf`. The
training loss masks these out by construction (exp(-inf) = 0).

## Run

```bash
cd 02c-distill-scaled
uv sync

# Phase 1: data gen — depth-15 SF, multipv-8 soft targets, 4000 games
uv run python scripts/generate_data.py \
    --n-games 4000 --workers 6 --depth 15 --multipv 8 \
    --temperature-pawns 1.0 \
    --output data/sf_d15_mpv8_4000g.npz

# Phase 2: train — 20×256 network, 20 epochs on MPS
uv run python scripts/train.py \
    --data data/sf_d15_mpv8_4000g.npz \
    --epochs 20 --device mps \
    --n-blocks 20 --n-filters 256 \
    --ckpt-dir checkpoints/run01

# Phase 3: eval vs Stockfish UCI_Elo=1320 — 100 games, 800 sims (apples-to-apples
# with 02b's 1185-Elo result)
uv run python scripts/eval.py \
    --ckpt checkpoints/run01/distilled_epoch019.pt \
    --workers 4 --games-per-worker 25 --sims 800 \
    --n-blocks 20 --n-filters 256 \
    --stockfish-elo 1320
```

Or run the whole pipeline overnight:

```bash
bash scripts/run_overnight.sh
```

## Open questions

**Should we adopt the checkpoint-indexing scheme above?** Pros: every
ckpt becomes self-describing (auto-load without `--n-blocks`/
`--n-filters`); every model is traceable back to the exact dataset that
produced it; multiple runs on the same data don't collide. Cons:
breaking change to ckpt loading (old ckpts need an explicit migration
or a fallback path); more directory nesting. Open for decision before
the next training run.

## Expected outcome (pre-registration)

Going in, I'm predicting **real Elo ~1500-1700** — bigger net + better
targets buys roughly +300-500 Elo over 02b's 1185, but still short of
the teacher (≈2500). The remaining gap is mostly the "search at training
time" lever we're NOT pulling here (that would be AZ-style self-play
fine-tune on top of the distilled prior, multi-day GPU job — see plan
in main repo README).
