# distill-go

Full Go distillation pipeline. Mirrors the chess pipeline component-for-component:

| chess                          | go                                      |
|--------------------------------|-----------------------------------------|
| `wm_chess.stockfish_data`      | `distill_go.katago_data`                |
| `wm_chess.board` (chess rules) | `distill_go.board` (Go rules engine)    |
| `wm_chess.network`             | `distill_go.network` (param. by board)  |
| `wm_chess.mcts`                | `distill_go.mcts`                       |
| `distill_soft.train_supervised`| `distill_go.train`                      |
| `experiments/selfplay/eval.py` | `distill-go/scripts/eval.py`            |

The on-disk `.npz` schema is byte-identical to chess, so the chess
`merge_chunks.py` and S3 partial-sync entrypoint apply unchanged.
This package promotes the spike at `experiments/distill-go-spike/` to a real
pipeline with rules-aware MCTS, training, and an Elo gauntlet.

## Components built (top to bottom of dependency tree)

- `src/distill_go/board.py` — Go rules engine. Capture, suicide, positional
  superko, Tromp-Taylor scoring, pass / two-pass game end. Plus the 4-plane
  spike-compatible encoder and a 17-plane AlphaGo-Zero-style history encoder.
- `src/distill_go/config.py` — single hyperparameter struct (`GoConfig`).
- `src/distill_go/network.py` — `AlphaZeroGoNet`: stem + N residual blocks +
  policy head (linear to S²+1) + value head (tanh scalar). Parameterized for
  any board size and input-plane count.
- `src/distill_go/katago_data.py` — KataGo analysis-engine subprocess
  wrapper. Lifted from the spike, adapted to use the rules-aware `GoBoard`
  for `z` targets grounded in the actual game outcome.
- `src/distill_go/katago_data_parallel.py` — `mp.Pool` orchestrator with
  per-worker chunking and resume-past-existing-files (matches chess datagen
  semantics — see project memory note `feedback_datagen_resume`).
- `src/distill_go/mcts.py` — PUCT MCTS over `GoBoard`. Algorithm identical
  to `wm_chess.mcts.run_mcts`; only the per-game state type differs.
- `src/distill_go/train.py` — `GoMultipvDataset` + `train_step` with the
  same soft-CE-on-multipv loss as `distill_soft.train_supervised`.
- `src/distill_go/merge.py` — concatenate worker chunks into one .npz.
- `scripts/generate_data.py` / `scripts/train.py` / `scripts/eval.py` — CLIs.

## End-to-end demo (9×9, single Mac)

```bash
export KATAGO_BIN=$(which katago)
export KATAGO_MODEL=/opt/homebrew/Cellar/katago/1.16.4/share/katago/g170e-b20c256x2-s5303129600-d1228401921.bin.gz
export KATAGO_CONFIG=/opt/homebrew/Cellar/katago/1.16.4/share/katago/configs/analysis_example.cfg

# 1. Datagen: 4 workers × 25 games at 9×9 / 100 visits → ~5000 positions
uv run python scripts/generate_data.py \
    --workers 4 --games-per-worker 25 --board-size 9 --visits 100 \
    --chunk-size 5 \
    --output-dir /tmp/go9_data --merged-path /tmp/go9_merged.npz

# 2. Train: 4-block × 64-filter ResNet, 15 epochs
uv run python scripts/train.py \
    --data /tmp/go9_merged.npz --board-size 9 --n-input-planes 4 \
    --epochs 15 --batch-size 128 --n-blocks 4 --n-filters 64 \
    --ckpt-dir /tmp/go9_ckpt

# 3. Eval: 20 games vs KataGo at 50 visits → Elo diff
uv run python scripts/eval.py \
    --ckpt /tmp/go9_ckpt/distilled_epoch014.pt \
    --board-size 9 --n-input-planes 4 --n-blocks 4 --n-filters 64 \
    --student-sims 100 --katago-visits 50 \
    --games 20 --output /tmp/go9_eval.json
```

## 19×19 scale-up

The 9×9 demo uses everything in RAM. For 19×19 the data and model are both
large enough to need the chess-pipeline patterns this repo already has:

1. **Datagen**: same `generate_data.py`, but cluster-scale. Lift
   `infra-eks/Dockerfile` → `wm-go` image with KataGo + a model baked in,
   `cluster-gen-d15-250k-eu.yaml` → `cluster-gen-go-N-eu.yaml`,
   `k8s/job-gen-d15-250k.yaml` → `job-gen-go-N.yaml`. The chunk schema is
   already compatible, so `merge_chunks.py` and the partial-sync entrypoint
   transfer unchanged.
2. **Training**: swap `GoMultipvDataset` (in-RAM) for the streaming-extract-
   to-memmap loader in `distill_soft.train_supervised._extract_npz_to_memmap_dir`
   — at 19×19 the uncompressed `states` array is ~10× larger and won't fit
   in instance RAM.
3. **Network**: bump `n_blocks` and `n_filters` to AlphaGo-Zero size (~20×256).
4. **Eval**: same `eval.py`, just longer per-game wall-clock.

The only piece that scales non-trivially is the rules engine, which is
already O(board points) per move — fine for 19×19.

## Tests

```bash
uv run pytest tests/ -v
```

21 tests cover the rules engine (capture, suicide, ko, scoring, mask) and
network/MCTS shape smoke tests. No KataGo binary required for tests.

## Status (relative to chess pipeline parity)

- Pipeline: datagen + train + MCTS + eval all built and run end-to-end.
- Demo result: see `results.md` for the 9×9 Elo number.
- Infra scaffolding for 19×19 (Dockerfile, EKS Job, cluster) **not** built
  in this round — see "19×19 scale-up" above for the lift.
