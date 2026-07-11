# distill-go-spike

A **spike** for porting the chess distillation pipeline to Go. Goal: prove
the KataGo-as-teacher integration produces data in the **same on-disk
schema** as our chess pipeline so the existing training loop, merge step,
and infra (EKS, S3, daemons) reuse with minimal changes.

Not a finished product. ~300 lines of Python + tests, written in one
sitting to de-risk the larger Go fork.

## What this shows

KataGo's **analysis engine** ([protocol docs](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md))
accepts JSON requests on stdin and emits JSON responses on stdout. For
each position we ask it to evaluate with N MCTS visits and return the
top-K candidate moves with their visit counts. We **softmax the visit
counts** to get a probability distribution — directly equivalent to the
chess pipeline's `softmax(cp / 100 * T)` over Stockfish multipv. That
soft policy becomes the student's target.

The data we write per position:

| field | shape (Go) | shape (chess equivalent) |
|---|---|---|
| `states` | `(n, 4, 19, 19)` float32 | `(n, 19, 8, 8)` float32 |
| `moves` | `(n,)` int64 (flat board idx, pass = 361) | `(n,)` int64 (chess move encoding) |
| `multipv_indices` | `(n, K)` int64 | `(n, K)` int64 |
| `multipv_logprobs` | `(n, K)` float32 | `(n, K)` float32 |
| `zs` | `(n,)` float32 | `(n,)` float32 |

The **K-dim policy + scalar value** structure is identical; only the
board shape and move space differ. That means:

- `merge_chunks.py` works as-is (chunk-size agnostic, key-list driven).
- The training loop in `experiments/distill-soft/src/distill_soft/train_supervised.py` works as-is once the network's input channels match.
- The whole `infra-eks/` cluster pattern — image build, gen Job spec,
  partial-sync entrypoint, autoeval daemon — lifts unchanged. We'd
  swap the container image (`wm-chess` → `wm-go`) and the teacher
  binary (`stockfish` → `katago`) inside it.

## What this does NOT show

- **No production self-play orchestration.** One game at a time, single
  process. The chess pipeline's worker parallelism + crash-safe chunking
  + S3 partial sync need to be lifted across to Go, but they're orthogonal
  to the data-format question this spike answers.
- **No real KataGo weights.** Set `KATAGO_BIN` and `KATAGO_MODEL` env vars
  before running `scripts/run_spike.py`. The unit tests in `tests/` cover
  the data-conversion logic with mocked responses, no binary needed.
- **No legality enforcement.** KataGo handles all rules (ko, suicide,
  scoring) upstream; the local `GoBoard` is bookkeeping only.
- **Simplified board representation.** 4 input planes (black stones,
  white stones, side-to-move, ones) vs. AlphaGo Zero's 17 (8-move
  history × 2 + side-to-move). History planes can be added incrementally
  without changing the training shape.

## Layout

```
src/distill_go_spike/
  board.py          # GoBoard + plane serialization + GTP coord parsing
  katago_data.py    # KataGoAnalysisEngine subprocess wrapper +
                    # analysis-response → soft-policy converter +
                    # play_one_game() orchestrator
scripts/
  run_spike.py            # Entry point: one game, writes one .npz
  run_spike_parallel.py   # N workers × M games each, per-worker chunked .npz output
tests/
  test_distill_go_spike.py   # 26 unit tests, KataGo-binary-free
```

## Running the tests (no KataGo needed)

```bash
cd experiments/distill-go-spike
uv run --project . pytest tests/ -v
```

All tests verify the data-conversion path with mocked KataGo responses.

## Running an actual game (needs KataGo)

```bash
# Install KataGo:
brew install katago     # macOS
# or build from https://github.com/lightvector/KataGo

# Download a network from https://katagotraining.org/networks/
# (~150 MB for the strongest networks; the b18c384 series is the current
# default. Smaller b6c96 nets are fine for the spike.)

export KATAGO_BIN=$(which katago)
export KATAGO_MODEL=/path/to/kata1-b18c384nbt-s9131461376-d4087399203.bin.gz

uv run --project . python scripts/run_spike.py \
    --board-size 9 --visits 400 --top-k 8 --output /tmp/go_spike.npz

# Should produce ~80-150 positions of training data in ~30-60s.
# Schema-verify with:
uv run --project . python -c "
import numpy as np
d = np.load('/tmp/go_spike.npz')
for k in d.files: print(k, d[k].shape, d[k].dtype)
"
```

## Next steps (out of scope for the spike)

1. **Add history planes.** Stack the last 8 board positions for each color
   → 17 input planes, matching AlphaGo Zero. Drop-in compatible with the
   spike's `play_one_game` (record + concat in `board_to_planes`).
2. **Self-play parallelism.** Mirror `stockfish_data.generate_dataset_parallel`:
   N workers, each spawning its own KataGo process, writing per-worker
   `worker_NN_chunk_NNNN.npz` chunks every chunk-size games.
3. **Containerize.** Build `Dockerfile.go-gen` with the KataGo binary + a
   network pre-baked, push to ECR as `wm-go`.
4. **EKS gen Job.** Copy `infra-eks/k8s/job-gen-d15-250k.yaml` →
   `job-gen-go-N.yaml`, swap image + entrypoint args, run it.
5. **Training.** Point `experiments/distill-soft/src/distill_soft/train_supervised.py`
   at a Go dataset (or fork to `experiments/distill-go/`). The network's
   first conv layer needs `n_input_planes` bumped from 19 → 4 (or 17 with
   history); everything else carries over.

The big questions the spike doesn't answer but the next step would:
- Does KataGo's MCTS-visit distribution transfer cleanly via soft targets,
  or do we want the raw network policy (`includePolicy=true`) instead?
- What temperature on the visit counts gives the best student?
- 9x9 first to validate the loop, or jump straight to 19x19?
