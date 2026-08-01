# world-models — chess & go, distilled from Stockfish/KataGo on one GPU

**Chess**: [2,301 Elo](https://shehio.github.io/world-models/experiments/#peak-elo)
(R2 v2 ep 14 vs Stockfish UCI=1,800, sims=4,000, 95% CI [2,190, 2,601])
distilled from Stockfish on ~46M positions. Tighter-CI second-place:
2,153 Elo at sims=8,000 vs UCI=2,000 (CI ±70).
**Go (9×9)**: [≥ 2,366](https://shehio.github.io/world-models/go/)
(parity with KataGo @v=200, anchored to GnuGo L10) from 1.236M
KataGo-labeled positions. Both on **one L40S GPU per training run,
~16 GPU-hours each** — roughly three-to-four orders of magnitude less
compute than the AlphaZero training run (~10³× by device-hours, ~10⁴×
by FLOPs).

Live narrative + method + ablations + self-play postmortems on the
site: **[shehio.github.io/world-models](https://shehio.github.io/world-models/)**.

## What's in here

| Pipeline | Code | Headline |
|---|---|---|
| Chess · soft distillation | [`experiments/distill-soft/`](./experiments/distill-soft/) | **2,301 Elo** vs UCI=1,800 at sims=4,000 — R2 v2 ep 14, 95% CI [2,190, 2,601] |
| Chess · hard distillation | [`experiments/distill-hard/`](./experiments/distill-hard/) | ~1,185 Elo · soft-vs-hard ablation comparison point |
| Chess · self-play RL | [`experiments/selfplay/`](./experiments/selfplay/) | Faithful AlphaZero v1–v4 (+368 Elo vs random); from the distilled teacher, ungated **regresses ~370 Elo**, gated **holds but doesn't climb** ([postmortem](https://shehio.github.io/world-models/next/#selfplay-postmortem)) |
| Go (9×9) · distillation | [`experiments/distill-go/`](./experiments/distill-go/) | **≥ 2,366 Go Elo** (anchored to GnuGo L10) · 8×128 net on 1.236M KataGo-labeled positions |
| Go (9×9) · self-play RL | [`experiments/distill-go/scripts/selfplay_loop.py`](./experiments/distill-go/scripts/selfplay_loop.py) | First completed multi-iter self-play in the project · iter 42 H2H vs prior = 21W/19L (Elo Δ +17 ± 100, no improvement) |
| Chess · MuZero (learned dynamics) | [`experiments/muzero-chess/`](./experiments/muzero-chess/) | Negative result at 1-GPU compute: from-scratch caps ~700–900 Elo; distill-init ~1,700 after the MCTS sign-bug fix ([postmortem](https://shehio.github.io/world-models/next/#selfplay-gated-final)) |

All six share `wm_chess/` (board, network, MCTS, arena, catalog,
merge tools), the same on-disk `.npz` schema, and the same datagen +
training infrastructure (`infra-eks/`).

## Layout

```
.
├── wm_chess/                 Shared core: board, network, MCTS, arena, merge tools
├── experiments/
│   ├── selfplay/             Faithful AlphaZero (v1–v4 self-play, PUCT-MCTS, ResNet)
│   ├── distill-hard/         Hard-target distillation from Stockfish d6/d10
│   ├── distill-soft/         Soft multipv distillation — the headline pipeline
│   ├── muzero-chess/         MuZero on chess — learned dynamics, K-step unroll
│   ├── distill-go/           9×9 Go distilled from KataGo (+ selfplay_loop.py)
│   └── distill-go-spike/     The one-day go spike that motivated distill-go
├── infra-eks/                EKS manifests · Dockerfiles · daemons · bare-EC2 launchers
├── library/                  Indexed game library + auto-generated CATALOG.md
├── docs/notes/               Engineering notes — operational gotchas, infra patterns
├── site/                     Hugo site (the live narrative)
├── scripts/                  Cross-repo tooling (sync_experiments_log.py, ...)
├── EVALS.md                  Auto-eval daemon · UCI anchors · Elo math · bisection
├── EXPERIMENTS_LOG.md        Auto-generated from site/content/experiments.md
└── README.md                 you are here
```

The five chess packages (`wm_chess/` + 4 in `experiments/`) share one
`uv` workspace with a single `uv.lock` at root. The Go packages
(`experiments/distill-go`, `experiments/distill-go-spike`) are
standalone — they each have their own `uv.lock` and `uv sync` from
their own directory.

## Quick start

```bash
# Chess workspace
uv sync --all-packages --extra test
uv run --project wm_chess               python -m pytest wm_chess/tests/
uv run --project experiments/selfplay   python -m pytest experiments/selfplay/tests/

# Go (standalone)
cd experiments/distill-go && uv sync --extra test && uv run python -m pytest tests/

# AWS pipeline work
cp .env.example .env       # fill in account / bucket names
```

End-to-end pipelines (datagen → training → eval) live under
`infra-eks/`. The launchers in `infra-eks/launchers/` reproduce any
single experiment on a bare EC2 box; the EKS Indexed Jobs in
`infra-eks/k8s/` are the parallel-datagen path.

## Headline results

| | Number | Source |
|---|---|---|
| Chess · best point estimate | **2,301 Elo** (CI [2,190, 2,601]) | R2 v2 ep 14, sims=4,000, vs UCI=1,800 |
| Chess · tightest-CI measurement | 2,153 Elo (CI [2,084, 2,235]) | R2 v2 ep 4, sims=8,000, vs UCI=2,000 |
| Chess · self-play improvement so far | none — ungated (attempt #7) regressed to ~1,730; gated holds the teacher's ~2,101 with no candidate promoted | [postmortem](https://shehio.github.io/world-models/next/#selfplay-postmortem) |
| Go · 9×9 distillation lower-bound Elo | **≥ 2,366** (Go-Elo, GnuGo-anchored — *not* the AlphaGo-paper scale; [caveat](https://shehio.github.io/world-models/go/)) | 8×128 ep 15 = parity with KataGo @v200, anchored to GnuGo L10 |
| Go · self-play improvement | +17 ± 100 Elo over prior at iter 42 (24h, one L4 GPU) | h2h, 40 games, alternating colors |

## Experiment Tracking and Cloud Runs

Runs are tracked in [Weights & Biases](https://wandb.ai/shehio/world-models)
under entity `shehio`, project `world-models`. Tracking is **strictly
additive**: every script still writes the same `train_history.json`,
`.train_progress.json`, `run_metadata.json`, `history.json` and stdout it
always did, and every script still runs with wandb off. The shared helper
is `wm_chess/src/wm_chess/wandb_utils.py` (and a copy in
`experiments/distill-go/src/distill_go/wandb_utils.py`, since distill-go is
a standalone uv project). It returns `None` instead of a run whenever
tracking is unavailable, and callers guard every log call on that.

### What Is Instrumented

| Script | Hook site | Logged |
|---|---|---|
| `experiments/distill-soft/scripts/train.py` | epoch loop + `_write_train_progress` | the `train_history.json` epoch record verbatim; the sub-epoch progress payload under a `batch/` prefix |
| `experiments/distill-soft/scripts/eval.py` | end of match | W/D/L, score with 95% CI, Elo gap, absolute Elo with CI, as **summary** metrics |
| `experiments/distill-soft/scripts/elo_bisect.py` | each bisection probe | per-probe UCI Elo, score, running estimate, bracket; final Elo estimate with bracket as summary metrics |
| `experiments/distill-go/scripts/train.py` | epoch loop | the `train_history.json` epoch record verbatim |
| `experiments/distill-go/scripts/selfplay_loop.py` | per-iteration | games, positions, worker errors, self-play and train wallclock, buffer size, train losses (incl. the KataGo aux heads), eval vs random |
| `experiments/distill-go/scripts/h2h.py` | end of match | the JSON summary as summary metrics, with the CIs flattened to scalars |
| `experiments/selfplay/scripts/selfplay_loop.py` | per-iteration | games, mean plies, mean outcome, buffer, train losses, eval vs random |
| `experiments/selfplay/scripts/selfplay_loop_mp.py` | per-iteration | the above plus LR, gate score and promote/reject, Stockfish eval timing |
| `experiments/muzero-chess/scripts/run_local.py`, `run_cloud.py` | `on_iter_end` | the driver's `history` rows for that iter (game, losses, eval) flattened into one record |

Hyperparameters land in the run config automatically: `init_wandb` takes
the parsed argparse namespace, so every flag a script exposes is
recorded, plus derived values such as parameter count, position count and
multipv K.

### Running With and Without wandb

```bash
export WANDB_API_KEY=...            # once per machine

# tracked
uv run --project experiments/distill-soft python experiments/distill-soft/scripts/train.py \
    --data /work/data/multipv.npz --ckpt-dir /work/checkpoints/run01

# untracked, three equivalent ways
... scripts/train.py --no-wandb          # per-run flag
WANDB_MODE=disabled ... scripts/train.py # env var
                                         # or simply don't install wandb
```

Every script above takes `--no-wandb`. With wandb missing, uninstalled or
unreachable, `init_wandb` prints one notice and the run proceeds
untracked. `wandb` is declared in `wm_chess/pyproject.toml` (which the
four chess members all depend on) and in
`experiments/distill-go/pyproject.toml`; both `uv.lock` files are updated.

### Launching Sweeps

Sweep configs live in [`sweeps/`](./sweeps/), one per A/B question:

| Config | Question |
|---|---|
| `sweeps/distill-soft-targets.yaml` | soft multipv targets vs hard played-move targets, same data and net |
| `sweeps/distill-soft-value-weight-k.yaml` | value-loss weight (0.25 / 1 / 4) crossed with multipv K (4 / 8 / 16) |
| `sweeps/distill-go-network-size.yaml` | Go network size 6×96 vs 8×128 vs 10×160, via `train.py --arch BxF` |

```bash
wandb sweep --entity shehio --project world-models sweeps/distill-soft-targets.yaml
wandb agent shehio/world-models/<SWEEP_ID>     # run from the experiment dir
```

Each config carries the same usage comment inline. The K axis needs one
npz per K from `generate_data.py --multipv K`; the paths in the config are
placeholders to point at real data.

### Modal

[`experiments/distill-soft/modal_app.py`](./experiments/distill-soft/modal_app.py)
sends the exact same `scripts/train.py` to a Modal GPU, with a
`@app.local_entrypoint` that mirrors the local CLI flag for flag:

```bash
modal secret create wandb WANDB_API_KEY=...        # once
modal run modal_app.py --data /data/multipv.npz --epochs 20
modal run modal_app.py --gpu A100 --smoke          # wiring check, no data needed
```

Default GPU is `L40S`, matching the headline runs (one L40S, roughly 16
GPU-hours per training run). Pass `--gpu A100` where L40S is unavailable.
Data and checkpoints live on a Modal volume so they outlive the container.
This is a code path, not a wired-up deployment: it has not been executed
against a live Modal account from this repo.

### Smoke Mode

`experiments/distill-soft/scripts/train.py --smoke` swaps the real dataset
for a tiny synthetic in-memory one with the same fields and shapes. It
exercises the whole loop (batching, target shape, checkpointing, wandb
logging) in seconds without any data file. The loss numbers are noise by
construction and are only ever a wiring check.

Fireworks and Harbor are not applicable here: this repo trains and
evaluates its own small ResNets and MuZero networks, with no hosted LLM
inference and no Harbor-managed services.

## Tests + CI

GitHub Actions runs every workspace member's test suite on every push
to main (`.github/workflows/ci.yml`). A separate job regenerates
`EXPERIMENTS_LOG.md` from the site and fails if it diverges, so the
two stay in sync by construction.

| Package | Tests |
|---|---:|
| `wm_chess` (shared core) | 84 |
| `experiments/selfplay` | 57 |
| `experiments/distill-hard` | 6 |
| `experiments/distill-soft` | 104 |
| `experiments/muzero-chess` | 48 |
| `experiments/distill-go` | 56 |
| `scripts/` (sync tooling) | 14 |
| **Total** | **~369** |

## References

- Silver et al., [*Mastering Chess and Shogi by Self-Play (AlphaZero)*](https://arxiv.org/abs/1712.01815) (2017)
- Schrittwieser et al., [*Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)*](https://arxiv.org/abs/1911.08265) (2019)
- Wu, [*Accelerating Self-Play Learning in Go (KataGo)*](https://arxiv.org/abs/1902.10565) (2019)
- [Leela Chess Zero](https://lczero.org) — self-play AlphaZero engine; the prior-plus-self-play idea this project adapts (we build the prior by distilling Stockfish, not by self-play)

Comparison pages on the site: [vs AlphaZero](https://shehio.github.io/world-models/vs-alphazero/) · [vs Lc0](https://shehio.github.io/world-models/vs-leela/) · [vs MuZero](https://shehio.github.io/world-models/vs-muzero/)
