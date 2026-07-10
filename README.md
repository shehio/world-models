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
