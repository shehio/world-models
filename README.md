# world-models — AlphaZero's network on one GPU

**[2,189 Elo](https://shehio.github.io/world-models/experiments/#stacking)**
on chess (vs Stockfish UCI=1,800, sims=4,000, CI [2,098, 2,354]) from
the same 20×256 ResNet AlphaZero used — supervised distillation from
Stockfish on 30M positions, one L40S GPU, ~16 GPU-hours. Roughly
**four orders of magnitude less compute** than the AlphaZero paper.

The live narrative — method, ablation pages, current Elo, infra, and
the Go pipeline — lives on the site:
**[shehio.github.io/world-models](https://shehio.github.io/world-models/)**.

## What's in here

Two games, four trainable pipelines, one shared core:

| Pipeline | Code | Headline |
|---|---|---|
| **Chess · soft distillation** | [`experiments/distill-soft/`](./experiments/distill-soft/) | **2,189 Elo** vs Stockfish UCI=1,800 at sims=4,000 (d10 30M ep 15). The headline pipeline. |
| **Chess · hard distillation** | [`experiments/distill-hard/`](./experiments/distill-hard/) | ~1,185 Elo from a 3M-param net on 250K positions — the simpler sibling, used as a soft-vs-hard ablation point. |
| **Chess · self-play RL** | [`experiments/selfplay/`](./experiments/selfplay/) | Faithful AlphaZero (v1–v4): +368 Elo vs random from scratch on a laptop. The starting point for the distill-then-RL loop. |
| **Go (9×9) · distillation** | [`experiments/distill-go/`](./experiments/distill-go/) | Parity with KataGo @ v200 in 15 epochs; 8×128 net on 1.236M KataGo positions. |

All four share `wm_chess/` (board, network, MCTS, arena, catalog,
merge tools), the same on-disk `.npz` schema, and the same datagen
infrastructure (`infra-eks/`: EKS Indexed Jobs + bare-EC2 launchers,
spot fallback across regions, multi-region auto-eval daemon).

## Layout

```
.
├── wm_chess/                 Shared core: board, network, MCTS, arena, catalog, merge tools
├── experiments/
│   ├── selfplay/             Faithful AlphaZero (v1–v4 self-play, PUCT-MCTS, ResNet)
│   ├── distill-hard/         Hard-target distillation from Stockfish d6/d10
│   ├── distill-soft/         Soft multipv distillation — the headline pipeline
│   ├── distill-go/           9×9 Go distilled from KataGo
│   └── distill-go-spike/     The one-day go spike that motivated distill-go
├── infra-eks/                EKS cluster manifests, Dockerfiles, daemons, bare-EC2 launchers
├── library/                  Indexed game library + auto-generated CATALOG.md
├── docs/notes/               Engineering notes — operational gotchas, infra patterns
├── site/                     Hugo site (the live narrative + ablation pages)
├── scripts/                  Cross-repo tooling (sync_experiments_log.py, ...)
├── EVALS.md                  Auto-eval daemon, UCI anchors, Elo math, bisection
├── EXPERIMENTS_LOG.md        Generated from site/content/experiments.md — do not edit by hand
└── README.md                 you are here
```

The chess packages (root, `wm_chess`, the three `experiments/*` chess
projects) form one `uv` workspace with a shared `uv.lock`; the Go
packages (`experiments/distill-go`, `experiments/distill-go-spike`)
are standalone with their own locks. Run
`uv sync --all-packages --extra test` from root once for the chess
side; the Go projects use `uv sync --extra test` from their own
directory.

## Quick start

```bash
# 1. one-shot workspace setup
uv sync --all-packages --extra test

# 2. tests for any one member
uv run --project wm_chess               python -m pytest wm_chess/tests/
uv run --project experiments/selfplay   python -m pytest experiments/selfplay/tests/
uv run --project experiments/distill-go python -m pytest experiments/distill-go/tests/

# 3. for AWS pipeline work
cp .env.example .env       # fill in your account / bucket names
```

End-to-end pipelines (datagen → training → eval) live under
`infra-eks/`. The launcher scripts in `infra-eks/launchers/` are the
quickest way to reproduce any single experiment on a bare EC2 box; the
EKS Indexed Jobs in `infra-eks/k8s/` are the parallel-datagen path.

## Tests + CI

GitHub Actions runs every workspace member's test suite on every push
to main (see [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)).
The same workflow regenerates `EXPERIMENTS_LOG.md` from the site and
fails if the result differs from the committed copy, so the two stay
in sync by construction.

| Package | Tests | Wall |
|---|---:|---:|
| `wm_chess` (shared core + catalog + merge tools) | 84 | ~17 s |
| `experiments/selfplay` | 49 | ~6 s |
| `experiments/distill-hard` | 6 | ~3 s |
| `experiments/distill-soft` | 93 | ~3 s |
| `experiments/distill-go` | 21 | ~13 s |
| **Total** | **253** | **~42 s** |

## References

- Silver et al., [*Mastering Chess and Shogi by Self-Play (AlphaZero)*](https://arxiv.org/abs/1712.01815) (2017)
- DeepMind blog, [*AlphaZero: Shedding new light on chess, shogi, and Go*](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-the-grand-games-of-chess-shogi-and-go/) (2018)
- Wu, [*Accelerating Self-Play Learning in Go*](https://arxiv.org/abs/1902.10565) — KataGo's playout-cap randomization (used in `experiments/selfplay/` and `experiments/distill-go/`)
- [Leela Chess Zero](https://lczero.org) — the distill-then-RL recipe this project is following
