# world-models — AlphaZero on chess, three training-procedure ablations

A self-paced study of how the **training signal** changes what an
AlphaZero ResNet can learn — RL self-play, hard Stockfish targets, and
soft multipv Stockfish targets, evaluated on the same architecture.

The thesis: holding the network constant and varying only the
*training procedure* surfaces the real differences between
methods. Three sibling experiments, one shared chess core, one
parallel AWS datagen pipeline.

## Projects

| # | Project | Headline result | Latent | Planning | Training signal |
|---|---|---|---|---|---|
| **02** | [AlphaZero-chess](./experiments/selfplay/) | v4: +368 Elo vs random, 0 losses of 200 ([results](./experiments/selfplay/results.md)) | (true board state) | MCTS over real env | Self-play RL |
| **02b** | [Stockfish distillation](./experiments/distill-hard/) | Real Elo 1185 (19W/25D/56L vs SF1320) | same as 02 | MCTS at eval only | **Hard** targets from SF d10 |
| **02c** | [Scaled distillation](./experiments/distill-soft/) | **d15 ckpt: Elo ~1807** (+622 over 02b) — bigger teacher + 5M positions + GPU-EKS infra ([results](./experiments/distill-soft/results.md)) | same as 02 (bigger) | MCTS at eval only | **Multipv soft** targets from SF d10/d15 |

The 02 → 02b → 02c arc holds *architecture* constant (AlphaZero ResNet)
and varies the **training signal** (RL self-play → SF hard targets → SF
multipv soft targets). The conceptual differences and the ceilings each
hits are in
[DISTILLATION_VS_ALPHAZERO.md](./DISTILLATION_VS_ALPHAZERO.md).

## Layout

```
.
├── pyproject.toml                  Workspace root (uv workspace; one lock)
├── uv.lock
├── .env.example                    Template for AWS pipeline env vars
├── wm_chess/                       Shared chess core (board, network,
│                                     MCTS, arena, config, catalog,
│                                     merge tools) — used by all three
│                                     experiments below
├── experiments/
│   ├── selfplay/                   Self-play AlphaZero (5 progressive runs)
│   ├── distill-hard/               Distillation: 10×128 + hard targets
│   └── distill-soft/               Distillation: 20×256 + multipv soft targets
├── infra-eks/                      EKS-native parallel datagen + GPU training
│   ├── daemons/                      auto-eval (per-ckpt EC2) + pod-sync
│   └── launchers/                    bare-EC2 one-shots (L40S, deep-sims eval, …)
├── library/                        Indexed game library + auto-generated CATALOG
├── DISTILLATION_VS_ALPHAZERO.md    02 vs 02b vs 02c training procedures
├── EVALS.md                        How the auto-eval daemon measures Elo
├── dashboard.html                  Self-contained run-evolution dashboard
└── README.md                       you are here
```

The four `pyproject.toml` files (one at root + one per workspace
member) share a single `uv.lock` at the root. Run `uv sync
--all-packages --extra test` from the root once, then `uv run --project
<member> python -m pytest <member>/tests/` to test any one in isolation.

## Quick start

```bash
# 1. clone, then set up the workspace (single venv, single lock)
uv sync --all-packages --extra test

# 2. run any experiment's tests
uv run --project wm_chess           python -m pytest wm_chess/tests/
uv run --project experiments/selfplay python -m pytest experiments/selfplay/tests/

# 3. for AWS pipeline work, copy the env template and fill in your account
cp .env.example .env
# edit .env with your AWS_ACCOUNT_ID, bucket names, etc.

# 4. for end-to-end datagen on EKS, see infra-eks/README.md
```

## Navigation by interest

- **Want the AWS datagen pipeline** → [`infra-eks/README.md`](./infra-eks/README.md)
  — eksctl cluster, Docker image, Indexed Job parallel datagen on spot
  CPU, S3-staged shards, k8s-native cross-pod merge. The historical
  one-EC2-box Terraform pipeline is preserved as a writeup in
  [02c/README — AWS / cloud architecture](./experiments/distill-soft/README.md#aws--cloud-architecture).
- **Want how distillation runs are trained + evaluated** →
  [`EVALS.md`](./EVALS.md) (auto-eval daemon, UCI=1350 anchor, Elo math,
  multi-region fallback, `elo_bisect.py`),
  [`infra-eks/daemons/README.md`](./infra-eks/daemons/README.md) (the
  poller + per-checkpoint EC2),
  [`infra-eks/launchers/README.md`](./infra-eks/launchers/README.md)
  (one-shot bare-EC2 launchers for L40S training and ablation runs).
- **Want to see the network architecture** →
  [02c/README — Network architecture](./experiments/distill-soft/README.md#network-architecture)
  (Mermaid; renders inline on GitHub). Same architecture used across
  02 / 02b / 02c.
- **Want to understand multipv / softmax T / hard-vs-soft targets** →
  [02c/README — Vocabulary](./experiments/distill-soft/README.md#vocabulary-multipv-softmax-t-in-pawns-target-shape).
- **Want to see the negative result** → [02c/results.md](./experiments/distill-soft/results.md).
- **Want the AlphaZero self-play story (v1 → v5)** → [02/results.md](./experiments/selfplay/results.md).

## Why chess

- **Symbolic state** — skip perception, focus on search + value learning.
- **Deterministic transitions** — uniform across all three experiments.
- **Cheap to simulate** — `python-chess` does ~10⁶ legal-move generations/sec.
- **Honest evaluation is easy** — play vs Stockfish at fixed Elo, vs random, vs prior checkpoints.

Each experiment fits in hours on a Mac (with AWS as an option when an
experiment genuinely needs many CPUs or a GPU). Beating random and a
weak Stockfish baseline is the success criterion; reproducing
AlphaZero's published Elo is not.

## Tests + CI

GitHub Actions runs every workspace member's test suite on each push
to main (see `.github/workflows/ci.yml`).

| Package | Tests | Wall |
|---|---:|---:|
| `wm_chess` (shared core + catalog + merge tools + stubs) | 84 | ~3 s |
| `experiments/selfplay` | 42 | ~4 s |
| `experiments/distill-hard` | 6 | ~2 s |
| `experiments/distill-soft` | 80 | ~3 s |
| **Total** | **212** | **~12 s** |

Run any one with `uv run --project <member> python -m pytest
<member>/tests/`.

## References

- Silver et al., *Mastering Chess and Shogi by Self-Play (AlphaZero)* (2017) — https://arxiv.org/abs/1712.01815
- DeepMind blog, *AlphaZero: Shedding new light on chess, shogi, and Go* (2018) — https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-the-grand-games-of-chess-shogi-and-go/
