# Learning World Models

A self-paced study of model-based RL: from the original *World Models*
paper through AlphaZero to MuZero, with chess as the testbed and two
distillation sibling projects that stress-test the *training procedure*
on a fixed architecture.

The thesis: world-model methods share one skeleton — *learn a state,
learn how it evolves, plan or act in that learned space* — but disagree
sharply on **what the latent is for**. Implementing them side-by-side
makes the disagreement concrete.

## Projects

| # | Project | Headline result | Latent | Planning | Training signal |
|---|---|---|---|---|---|
| **02** | [AlphaZero-chess](./experiments/selfplay/) | v4: +368 Elo vs random, 0 losses of 200 ([results](./experiments/selfplay/results.md)) | (true board state) | MCTS over real env | Self-play RL |
| **02b** | [Stockfish distillation](./experiments/distill-hard/) | Real Elo 1185 (19W/25D/56L vs SF1320) | same as 02 | MCTS at eval only | **Hard** targets from SF d10 |
| **02c** | [Scaled distillation](./experiments/distill-soft/) | Real Elo 1086 — **negative result**, bigger net + soft targets lost ([results](./experiments/distill-soft/results.md)) | same as 02 (bigger) | MCTS at eval only | **Multipv soft** targets from SF d10 |

The 02 → 02b → 02c arc holds *architecture* constant (AlphaZero ResNet)
and varies the **training signal** (RL self-play → SF hard targets → SF
multipv soft targets). The conceptual differences and the ceilings each
hits are in
[DISTILLATION_VS_ALPHAZERO.md](./DISTILLATION_VS_ALPHAZERO.md).

## What's in this repo

```
.
├── wm_chess/                       Shared chess core (board, network,
│                                     MCTS, arena, config) — used by all
│                                     three chess experiments below
├── experiments/
│   ├── selfplay/                   Self-play AlphaZero (5 progressive runs)
│   ├── distill-hard/               Distillation: 10×128 + hard targets
│   └── distill-soft/               Distillation: 20×256 + multipv soft targets
├── library/                        Indexed game library + auto-generated CATALOG
├── infra/                          Terraform: one-EC2-box AWS harness
├── DISTILLATION_VS_ALPHAZERO.md    02 vs 02b vs 02c training procedures
├── dashboard.html                  Self-contained run-evolution dashboard
└── README.md                       you are here
```

Each project under `experiments/` is a self-contained `uv` package that
imports the canonical board/network/MCTS/arena/config from
`wm_chess/`. Run them independently with `uv run ...`.

## Navigation by interest

- **Want to see the diagrams** → [02c/README — Network architecture](./experiments/distill-soft/README.md#network-architecture)
  (Mermaid; renders inline on GitHub). Same architecture used across
  02 / 02b / 02c.
- **Want the cheapest way to run the AWS pipeline** → [02c/README — Two-box split pipeline](./experiments/distill-soft/README.md#two-box-split-pipeline-the-cheapest-end-to-end)
  (~$1.15 end-to-end vs ~$9 on a single GPU box).
- **Want to understand multipv / softmax T / hard-vs-soft targets** →
  [02c/README — Vocabulary](./experiments/distill-soft/README.md#vocabulary-multipv-softmax-t-in-pawns-target-shape).
- **Want the AWS infra story** → [02c/README — AWS / cloud architecture](./experiments/distill-soft/README.md#aws--cloud-architecture).
- **Want to see the negative result** → [02c/results.md](./experiments/distill-soft/results.md).
- **Want the AlphaZero self-play story (v1 → v5)** → [02/results.md](./experiments/selfplay/results.md).

## Why chess (for 02 / 02b / 02c / 03)

- **Symbolic state** — skip perception, focus on search + value learning.
- **Deterministic transitions** — what MuZero's dynamics network assumes.
- **Cheap to simulate** — `python-chess` does ~10⁶ legal-move generations/sec.
- **Honest evaluation is easy** — play vs Stockfish at fixed Elo, vs random, vs prior checkpoints.

Each project fits in hours on a Mac (with AWS as an option when an
experiment genuinely needs a GPU). Beating random and a weak Stockfish
baseline is the success criterion; reproducing AlphaZero's published
Elo is not.

## Tests + CI

GitHub Actions runs all four test suites on every push to main
(see `.github/workflows/ci.yml`).

| Package | Tests | Wall |
|---|---:|---:|
| `wm_chess` (shared core + catalog) | 61 (+1 skipped) | ~3 s |
| `experiments/selfplay` | 42 | ~4 s |
| `experiments/distill-hard` | 6 | ~2 s |
| `experiments/distill-soft` | 63 | ~2 s |
| **Total** | **172** | **~11 s** |

Run any one with `uv run python -m pytest tests/` from that package's
directory.

## References

- Ha & Schmidhuber, *World Models* (2018) — https://worldmodels.github.io/
- Silver et al., *Mastering Chess and Shogi by Self-Play (AlphaZero)* (2017) — https://arxiv.org/abs/1712.01815
- Schrittwieser et al., *MuZero* (2019) — https://arxiv.org/abs/1911.08265
