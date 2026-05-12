# Learning World Models — a three-project journey

A self-paced study of model-based reinforcement learning, working up from
the original "World Models" paper to MuZero, with chess as the testbed
for the latter two.

The thesis: world-model methods all share the same skeleton — *learn a
compressed state, learn how it evolves, plan or act in that learned
space* — but they make wildly different choices about **what the latent
is for**. Implementing them side-by-side makes those choices concrete.

| # | Project | Latent trained for | Planning | Env |
|---|---|---|---|---|
| 01 | [Ha & Schmidhuber, *World Models*](./01-ha-world-models/) | Pixel reconstruction (VAE) | None — reactive policy | CarRacing-v3 |
| 02 | [AlphaZero-chess](./02-alphazero-chess/) | (no latent — true board state) | MCTS over the real env | Chess |
| 03 | [MuZero-chess](./03-muzero-chess/) | Predicting value/policy/reward | MCTS in **learned latent space** | Chess |

The progression mirrors a real conceptual arc:

1. **Ha** shows you can learn a usable world model from random data, but
   the model is trained for reconstruction — it remembers things the
   policy doesn't care about. The controller is reactive.
2. **AlphaZero** drops perception entirely (chess is symbolic) and
   focuses on the other half of the story: search-augmented policy
   improvement via MCTS over the *true* environment.
3. **MuZero** is the synthesis. Like Ha, it learns a model. Like
   AlphaZero, it plans with MCTS. Unlike Ha, the latent is trained only
   to support value/policy/reward prediction — no decoder, no
   reconstruction loss. The planner never queries the real env after the
   first move.

## Status

- [x] **01** — runs end-to-end on Apple Silicon (~2h budget). Trained
  agent beats random on unseen seeds. See its [README](./01-ha-world-models/README.md).
- [x] **02b** — *not faithful to AlphaZero*: experimental sibling project
  on the merged `stockfish-distill` branch. Same network architecture
  trained by **supervised distillation from Stockfish self-play games**
  (hard one-hot targets) instead of pure RL. Result: **19 wins / 25 draws / 56
  losses out of 100 games against Stockfish UCI_Elo=1320** (estimated agent
  ~1185 Elo). Confirmed our network capacity was not the bottleneck for v3c
  — pure self-play just couldn't generate strong-enough training targets in
  the compute budget. See [02b/README](./02b-alphazero-stockfish-distill/README.md).
- [ ] **02c** — scaled distillation: 20×256 ResNet (paper-sized, ~21M
  params) trained on Stockfish d10 with **multipv=8 soft policy targets**
  (the full top-8 move distribution per position, not just SF's argmax).
  Tests whether bigger capacity + richer targets can narrow the
  student-teacher gap further. First 10-epoch run on AWS landed at real
  Elo ~998 (under 02b's 1185); 30-epoch follow-up is in progress.
  See [02c/README](./02c-distill-scaled/README.md).
- **How 02b/02c relate to real AlphaZero** is laid out in
  [DISTILLATION_VS_ALPHAZERO.md](./DISTILLATION_VS_ALPHAZERO.md) — same
  architecture, different training procedure, different ceiling.
- [x] **02** — implemented end-to-end through five progressive runs.
  v3c's stack: batched MCTS (K=8 parallel descents, virtual loss, 1.9×
  speedup), multi-process self-play (6 workers), KataGo-style sharded
  replay buffer with conservative train-steps:games ratio, and Playout
  Cap Randomization (cheap reduced-sim moves for play, expensive
  full-sim moves only for target generation). 32 unit tests pass.
  v3c Elo: **+315 vs random** combined estimate, 141/58/1 of 200.
  **v4** (the most recent run) reproduces v3c **from random init** with
  the paper's actual optimizer (SGD momentum 0.9 + step LR decay 0.02
  → 0.002 → 0.0002 at iters 30/60), 7.4h budget. **+368 Elo vs random
  direct, 157/43/0 of 200** (first run with zero losses at this scale).
  Head-to-head vs v3c is roughly equivalent (6W/67D/27L of 100), and
  vs Stockfish UCI_Elo=1320 at 800 sims is 0/7/93 (abs Elo 744, vs
  v3c's 768 — within noise). v4 confirms v3c's strength wasn't an
  Adam-specific artifact or a benefit of the cumulative resume chain.
  See [results.md](./02-alphazero-chess/results.md) for the full story
  including the v3a plateau, the AZ-clone fixes that broke it, the PCR
  +25 Elo, and v4's three training phases (the real breakout after the
  first LR decay; the misleading loss climb at very low LR).
- [ ] **03** — scaffold up; depends on 02 (shares MCTS, board encoder).

## Layout

```
.
├── 01-ha-world-models/   World Models (2018) on CarRacing-v3
├── 02-alphazero-chess/   AlphaZero on chess via python-chess
└── 03-muzero-chess/      MuZero on chess
```

Each subfolder is a self-contained project with its own `pyproject.toml`
and README. Run them independently with `uv`.

## Why chess for 02 and 03

- **Symbolic state**, so we can skip perception and focus on search +
  value learning.
- **Deterministic transitions**, which is what MuZero's deterministic
  dynamics network actually assumes.
- **Cheap to simulate** — `python-chess` does ~10⁶ legal-move generations
  per second, so MCTS isn't bottlenecked by env speed.
- **Honest evaluation** is easy: play vs. Stockfish at depth 1–3, vs.
  random, vs. each prior checkpoint.

The goal is *learning*, not SOTA. Each project should fit on a Mac in
hours, not weeks. Beating random + a weak Stockfish baseline is a
success criterion; reproducing AlphaZero's published Elo is not.

## References

- Ha & Schmidhuber, *World Models* (2018) — https://worldmodels.github.io/
- Silver et al., *Mastering Chess and Shogi by Self-Play (AlphaZero)* (2017) — https://arxiv.org/abs/1712.01815
- Schrittwieser et al., *MuZero* (2019) — https://arxiv.org/abs/1911.08265
