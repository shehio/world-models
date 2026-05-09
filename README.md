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
  instead of pure RL. Result: **19 wins / 25 draws / 56 losses out of
  100 games against Stockfish UCI_Elo=1320** (estimated agent ~1185 Elo).
  Confirmed our network capacity was not the bottleneck for v3c — pure
  self-play just couldn't generate strong-enough training targets in the
  compute budget. See [02b/README](./02b-alphazero-stockfish-distill/README.md).
- [x] **02** — implemented end-to-end through four progressive runs.
  Final agent (v3c) uses batched MCTS (K=8 parallel descents, virtual
  loss, 1.9× speedup), multi-process self-play (6 workers),
  KataGo-style sharded replay buffer with conservative train-steps:games
  ratio, and Playout Cap Randomization (cheap reduced-sim moves for play,
  expensive full-sim moves only for target generation). 32 unit tests
  pass.
  Final Elo: **+315 vs random [95% CI: +275, +375]** combined estimate
  across direct + h2h-via-v2 + h2h-via-v3b methods. **141W / 58D / 1L**
  out of 200 games against random. See
  [results.md](./02-alphazero-chess/results.md) for the full story
  including the v3a plateau, the AZ-clone fixes that broke it, and the
  PCR addition that pushed an extra +25 Elo.
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
