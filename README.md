# Learning World Models — a chess and self-play journey

A self-paced study of model-based reinforcement learning, working up from
the original "World Models" paper to MuZero, with chess as the testbed
for the latter two — plus two siblings that stress-test the *training
procedure* on the same architecture (supervised distillation from
Stockfish, vanilla and scaled).

The thesis: world-model methods all share the same skeleton — *learn a
compressed state, learn how it evolves, plan or act in that learned
space* — but they make wildly different choices about **what the latent
is for**. Implementing them side-by-side makes those choices concrete.

| # | Project | Latent trained for | Planning | Env | Training signal |
|---|---|---|---|---|---|
| 01 | [Ha & Schmidhuber, *World Models*](./01-ha-world-models/) | Pixel reconstruction (VAE) | None — reactive policy | CarRacing-v3 | Random rollouts |
| 02 | [AlphaZero-chess](./02-alphazero-chess/) | (no latent — true board state) | MCTS over the real env | Chess | Self-play RL |
| 02b | [Stockfish distillation](./02b-alphazero-stockfish-distill/) | Same as 02 | MCTS at eval only | Chess | **Supervised** from SF hard targets |
| 02c | [Scaled distillation](./02c-distill-scaled/) | Same as 02 (bigger) | MCTS at eval only | Chess | **Supervised** from SF multipv soft targets |
| 03 | [MuZero-chess](./03-muzero-chess/) | Predicting value/policy/reward | MCTS in **learned latent space** | Chess | Self-play + dynamics fit |

How 02 vs 02b vs 02c relate — same network, different training
procedure, different ceiling — is laid out in
[DISTILLATION_VS_ALPHAZERO.md](./DISTILLATION_VS_ALPHAZERO.md).

## Status

- [x] **01** — runs end-to-end on Apple Silicon (~2h budget). Trained
  agent beats random on unseen seeds.
  See [01/README](./01-ha-world-models/README.md).

- [x] **02** — five progressive runs (v1 → v5). v3c was the strongest
  Adam-trained run; **v4 reproduced it from scratch with the paper's
  SGD+momentum+step-LR optimizer**: +368 Elo vs random direct, 157/43/0
  of 200 (first run with zero losses at this scale). 32 unit tests pass.
  Head-to-head vs Stockfish UCI_Elo=1320 at 800 sims: 0/7/93 (abs Elo
  744). See [02/results.md](./02-alphazero-chess/results.md) for the
  full v3a-plateau-then-PCR-then-v4 arc.

- [x] **02b** — same architecture, trained by **supervised distillation
  from Stockfish self-play games with hard one-hot targets**. At
  Stockfish depth=10: 19W/25D/56L of 100 vs SF UCI_Elo=1320 → **real Elo
  1185**. Confirmed network capacity was not the bottleneck for v3c —
  pure self-play just couldn't generate strong-enough training targets
  in the compute budget.
  See [02b/README](./02b-alphazero-stockfish-distill/README.md).

- [x] **02c** — **negative result, documented in full.** Scaled
  distillation: 20×256 ResNet (~21M params, paper-sized) trained on
  Stockfish d10 with **multipv=8 soft policy targets** for 30 epochs on
  AWS. Result: real Elo **1086** — ~130 Elo *behind* 02b's 1185 at the
  same teacher quality. Bigger net + richer targets, fully trained, lost
  to smaller net + hard one-hot. Three honest hypotheses (soft targets
  dilute signal at d10, undertrained-per-parameter, T=1 too smooth) and
  follow-up experiments (A/B/C/D) queued. Crash-safe data library +
  checkpoint indexing scheme designed after the run so the next one is
  reproducible end-to-end.
  See [02c/README](./02c-distill-scaled/README.md) and
  [02c/results.md](./02c-distill-scaled/results.md).

- [ ] **03** — scaffold up; depends on 02 (shares MCTS, board encoder,
  ResNet trunk).

## Layout

```
.
├── 01-ha-world-models/                  World Models (2018) on CarRacing-v3
├── 02-alphazero-chess/                  Self-play AlphaZero on chess (5 runs, v1..v5)
├── 02b-alphazero-stockfish-distill/     Distillation, 10×128 + hard targets
├── 02c-distill-scaled/                  Distillation, 20×256 + multipv soft targets
├── 03-muzero-chess/                     MuZero (scaffold only)
├── infra/                               Terraform: one-EC2-box AWS experiment harness
├── DISTILLATION_VS_ALPHAZERO.md         How 02 / 02b / 02c training procedures differ
├── dashboard.html                       Self-contained run-evolution dashboard
└── README.md
```

Each `0*` subfolder is a self-contained project with its own
`pyproject.toml` and README. Run them independently with `uv`.

## Why chess for 02 / 02b / 02c / 03

- **Symbolic state**, so we can skip perception and focus on search +
  value learning.
- **Deterministic transitions**, which is what MuZero's deterministic
  dynamics network actually assumes.
- **Cheap to simulate** — `python-chess` does ~10⁶ legal-move generations
  per second, so MCTS isn't bottlenecked by env speed.
- **Honest evaluation** is easy: play vs Stockfish at depth 1–3, vs
  random, vs each prior checkpoint.

The goal is *learning*, not SOTA. Each project should fit on a Mac in
hours, not weeks (with AWS as an option when an experiment genuinely
needs a GPU). Beating random + a weak Stockfish baseline is a success
criterion; reproducing AlphaZero's published Elo is not.

## Infrastructure

[`infra/`](./infra/) is a single Terraform module that provisions one
EC2 instance for running the 02c pipeline (or any other project that
needs a GPU): default VPC, single security group, AZ-filtered subnet,
AWS Deep Learning Base GPU AMI, spot-or-on-demand toggle, idempotent
first-boot bootstrap that installs Stockfish + uv. `var.instance_type`
is the single dial for scale. Architecture diagram and trade-offs are
in
[02c/README § AWS / cloud architecture](./02c-distill-scaled/README.md#aws--cloud-architecture).

## References

- Ha & Schmidhuber, *World Models* (2018) — https://worldmodels.github.io/
- Silver et al., *Mastering Chess and Shogi by Self-Play (AlphaZero)* (2017) — https://arxiv.org/abs/1712.01815
- Schrittwieser et al., *MuZero* (2019) — https://arxiv.org/abs/1911.08265
