# muzero-chess — MuZero on chess (learned dynamics)

The learned-world-model counterpart to the AlphaZero-family pipelines in
this repo. Where every other experiment runs MCTS over the *real*
environment (`python-chess` via `wm_chess.board`), this one trains
MuZero's three networks and searches over the **learned** dynamics:

```
h_θ (representation)   obs (B,19,8,8)      → latent (B,C,8,8)
g_θ (dynamics)         (latent, action)    → (next latent, reward)
f_θ (prediction)       latent              → (policy 4672, value)
```

MCTS descends `g_θ` instead of `board.push(move)`; training unrolls
K=5 steps through the dynamics with n-step value targets, per
Schrittwieser et al. 2019. Board encoding (19 planes), the 8×8×73 move
encoding, and the eval arena all come from `wm_chess`.

Narrative + comparison to the distillation results:
[vs MuZero](https://shehio.github.io/world-models/vs-muzero/) ·
[negative-results postmortem](https://shehio.github.io/world-models/experiments/#selfplay-muzero-negative).

## Two modes

1. **From-scratch** (`driver.train_loop`, `scripts/run_local.py` /
   `run_cloud.py`): tabula-rasa MuZero — self-play → replay buffer →
   K-step-unrolled train step → periodic Stockfish eval.
2. **Distill-init** (`distill_dynamics`, `scripts/run_distill_init.py`):
   factor the repo's 2,301-Elo distilled `AlphaZeroNet` into a *frozen*
   h + f (`teacher.py` — `f(h(obs))` reproduces the teacher bit-for-bit),
   and train **only** the dynamics g on teacher self-play transitions
   (latent MSE + value-equivalence through the frozen f + reward MSE).
   Root eval starts at teacher strength by construction; the question is
   whether search over the learned g preserves it.

## Results (negative, at 1-GPU compute)

| Run | Result |
|---|---|
| From-scratch, 64-ch latent, sims=400, ~200 iters on one L4 | capped **~700–900 Elo** vs SF UCI=1,350 — compute-starved without a teacher |
| Distill-init v1 | collapsed under MCTS (0/30 vs Stockfish at sims=800) while the distill loss plateaued — unrolled latents drift off the manifold the frozen f understands |
| Distill-init, after the MCTS sign-bug fix | **~1,700 Elo** — a real net, but still short of the ~2,101 teacher on the same harness |

The v1 collapse was partly an **inverted value-sign in MCTS child
selection** (fixed in `mcts.py`: children are ranked from the parent's
POV, negamax-style). Levers added while chasing the rest of the gap:
`latent_loss_weight` / `pred_loss_weight`, paper-faithful
`dynamics_grad_scale=0.5`, and `g_output_relu` (keep g's latents on the
teacher trunk's post-ReLU manifold). See the
[postmortem](https://shehio.github.io/world-models/experiments/#selfplay-muzero-negative)
for the full story.

## Layout

```
src/muzero_chess/
    config.py            MuZeroConfig + distill_init_config; encoding constants
    networks.py          h / g / f as one MuZeroNet
    mcts.py              PUCT MCTS over the learned dynamics (batched, top-K expansion)
    selfplay.py          plays a game with the learned model; returns a GameRecord trajectory
    replay.py            trajectory buffer; samples K-step unroll windows
    train.py             one gradient step: policy CE + value MSE + reward MSE over the unroll
    driver.py            train_loop: self-play → buffer → train → eval/ckpt
    teacher.py           factor a distilled AlphaZeroNet into frozen h + f
    distill_dynamics.py  distill-init: transition datagen + dynamics-only training
    eval.py              wraps the net as a wm_chess.arena Policy; Elo vs Stockfish UCI
scripts/
    run_local.py             CPU smoke of the from-scratch loop
    run_cloud.py             cloud entry point (S3 checkpoint sync; env-var hparams)
    run_distill_init.py      cloud entry point for distill-init
    smoke_distill_init.py    CPU smoke of distill-init with a random stand-in teacher
tests/                       48 tests — networks, MCTS, replay, self-play, teacher, train, eval
```

## Run

```bash
# From the repo root (workspace member — shares the root uv.lock)
uv sync --all-packages --extra test

# Tests
uv run --project experiments/muzero-chess python -m pytest experiments/muzero-chess/tests/

# CPU smoke: from-scratch loop wires up end-to-end, loss trends down
uv run --project experiments/muzero-chess python experiments/muzero-chess/scripts/run_local.py

# CPU smoke: distill-init plumbing with a random stand-in teacher
uv run --project experiments/muzero-chess python experiments/muzero-chess/scripts/smoke_distill_init.py
```

The real runs go through `infra-eks/entrypoint-muzero-chess.sh` and
`infra-eks/entrypoint-muzero-distill-init.sh` on GPU boxes.
