# World Models — hands-on, end-to-end

A small, runnable implementation of [Ha & Schmidhuber 2018, *World Models*](https://worldmodels.github.io/)
on Gymnasium's `CarRacing-v3`. Built in one ~3-hour session as a learning
project. Everything trains on a Mac with **MPS** (Apple Silicon GPU) — no
CUDA, no cluster.

## The big idea

Traditional model-free RL learns a single neural net that maps pixels →
actions, end-to-end, all from the reward signal. That's hard: the reward
is sparse, the network has to learn *perception*, *prediction*, and
*control* simultaneously, and pixel inputs are huge.

A **World Model** factors the agent into three pieces and trains them
**separately**:

```
            ┌──────────┐    z_t       ┌──────────┐                ┌──────────┐
 obs_t ───▶ │ V (VAE)  │ ──────────▶  │ M (MDN-  │ ─[h_t]────────▶│ C        │ ──▶ a_t
            │ encoder  │              │  RNN)    │                │ (linear) │
            └──────────┘              └──────────┘                └──────────┘
                  │ decoder
                  ▼
            obs_t reconstructed       (predicts z_{t+1} as a GMM)
```

| Module | What it learns | How it's trained | Params |
|---|---|---|---|
| **V** — VAE | "What does the world *look like*?" — compress 64×64×3 frames into a 32-dim latent. | Reconstruction loss (BCE) + KL on rollouts collected by a *random* policy. | ~4M |
| **M** — MDN-RNN | "How does the world *evolve*?" — given (z_t, a_t), predict a Gaussian-mixture distribution over z_{t+1}. | Negative log-likelihood on encoded rollouts. | ~700k |
| **C** — controller | "What should I *do*?" — map [z_t, h_t] → action. | CMA-ES on real-env returns. | **867** |

### Why this is interesting

1. **It works with random data.** V and M never see "good" gameplay — they
   learn from pure random exploration. Only C cares about reward.
2. **C is tiny.** A single linear layer, ~1k parameters. The hard work
   happened upstream. This is what makes evolution feasible — CMA-ES
   scales poorly past ~10k parameters.
3. **The agent can dream.** Once M is trained, you can ask it "what
   happens if I do X?" without touching the real environment. The famous
   demo: train C inside the dream, deploy in reality.

### Why a *mixture* density network

The world is non-deterministic from the agent's view. Approaching a
corner, the next frame might be "tree on left" or "tree on right" — a
single-Gaussian predictor would average those into "tree in the middle,"
which is wrong. A GMM with K components lets the model commit to multiple
plausible futures, weighted by mixing probabilities π_k.

### Why CMA-ES, not PPO

CarRacing's reward is *not* differentiable through the env. With a
~1k-parameter controller, evolution strategies are competitive with
gradient methods *and* don't need a rollout buffer or value estimator.
[CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) maintains a multivariate
Gaussian search distribution and updates its mean and covariance toward
better candidates each generation.

## Repo layout

```
src/world_models/
    config.py        single source of truth for hyperparameters
    env.py           gymnasium factory + frame preprocessing
    vae.py           V — encoder + decoder + reparam trick + loss
    mdn_rnn.py       M — LSTM with mixture-density head + sampling
    controller.py    C — tiny linear/MLP policy; flat-vector interface for CMA-ES
    data.py          frame dataset + (z, a) sequence dataset
    utils.py         device pick + seed utilities
scripts/
    collect_rollouts.py    random rollouts -> data/rollouts/*.npz
    train_vae.py           train V, save reconstruction grids
    encode_rollouts.py     V -> data/latents/*.npz
    train_rnn.py           train M on encoded sequences
    train_controller.py    CMA-ES on the live env
    play.py                full V+M+C agent in env, save video + GIF
    dream.py               side-by-side real vs. M-imagined rollout
```

## Running it

Order matters — each step consumes the artifact from the previous one.

```bash
uv sync                                 # one-time install
uv run python scripts/collect_rollouts.py     # ~3-5 min — saves data/rollouts/*.npz
uv run python scripts/train_vae.py            # ~15-20 min on MPS
uv run python scripts/encode_rollouts.py      # ~1 min — saves data/latents/*.npz
uv run python scripts/train_rnn.py            # ~10-15 min on MPS
uv run python scripts/train_controller.py     # ~10-15 min — CMA-ES
uv run python scripts/play.py                 # writes videos/play.mp4 + .gif
uv run python scripts/dream.py                # writes videos/dream.gif
```

All knobs (epochs, batch sizes, model sizes, CMA-ES population, etc.)
live in `src/world_models/config.py`. Bump them up to push the agent
further; the 2-3h budget here is intentionally tight.

## What the artifacts look like

- `checkpoints/vae_recon_epoch{N}.png` — top row real frames, bottom row
  V's reconstructions. The clearest signal that V is learning.
- `videos/play.gif` — full agent driving a CarRacing track.
- `videos/dream.gif` — left half: real frames; right half: M's
  imagination conditioned on the same actions. When M is good, the dream
  side stays road-coherent for tens of frames before drifting.

## Where to go next

- **More rollouts.** 10k frames is the bare minimum for V; 1M+ frames is
  what the original paper used.
- **Train C in dreams.** Replace `train_controller.py`'s real-env eval
  with an M-rollout. This is the headline demo of the paper.
- **Bigger M.** A multi-layer LSTM or a small transformer dramatically
  improves long-horizon hallucination quality.
- **Discrete envs.** `CarRacing-v3` with `continuous=False` simplifies
  the controller. Atari via `ale-py` works with the same V+M backbone if
  you swap in a stack of grayscale frames.

## References

- Ha & Schmidhuber, *World Models* (2018) — https://worldmodels.github.io/
- Hansen, *The CMA Evolution Strategy: A Tutorial* (2016)
- Kingma & Welling, *Auto-Encoding Variational Bayes* (2014)
- Bishop, *Mixture Density Networks* (1994)
