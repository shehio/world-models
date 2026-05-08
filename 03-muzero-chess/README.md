# 03 — MuZero on chess

MuZero is "AlphaZero, but the env is also learned." Same MCTS skeleton,
same self-play-with-search loop, but **after the root, every node in
the search tree lives in a learned latent space** — there is no
`board.push(move)` happening inside MCTS. The dynamics network produces
the next latent state and the predicted reward.

> **Status:** scaffold only. Builds on 02's board encoder. Modules are
> stubbed; nothing trains yet.

## Why this is interesting

In chess the env is exactly known and cheap. So *why learn dynamics?*
The answer isn't strength on chess — it's that the same algorithm runs
on Atari and Go and stochastic-reward problems where the env can't be
queried at search time. By implementing it on chess we can verify the
implementation against AlphaZero (which has a perfect model) and
isolate the contribution of the learned model from search & RL bugs.

## The three networks

```
representation  h_θ : real_obs    -> s_0            (an encoder)
dynamics        g_θ : (s_t, a_t)  -> (s_{t+1}, r_t) (recurrent rollout)
prediction      f_θ : s_t         -> (π_t, v_t)
```

Composed:
```
real obs ──h_θ──> s_0
                    │
                  f_θ ─> (π_0, v_0)            <- expand root, run MCTS
                    │
              (s_0, a) ──g_θ──> (s_1, r_0)
                                  │
                                f_θ ─> (π_1, v_1)
                                  │
                            (s_1, a') ──g_θ──> (s_2, r_1)  ...
```

After the root expansion, **MCTS never calls `chess.Board.push` again**.
Every successor in the tree is a tensor.

## Training: K-step unrolls

For each replay sample at game-time `t`, we unroll the model K=5 steps
forward through the *real* actions taken in self-play, and compare
predictions against three real targets at each step:

| Step k | Predict | Target |
|---|---|---|
| 0..K | π_{t+k} | MCTS visit distribution at game-time t+k |
| 0..K | v_{t+k} | n-step bootstrapped return from t+k |
| 1..K | r_{t+k-1} | actual reward from env at t+k-1 (in chess: 0 except terminal) |

Gradient through the dynamics network is scaled by 1/K so each unroll
contributes equally regardless of length. The representation network
only sees gradients from the k=0 step.

## Differences from AlphaZero (02)

| Aspect | AlphaZero (02) | MuZero (03) |
|---|---|---|
| Tree expansion | `board.push(move)` then encode | `g(s, a) -> s'` |
| Reward in tree | None until terminal | Predicted at every transition |
| Value normalization | Q ∈ [-1, 1] (game outcome) | Q ∈ min-max-normalized (rewards may not be bounded) |
| Network | one f(s) → (π, v) | three: h, g, f |
| Inputs | every position re-encoded | only the root position is encoded |
| Training data | (state, π, z) | (root_state, [actions], [πs], [target_values], [target_rewards]) |

## Layout

```
src/muzero/
    board.py        same input planes as 02; copied / shared lib
    networks.py     h, g, f as separate nn.Modules + tied wrapper
    mcts.py         PUCT MCTS in latent space + min-max value normalization
    selfplay.py     plays a game, stores trajectories not (s, π, z) tuples
    replay.py       trajectory buffer; samples (start_idx, K-step window)
    train.py        K-step unroll loss with gradient scaling
    arena.py        same as 02 (uses its own root encoder)
    config.py       hyperparameters
scripts/
    selfplay_loop.py
    eval_vs_random.py
tests/
    test_unroll.py  shapes through K-step unroll match expected
```

## Open questions / decisions deferred

- **Reuse 02's `board.py`** by importing or by copying? Probably copy
  for now — keeps each project self-contained and lets us fork the
  encoding without breaking 02. If divergence is small, factor a
  `chess_common` package later.
- **Reanalyze.** MuZero paper re-runs MCTS on old replay data with the
  current network for fresher targets. Skip for v1.
- **Value scaling.** Paper uses h(x) = sign(x)(√(|x|+1)−1)+εx for Atari;
  for chess values are in [-1, 1] so we can skip.
