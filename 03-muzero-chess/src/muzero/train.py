"""K-step unroll training.

For one batch sample:
    s_0, π_0, v_0       = initial_inference(obs_root)
    loss  = CE(π_target[0], π_0) + MSE(v_target[0], v_0)

    for k in 1..K:
        s_k, r_{k-1}, π_k, v_k = recurrent_inference(s_{k-1}, action_{k-1})
        loss += CE(π_target[k], π_k) + MSE(v_target[k], v_k) + MSE(r_target[k-1], r_{k-1})

Gradient scaling:
  - representation network: full gradient.
  - dynamics network: gradients scaled by 1/K so deeper unrolls don't dominate.
  - the latent state s_k is also scaled by 0.5 each step before being
    fed into the next dynamics call (paper's "scale_gradient" trick).

NOT YET IMPLEMENTED.
"""
from __future__ import annotations


def train_step(model, optimizer, batch, cfg) -> dict[str, float]:
    """One SGD step. Returns per-component losses for logging."""
    raise NotImplementedError
