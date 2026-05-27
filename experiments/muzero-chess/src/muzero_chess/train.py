"""K-step unrolled training step for MuZero.

The trick that makes MuZero work: every position is trained K times,
once at each step of an unrolled trajectory. So the dynamics function
sees its own outputs being used as inputs to subsequent dynamics calls.

Training sample (post-trajectory unfold):
  observation_0                       — the initial observation
  actions:    [a_1, a_2, ..., a_K]    — the K actions actually played
  pi_targets: [π_0, π_1, ..., π_K]    — MCTS visit distributions at each ply
  z_targets:  [z_0, z_1, ..., z_K]    — n-step bootstrapped value targets
  r_targets:  [r_1, r_2, ..., r_K]    — actual rewards observed at each step
                                        (chess: 0 always except ±1 at terminal)

Forward pass:
  s_0 = h(obs_0)            # representation: only called once
  (p_0, v_0) = f(s_0)
  for k = 1..K:
    (s_k, r_k_pred) = g(s_{k-1}, a_k)
    (p_k, v_k) = f(s_k)

Loss (averaged over batch and across K+1 steps):
  L_policy = mean( CE(p_k, π_k) )         k = 0..K
  L_value  = mean( MSE(v_k, z_k) )        k = 0..K
  L_reward = mean( MSE(r_k_pred, r_k) )   k = 1..K  (no reward at the root)
  total    = L_policy + L_value + L_reward
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def train_step(network, optimizer, batch: dict) -> dict[str, float]:
    """One SGD step on a K-step trajectory batch.

    `batch` (all on the same device):
      observation : (B, 19, 8, 8) float32
      actions     : (B, K)        int64 — actions taken from steps 1..K
      pi_targets  : (B, K+1, 4672) float32 — MCTS π at each step
      z_targets   : (B, K+1)      float32 — value targets at each step
      r_targets   : (B, K)        float32 — observed reward at steps 1..K
    """
    network.train()
    obs = batch["observation"]
    actions = batch["actions"]
    pi_t = batch["pi_targets"]
    z_t = batch["z_targets"]
    r_t = batch["r_targets"]
    B, K = actions.shape

    # Initial inference: encode the observation, predict at root.
    s, p0, v0 = network.initial_inference(obs)
    policy_loss = _policy_ce(p0, pi_t[:, 0])
    value_loss = F.mse_loss(v0, z_t[:, 0])
    reward_loss = obs.new_zeros(())     # no reward at the root

    # Unrolled steps 1..K.
    for k in range(K):
        s, r_pred, p_k, v_k = network.recurrent_inference(s, actions[:, k])
        # MuZero paper scales the loss at non-root steps by 1/K so the
        # representation network doesn't drown in dynamics gradients.
        scale = 1.0 / K
        policy_loss = policy_loss + scale * _policy_ce(p_k, pi_t[:, k + 1])
        value_loss = value_loss + scale * F.mse_loss(v_k, z_t[:, k + 1])
        reward_loss = reward_loss + scale * F.mse_loss(r_pred, r_t[:, k])

    total = policy_loss + value_loss + reward_loss

    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    return {
        "loss": float(total.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "reward_loss": float(reward_loss.item()),
    }


def _policy_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy with soft target distribution.

    target: (B, A) probability distribution (rows may be all-zero if no
    visits — in which case we skip those rows so log(0) doesn't blow up).
    """
    row_sum = target.sum(dim=-1)
    mask = (row_sum > 0).float()
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(target * log_probs).sum(dim=-1)
    denom = mask.sum().clamp(min=1.0)
    return (per_sample * mask).sum() / denom
