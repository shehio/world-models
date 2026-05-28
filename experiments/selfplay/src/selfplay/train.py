"""One training step. Loss = policy CE + value MSE; weight decay via optimizer."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def train_step(network, optimizer, batch, teacher_net=None, kl_beta: float = 0.0) -> dict[str, float]:
    """Loss = policy CE + value MSE (+ optional kl_beta * KL(policy || teacher)).

    The KL term anchors the policy to a frozen reference (the distilled prior):
    RL fine-tuning can sharpen where search is confident, but pays a penalty for
    drifting off the broad supervised distribution. kl_beta=0 (default) disables
    it and the step is identical to the original.
    """
    states, pi_targets, z_targets = batch
    network.train()
    logits, value_pred = network(states)

    # Cross-entropy with soft targets: -sum(target * log_softmax(logits)) per sample, mean batch.
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi_targets * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred, z_targets)
    loss = policy_loss + value_loss

    kl_val = 0.0
    if teacher_net is not None and kl_beta > 0.0:
        with torch.no_grad():
            teacher_logits, _ = teacher_net(states)
            teacher_logp = F.log_softmax(teacher_logits, dim=-1)
        # KL(current || teacher), differentiable w.r.t. the current policy only.
        probs = log_probs.exp()
        kl = (probs * (log_probs - teacher_logp)).sum(dim=-1).mean()
        loss = loss + kl_beta * kl
        kl_val = float(kl.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "kl": kl_val,
    }
