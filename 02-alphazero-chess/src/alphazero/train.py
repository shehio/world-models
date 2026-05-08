"""One training step. Loss = policy CE + value MSE; weight decay via optimizer."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def train_step(network, optimizer, batch) -> dict[str, float]:
    states, pi_targets, z_targets = batch
    network.train()
    logits, value_pred = network(states)

    # Cross-entropy with soft targets: -sum(target * log_softmax(logits)) per sample, mean batch.
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi_targets * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred, z_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }
