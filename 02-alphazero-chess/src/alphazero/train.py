"""Training step.

Loss = cross_entropy(pi_target, policy_logits)
     + mse(z_target, value_pred)
     + L2 weight decay (handled by optimizer)

The policy CE is over all 4672 logits; pi_target is already zero on
illegal moves (came from MCTS visit counts) so no explicit mask is
needed in the target, but logits are masked at *inference* time.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations


def train_step(network, optimizer, batch) -> dict[str, float]:
    """One SGD step. Returns {'loss': ..., 'policy_loss': ..., 'value_loss': ...}."""
    raise NotImplementedError
