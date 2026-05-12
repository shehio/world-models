"""Supervised distillation training with MULTIPV SOFT POLICY TARGETS.

Difference from 02b
-------------------
02b loss: F.cross_entropy(logits, hard_move_idx) — one-hot target on the
single move Stockfish played.

02c loss: cross-entropy between Stockfish's softmax-over-top-K distribution
and the network's softmax-over-4672. Implemented as:

    loss = -sum_i  p_target[i] * log_softmax(logits)[i]

The target distribution p_target is sparse — nonzero on only K≈8 actions
per position. We scatter those K probs into a dense vector at batch time.

Why soft targets help
---------------------
- Hard targets say "this is the move." That's 1 bit of position-level info.
- Soft targets say "e4 = 38%, Nf3 = 28%, c4 = 18%, others < 10%." That's
  log2(K!) ≈ a few bits per position about the ENTIRE TOP RANKING, not
  just the choice. The student learns "if I had to pick a non-best move,
  here's the order of preference," not just "this one is correct."

Top-1 accuracy comparison
-------------------------
We still compute "did network's argmax == Stockfish's actual played move"
as a sanity number. With soft targets it's typically a bit lower than
with hard targets (the network is hedging across the top-K), but the
*play* is stronger because the policy is better-shaped.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .board import N_POLICY


class MultipvDataset(Dataset):
    """Loads (state, multipv_indices, multipv_logprobs, played_move, z)
    from an NPZ created by 02c's stockfish_data.generate_dataset_parallel.

    Returns numpy arrays; the train loop tensorizes them.
    """

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.states = d["states"]
        self.moves = d["moves"]              # played move (top-1) per position
        self.multipv_indices = d["multipv_indices"]    # (N, K) int64
        self.multipv_logprobs = d["multipv_logprobs"]  # (N, K) float32
        self.zs = d["zs"]
        self.K = int(d["K"]) if "K" in d.files else self.multipv_indices.shape[1]

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, i: int):
        return (
            self.states[i],
            self.multipv_indices[i],
            self.multipv_logprobs[i],
            self.moves[i],
            self.zs[i],
        )


def _scatter_sparse_target_to_dense(
    mpv_indices: torch.Tensor,   # (B, K) long, may contain -1 for padding
    mpv_logprobs: torch.Tensor,  # (B, K) float, may contain -inf for padding
    n_actions: int,
    device,
) -> torch.Tensor:
    """Convert (indices, logprobs) sparse representation into a dense
    (B, n_actions) probability tensor. Padding (-1 index / -inf logprob)
    contributes zero probability.

    Returns a dense prob distribution. Each row sums to 1 (assuming the
    sparse logprobs were a valid softmax — which they are by construction
    in stockfish_data.py).
    """
    B, K = mpv_indices.shape
    # Convert logprobs → probs. Padding entries (-inf) → exp(-inf) = 0. ✓
    probs_k = mpv_logprobs.exp()  # (B, K)

    # We need to scatter these into (B, n_actions). For padded entries
    # (index=-1) we'd index out of range — replace index with 0 AND prob with 0
    # so scatter_add is a no-op for those slots.
    valid = mpv_indices >= 0
    safe_indices = mpv_indices.clamp(min=0)        # replace -1 with 0
    safe_probs = probs_k * valid.float()           # zero out padding

    dense = torch.zeros((B, n_actions), dtype=probs_k.dtype, device=device)
    dense.scatter_add_(1, safe_indices, safe_probs)
    return dense


def train_step(
    network,
    optimizer,
    batch,
    device,
    value_weight: float = 1.0,
) -> dict[str, float]:
    """One SGD step on a multipv-soft-targets batch.

    Batch is (states, mpv_indices, mpv_logprobs, played_moves, zs) — the
    five arrays from MultipvDataset.

    Policy loss is the cross-entropy of the softmax over network logits
    against Stockfish's sparse distribution. The played-move column is
    kept ONLY for the top-1-accuracy diagnostic.
    """
    states_np, mpv_idx_np, mpv_logp_np, played_np, zs_np = batch

    to_t = lambda x: torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x.to(device)
    states = to_t(states_np).float()
    mpv_indices = to_t(mpv_idx_np).long()
    mpv_logprobs = to_t(mpv_logp_np).float()
    played = to_t(played_np).long()
    zs = to_t(zs_np).float()

    network.train()
    logits, value_pred = network(states)

    # Build the dense target distribution from the sparse (indices, probs) pair.
    target_dist = _scatter_sparse_target_to_dense(
        mpv_indices, mpv_logprobs, n_actions=N_POLICY, device=device,
    )

    # Soft-target cross-entropy: -sum_i p_target_i * log_softmax(logits)_i
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(target_dist * log_probs).sum(dim=-1).mean()

    value_loss = F.mse_loss(value_pred, zs)

    loss = policy_loss + value_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Diagnostic: top-1 accuracy vs Stockfish's actually-played move.
    with torch.no_grad():
        top1 = (logits.argmax(dim=-1) == played).float().mean().item()
        # Also useful: top-K hit rate (was Stockfish's played move in the
        # network's top-K predictions?). This degrades more gracefully as
        # the policy becomes well-shaped.
        topk_pred = logits.topk(k=mpv_indices.shape[1], dim=-1).indices  # (B, K)
        topk_hit = (topk_pred == played.unsqueeze(-1)).any(dim=-1).float().mean().item()
        # Target-distribution entropy: how peaked is Stockfish about each position?
        # Lower entropy = more confident teacher; higher = more decision diversity.
        eps = 1e-12
        tgt_entropy = -(target_dist * (target_dist + eps).log()).sum(dim=-1).mean().item()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "top1_acc": float(top1),
        "topk_acc": float(topk_hit),
        "tgt_entropy": float(tgt_entropy),
    }
