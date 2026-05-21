"""Dataset + train step for Go distillation.

Same soft-CE-on-multipv loss as chess (`distill_soft.train_supervised`),
adapted for Go's flat policy space (S² + 1 actions, no plane-flatten).

The .npz schema is identical to chess. For 9x9 demo runs the dataset fits
in RAM; for 19x19 production we'd port the streaming-extract-to-memmap
loader from chess. Stub left for that future.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
import torch.nn.functional as F


class GoMultipvDataset:
    """Loads (state, mpv_indices, mpv_logprobs, played, z) from an NPZ.

    Designed for 9x9 demo scale (10K-100K positions fits easily in RAM).
    For larger 19x19 runs, swap in the streaming-extract-to-memmap loader
    from `distill_soft.train_supervised._extract_npz_to_memmap_dir`.
    """

    def __init__(self, npz_path: str, max_rows: int | None = None) -> None:
        d = np.load(npz_path)
        self.states = d["states"]
        self.moves = d["moves"]
        self.multipv_indices = d["multipv_indices"]
        self.multipv_logprobs = d["multipv_logprobs"]
        self.zs = d["zs"]
        self.K = (int(d["K"]) if "K" in d.files
                  else int(self.multipv_indices.shape[1]))
        if max_rows is not None and self.states.shape[0] > max_rows:
            self.states = self.states[:max_rows]
            self.moves = self.moves[:max_rows]
            self.multipv_indices = self.multipv_indices[:max_rows]
            self.multipv_logprobs = self.multipv_logprobs[:max_rows]
            self.zs = self.zs[:max_rows]

    def __len__(self) -> int:
        return int(self.states.shape[0])


def _scatter_sparse_target_to_dense(
    mpv_indices: torch.Tensor,
    mpv_logprobs: torch.Tensor,
    n_actions: int,
    device: torch.device,
) -> torch.Tensor:
    """(B,K) sparse → (B, n_actions) dense. Padding (-1, -inf) contributes 0."""
    probs_k = mpv_logprobs.exp()
    valid = mpv_indices >= 0
    safe_indices = mpv_indices.clamp(min=0)
    safe_probs = probs_k * valid.float()
    dense = torch.zeros((mpv_indices.shape[0], n_actions),
                        dtype=probs_k.dtype, device=device)
    dense.scatter_add_(1, safe_indices, safe_probs)
    return dense


def train_step(
    network,
    optimizer,
    batch,
    device: torch.device,
    n_actions: int,
    value_weight: float = 1.0,
    hard_targets: bool = False,
    use_amp: bool = False,
) -> dict[str, float]:
    """One SGD step on a multipv batch. Same loss shape as chess pipeline.

    batch: (states, mpv_indices, mpv_logprobs, played, zs) — all np.ndarray.
    n_actions: policy size = board_size² + 1.
    """
    states_np, mpv_idx_np, mpv_logp_np, played_np, zs_np = batch
    to_t = (
        lambda x: torch.from_numpy(x).to(device)
        if isinstance(x, np.ndarray) else x.to(device)
    )
    states = to_t(states_np).float()
    mpv_indices = to_t(mpv_idx_np).long()
    mpv_logprobs = to_t(mpv_logp_np).float()
    played = to_t(played_np).long()
    zs = to_t(zs_np).float()

    network.train()
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp else contextlib.nullcontext()
    )
    with autocast:
        logits, value_pred = network(states)
        if hard_targets:
            policy_loss = F.cross_entropy(logits, played)
            target_dist = F.one_hot(played, num_classes=n_actions).float()
        else:
            target_dist = _scatter_sparse_target_to_dense(
                mpv_indices, mpv_logprobs, n_actions=n_actions, device=device,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(target_dist * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(value_pred, zs)
        loss = policy_loss + value_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        top1 = (logits.argmax(dim=-1) == played).float().mean().item()
        K = mpv_indices.shape[1]
        topk_pred = logits.topk(k=K, dim=-1).indices
        topk_hit = (topk_pred == played.unsqueeze(-1)).any(dim=-1).float().mean().item()
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
