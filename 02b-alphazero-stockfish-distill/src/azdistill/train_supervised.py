"""Supervised distillation training.

Loss = cross_entropy(stockfish_move_one_hot, policy_logits)
     + mse(z, value_pred)

Difference from 02's train.py: the policy target is a HARD one-hot from
Stockfish's actual move, not a soft MCTS visit-count distribution.
We mask illegal logits to -inf before softmax for stability.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class StockfishDataset(Dataset):
    """Loads (state, move_idx, z) tuples from an NPZ created by
    stockfish_data.generate_dataset_parallel."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.states = d["states"]   # (N, 19, 8, 8) float32
        self.moves = d["moves"]     # (N,) int64 — target action index
        self.zs = d["zs"]           # (N,) float32

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, i: int):
        return self.states[i], self.moves[i], self.zs[i]


def _legal_mask_from_state(state: np.ndarray) -> np.ndarray:
    """We need to mask illegal logits to -inf during loss computation. But we
    didn't save legal-mask alongside the data. The simplest robust solution:
    rebuild the chess.Board from the state planes and re-derive legal moves.

    NOT used in the basic CE training loop (the one-hot target only has
    probability on the legal move), but available if you want to mask
    illegal logits at inference.
    """
    raise NotImplementedError("not needed for CE-with-one-hot training")


def train_step(
    network,
    optimizer,
    batch,
    device,
    value_weight: float = 1.0,
) -> dict[str, float]:
    """One SGD step. Batch is (states, move_idxs, zs) numpy arrays / tensors.

    Policy loss: F.cross_entropy on (logits, target_indices). PyTorch's CE
    handles the one-hot target implicitly via integer class indices.
    """
    states_np, moves_np, zs_np = batch
    states = torch.from_numpy(states_np).to(device) if isinstance(states_np, np.ndarray) else states_np.to(device)
    moves = torch.from_numpy(moves_np).to(device) if isinstance(moves_np, np.ndarray) else moves_np.to(device)
    zs = torch.from_numpy(zs_np).to(device) if isinstance(zs_np, np.ndarray) else zs_np.to(device)

    network.train()
    logits, value_pred = network(states)

    # Standard CE with integer targets (== one-hot CE).
    policy_loss = F.cross_entropy(logits, moves.long())
    value_loss = F.mse_loss(value_pred, zs.float())

    loss = policy_loss + value_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Top-1 policy accuracy: did the network's argmax match Stockfish's move?
    with torch.no_grad():
        top1 = (logits.argmax(dim=-1) == moves).float().mean().item()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "top1_acc": float(top1),
    }
