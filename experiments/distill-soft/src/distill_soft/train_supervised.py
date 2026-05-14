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

import json
import os
import zipfile

import numpy as np
import numpy.lib.format as npfmt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from wm_chess.board import N_POLICY


def _stream_copy(src, dst, n_bytes: int | None = None) -> None:
    """Copy from src to dst in 8MB chunks. Stops after n_bytes (or EOF)."""
    remaining = n_bytes if n_bytes is not None else float("inf")
    while remaining > 0:
        chunk = 8 * 1024 * 1024
        if remaining < chunk:
            chunk = int(remaining)
        buf = src.read(chunk)
        if not buf:
            break
        dst.write(buf)
        remaining -= len(buf)


def _extract_npz_to_memmap_dir(
    npz_path: str, extract_dir: str, max_rows: int | None = None,
) -> None:
    """Stream-extract a .npz (which is a zip of .npy files) to a directory
    of plain .npy files. Uses zipfile, not numpy, so peak memory is the
    zip-decompress buffer (a few MB), NOT the full uncompressed array.

    Necessary for our merged datasets — at 30M positions per dataset the
    uncompressed `states` array is ~145 GB, which doesn't fit in any
    sensible GPU instance's RAM. With this extraction + mmap_mode='r' in
    the loader, only the batch-sized slice the trainer touches is paged
    in. The full file stays on disk; the OS page cache handles
    sequential access efficiently.

    If `max_rows` is given, each row-shaped array is truncated to the
    first `max_rows` entries on disk — pairs with `--max-positions` so
    we don't waste 130 GB of disk extracting positions we'll throw away.
    Scalar arrays (K.npy, temperature_pawns.npy) are copied verbatim.

    Idempotent: skips re-extraction if all expected .npy files already
    exist.
    """
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(npz_path) as zf:
        for entry in zf.infolist():
            target = os.path.join(extract_dir, entry.filename)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                continue
            with zf.open(entry) as src, open(target, "wb") as dst:
                if max_rows is None:
                    _stream_copy(src, dst)
                    continue
                # Peek at the .npy header to learn shape + dtype, so we
                # know whether to truncate and how many bytes to copy.
                # Versions: 1.0 uses uint16 header length, 2.0 uses uint32.
                version = npfmt.read_magic(src)
                if version == (1, 0):
                    shape, fortran_order, dtype = npfmt.read_array_header_1_0(src)
                    write_header = npfmt.write_array_header_1_0
                elif version == (2, 0):
                    shape, fortran_order, dtype = npfmt.read_array_header_2_0(src)
                    write_header = npfmt.write_array_header_2_0
                else:
                    raise ValueError(f"unsupported .npy version: {version}")
                header_d = {
                    "descr": npfmt.dtype_to_descr(dtype),
                    "fortran_order": fortran_order,
                    "shape": shape,
                }
                if not shape or shape[0] <= max_rows:
                    # Scalar / already-small — write header then rest verbatim.
                    write_header(dst, header_d)
                    _stream_copy(src, dst)
                    continue
                # Truncate: rewrite header with smaller leading dim and
                # copy only max_rows * row_bytes.
                header_d["shape"] = (max_rows,) + tuple(shape[1:])
                write_header(dst, header_d)
                row_bytes = int(
                    np.prod(shape[1:]) if len(shape) > 1 else 1
                ) * dtype.itemsize
                _stream_copy(src, dst, max_rows * row_bytes)


class MultipvDataset(Dataset):
    """Loads (state, multipv_indices, multipv_logprobs, played_move, z)
    from an NPZ created by 02c's stockfish_data.generate_dataset_parallel.

    Returns numpy arrays; the train loop tensorizes them.

    For large datasets that don't fit in RAM, the loader transparently
    extracts the .npz to a sibling `_extracted/` directory and memory-maps
    each .npy file. Set `mmap=False` to force the old in-RAM path
    (useful for tiny test fixtures or if extract_dir disk is constrained).
    """

    def __init__(self, npz_path: str, mmap: bool = True,
                 extract_dir: str | None = None,
                 max_rows: int | None = None):
        if mmap:
            extract_dir = extract_dir or (npz_path + "_extracted")
            _extract_npz_to_memmap_dir(npz_path, extract_dir,
                                       max_rows=max_rows)
            self.states = np.load(os.path.join(extract_dir, "states.npy"),
                                  mmap_mode="r")
            self.moves = np.load(os.path.join(extract_dir, "moves.npy"),
                                 mmap_mode="r")
            self.multipv_indices = np.load(
                os.path.join(extract_dir, "multipv_indices.npy"),
                mmap_mode="r")
            self.multipv_logprobs = np.load(
                os.path.join(extract_dir, "multipv_logprobs.npy"),
                mmap_mode="r")
            self.zs = np.load(os.path.join(extract_dir, "zs.npy"),
                              mmap_mode="r")
            k_path = os.path.join(extract_dir, "K.npy")
            self.K = (int(np.load(k_path)) if os.path.exists(k_path)
                      else int(self.multipv_indices.shape[1]))
        else:
            d = np.load(npz_path)
            self.states = d["states"]
            self.moves = d["moves"]              # played move (top-1) per pos
            self.multipv_indices = d["multipv_indices"]    # (N, K) int64
            self.multipv_logprobs = d["multipv_logprobs"]  # (N, K) float32
            self.zs = d["zs"]
            self.K = (int(d["K"]) if "K" in d.files
                      else self.multipv_indices.shape[1])

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
    hard_targets: bool = False,
) -> dict[str, float]:
    """One SGD step on a multipv-soft-targets batch.

    Batch is (states, mpv_indices, mpv_logprobs, played_moves, zs) — the
    five arrays from MultipvDataset.

    Policy loss:
      - hard_targets=False (default): soft-CE against Stockfish's sparse
        distribution (top-K with softmax-over-cp logprobs).
      - hard_targets=True: standard CE against Stockfish's actually-played
        move (one-hot). Same loss as 02b. Lets us A/B the target shape
        on identical data + identical architecture.
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

    if hard_targets:
        # Same loss as 02b: standard CE against the actually-played move.
        policy_loss = F.cross_entropy(logits, played)
        # For the diagnostic block below we still want a "target_dist" of
        # sorts; build a delta on the played move so target-entropy reads 0.
        target_dist = F.one_hot(played, num_classes=N_POLICY).float()
    else:
        # Soft-target cross-entropy against Stockfish's multipv distribution:
        # -sum_i p_target_i * log_softmax(logits)_i
        target_dist = _scatter_sparse_target_to_dense(
            mpv_indices, mpv_logprobs, n_actions=N_POLICY, device=device,
        )
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
