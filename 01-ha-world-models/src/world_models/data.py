"""Datasets used during training.

`FrameDataset` flattens every rollout into a giant pile of (preprocessed,
CHW) frames for the VAE.

`LatentSequenceDataset` keeps episode boundaries — each example is a
contiguous (z, a) window of length `seq_len` so the LSTM can do truncated
BPTT through the dynamics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """Loads every rollout's frames into a single (N, C, H, W) float32 tensor.

    For our budget (~10k frames at 64x64x3 float32) this is well under
    1GB and fits in RAM, which is much faster than streaming from disk.
    """

    def __init__(self, rollout_dir: Path) -> None:
        files = sorted(Path(rollout_dir).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no rollouts found in {rollout_dir}")
        chunks = []
        for f in files:
            data = np.load(f)
            frames = data["frames"]  # (T, H, W, C) float32 in [0,1]
            chunks.append(np.transpose(frames, (0, 3, 1, 2)))
        self.frames = torch.from_numpy(np.concatenate(chunks, axis=0)).float()

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.frames[idx]


class LatentSequenceDataset(Dataset):
    """Yields (z[t:t+T], a[t:t+T], z[t+1:t+T+1]) windows from encoded rollouts."""

    def __init__(self, latent_dir: Path, seq_len: int) -> None:
        files = sorted(Path(latent_dir).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no latents found in {latent_dir}")
        self.seq_len = seq_len
        self.episodes: list[tuple[np.ndarray, np.ndarray]] = []
        for f in files:
            data = np.load(f)
            z, a = data["z"], data["a"]
            # Need at least seq_len + 1 steps to form a (z_t, a_t, z_{t+1}) window.
            if z.shape[0] >= seq_len + 1:
                self.episodes.append((z, a))

        self._index: list[tuple[int, int]] = []
        for ep_i, (z, _) in enumerate(self.episodes):
            for start in range(0, z.shape[0] - seq_len - 1):
                self._index.append((ep_i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_i, start = self._index[idx]
        z, a = self.episodes[ep_i]
        z_in = torch.from_numpy(z[start : start + self.seq_len]).float()
        a_in = torch.from_numpy(a[start : start + self.seq_len]).float()
        z_target = torch.from_numpy(z[start + 1 : start + 1 + self.seq_len]).float()
        return z_in, a_in, z_target
