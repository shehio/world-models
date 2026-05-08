"""Replay buffer of (state, pi, z) samples.

Two implementations:
  ReplayBuffer        — flat deque, uniform-sample. Simple. Used by the v1/v2 single-process loop.
  ShardedReplayBuffer — sliding window of per-iteration shards, drop the oldest
                        whole iteration when capacity exceeded. Closer to what
                        KataGo / suragnair/alpha-zero-general do, and avoids
                        the "iter 0's near-random play stays in the buffer
                        forever" failure mode.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Flat deque, uniform sampling. Capacity in #positions."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add_game(self, samples) -> None:
        for s, p, z in samples:
            self.buffer.append((s, p, z))

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idxs]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        pis = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        zs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        return states, pis, zs

    def __len__(self) -> int:
        return len(self.buffer)


class ShardedReplayBuffer:
    """Sliding-window-by-iteration buffer.

    Internally a deque of shards, where each shard is one iteration's worth of
    samples. When `max_shards` is exceeded, the oldest entire iteration is
    discarded. Sampling pools across all currently-resident shards.

    Why this matters:
        The flat-deque ReplayBuffer keeps individual samples around forever
        until enough new samples push them out. Early-iter "near-random play"
        samples can persist for a long time and pull the network back toward
        bad behavior. With per-iteration shards, ALL of iter K's data drops
        as a unit, on a predictable schedule. This is the standard AZ-clone
        recipe (suragnair: numItersForTrainExamplesHistory = 20).
    """

    def __init__(self, max_shards: int = 20):
        self.shards: deque = deque(maxlen=max_shards)
        # Cache the flat sample list when sampling, rebuild on add.
        self._flat_cache: list | None = None

    def add_iteration(self, samples_iter) -> None:
        """Add ALL samples from one iteration as a single shard."""
        shard = list(samples_iter)
        self.shards.append(shard)
        self._flat_cache = None  # invalidate

    def add_game(self, samples) -> None:
        """Backward-compatible single-game add — folds into the *current* shard.

        Useful in single-process loops that don't think in iter boundaries.
        Each call appends a fresh shard with one game's samples; if you want
        per-iter granularity, use add_iteration.
        """
        self.shards.append(list(samples))
        self._flat_cache = None

    def _flat(self):
        if self._flat_cache is None:
            self._flat_cache = [s for shard in self.shards for s in shard]
        return self._flat_cache

    def sample(self, batch_size: int, device: torch.device):
        flat = self._flat()
        idxs = np.random.randint(0, len(flat), size=batch_size)
        batch = [flat[i] for i in idxs]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        pis = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        zs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        return states, pis, zs

    def __len__(self) -> int:
        return sum(len(s) for s in self.shards)

    @property
    def n_shards(self) -> int:
        return len(self.shards)
