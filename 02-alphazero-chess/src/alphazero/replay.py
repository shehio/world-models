"""Fixed-capacity replay buffer of (state, pi, z) samples."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
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
