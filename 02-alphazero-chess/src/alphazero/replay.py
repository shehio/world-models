"""Fixed-capacity replay buffer of (state, pi, z) samples.

Sampling: uniform random with replacement; AlphaZero used windowed
recency, but uniform is fine at this scale.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add_game(self, samples: list[tuple]) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int):
        """Returns (states, pis, zs) as torch tensors on device."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.buffer)
