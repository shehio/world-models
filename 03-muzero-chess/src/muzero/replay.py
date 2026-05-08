"""Replay buffer of trajectories. Samples K-step training windows.

sample(batch_size, K) returns:
    obs_root   : (B, 19, 8, 8)        observation at start_idx
    actions    : (B, K, n_move_planes, 8, 8)  K real actions taken
    pi_targets : (B, K+1, 4672)       MCTS pis at start_idx..start_idx+K
    v_targets  : (B, K+1)             n-step bootstrapped value targets
    r_targets  : (B, K)               actual rewards from the env

Edge cases: trajectories shorter than K from start_idx are zero-padded
and a mask is returned alongside.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity_games: int):
        self.games: deque = deque(maxlen=capacity_games)

    def add_game(self, trajectory) -> None:
        self.games.append(trajectory)

    def sample(self, batch_size: int, unroll_steps: int):
        raise NotImplementedError
