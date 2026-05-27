"""Replay buffer with K-step trajectory sampling for MuZero training.

Each GameRecord stores one complete self-play game. Sampling picks a
random (game, start_ply) and produces the batch dict that train_step
expects:

    observation : (B, 19, 8, 8)   # only obs at start ply — unroll is via g_θ
    actions     : (B, K)
    pi_targets  : (B, K+1, 4672)  # MCTS π at each step (zeros past terminal → masked)
    z_targets   : (B, K+1)        # value target at each step
    r_targets   : (B, K)          # reward observed AFTER each action

Value target: Monte Carlo return, sign-flipped by mover. In chess with
γ=1 and reward-only-at-terminal, this is exactly the n-step bootstrap
for n large, so there's no benefit to bootstrapping off MCTS root values.

Absorbing past terminal: positions past the end of the game get z=0,
r=0, pi=zeros (CE loss masks zero rows), and random actions (the
dynamics net sees an in-distribution input but the loss signals say
"nothing more to predict").
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .config import MuZeroConfig


@dataclass
class GameRecord:
    """One self-play game's full trajectory.

    Conventions:
      observations[t] : (19, 8, 8) float32 board encoding BEFORE action a_t
      actions[t]      : int in [0, 4672) — the action played from ply t
      pis[t]          : (4672,) float32 — MCTS visit distribution at ply t
      z_per_ply[t]    : float — value target at ply t (outcome from t's mover view)
      r_per_action[t] : float — reward observed AFTER action a_t
                         (zero except possibly at the final, game-ending action)
    """
    observations: list[np.ndarray]
    actions: list[int]
    pis: list[np.ndarray]
    z_per_ply: list[float]
    r_per_action: list[float]

    @property
    def length(self) -> int:
        return len(self.actions)

    @classmethod
    def from_trajectory(
        cls,
        observations: list[np.ndarray],
        actions: list[int],
        pis: list[np.ndarray],
        outcome_for_starter: float,
        terminal_reward_for_mover: float,
    ) -> "GameRecord":
        """Build a GameRecord from raw self-play arrays.

        outcome_for_starter:        outcome from the perspective of the player
                                    to move at ply 0 (+1 win, 0 draw, -1 loss).
        terminal_reward_for_mover:  reward at the game-ending action from the
                                    perspective of the player who PLAYED that
                                    action. +1 if it delivered mate, 0 if draw
                                    (game ended without a winner).
        """
        T = len(actions)
        assert len(observations) == T, "need one obs per played ply"
        assert len(pis) == T, "need one pi per played ply"

        # z_t = outcome from mover-at-t's perspective. Mover at ply 0 is the
        # "starter"; alternates each ply. For chess, γ=1, reward-at-terminal:
        # z_t collapses to ±outcome with sign flipped by parity.
        z = [outcome_for_starter * (1.0 if t % 2 == 0 else -1.0) for t in range(T)]
        # Reward zero everywhere except the final action that ended the game.
        r = [0.0] * T
        if T > 0:
            r[-1] = terminal_reward_for_mover
        return cls(
            observations=observations,
            actions=actions,
            pis=pis,
            z_per_ply=z,
            r_per_action=r,
        )


class ReplayBuffer:
    """Bounded FIFO of GameRecords with K-step batch sampling."""

    def __init__(self, capacity: int, cfg: MuZeroConfig):
        self.capacity = capacity
        self.cfg = cfg
        self._games: list[GameRecord] = []

    def __len__(self) -> int:
        return len(self._games)

    @property
    def n_positions(self) -> int:
        return sum(g.length for g in self._games)

    def push(self, game: GameRecord) -> None:
        assert game.length > 0, "empty games shouldn't be pushed"
        self._games.append(game)
        if len(self._games) > self.capacity:
            self._games.pop(0)

    def sample(self, batch_size: int, K: int, rng: random.Random | None = None) -> dict:
        """Sample a (batch_size, K)-style batch ready for train_step."""
        assert len(self._games) > 0, "buffer is empty"
        r = rng if rng is not None else random
        A = self.cfg.action_dim

        obs_batch = np.empty((batch_size, 19, 8, 8), dtype=np.float32)
        actions = np.empty((batch_size, K), dtype=np.int64)
        pi_targets = np.zeros((batch_size, K + 1, A), dtype=np.float32)
        z_targets = np.zeros((batch_size, K + 1), dtype=np.float32)
        r_targets = np.zeros((batch_size, K), dtype=np.float32)

        for b in range(batch_size):
            game = r.choice(self._games)
            t = r.randint(0, game.length - 1)
            obs_batch[b] = game.observations[t]
            for k in range(K + 1):
                pos = t + k
                if pos < game.length:
                    pi_targets[b, k] = game.pis[pos]
                    z_targets[b, k] = game.z_per_ply[pos]
                # else: zero — absorbing state, pi-row=0 will mask CE loss.
            for k in range(K):
                pos = t + k
                if pos < game.length:
                    actions[b, k] = game.actions[pos]
                    r_targets[b, k] = game.r_per_action[pos]
                else:
                    # Random action past terminal — value doesn't matter for
                    # supervision (r=0, z=0), but g_θ still needs *some* input.
                    actions[b, k] = r.randint(0, A - 1)

        return {
            "observation": torch.from_numpy(obs_batch),
            "actions": torch.from_numpy(actions),
            "pi_targets": torch.from_numpy(pi_targets),
            "z_targets": torch.from_numpy(z_targets),
            "r_targets": torch.from_numpy(r_targets),
        }
