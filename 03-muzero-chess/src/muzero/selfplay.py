"""Self-play one game. Returns a Trajectory.

A Trajectory bundles the full game's data needed to construct training
targets later:
  observations  : list of (19, 8, 8) — only used at training time for the root
  actions       : list of int (4672-encoded)
  rewards       : list of float (chess: 0 except final; +1/-1/0)
  mcts_pis      : list of (4672,) — visit-count distributions at each ply
  mcts_values   : list of float — root value at each ply (used for n-step bootstrap)
  outcome       : final z from white POV

Replay sampler later picks a random (trajectory, start_index, K) and
emits the corresponding K-step training window.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    mcts_pis: list = field(default_factory=list)
    mcts_values: list = field(default_factory=list)
    outcome: float = 0.0


def play_game(model, cfg) -> Trajectory:
    raise NotImplementedError
