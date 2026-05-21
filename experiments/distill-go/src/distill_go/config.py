"""Hyperparameters for the Go pipeline.

Single source of truth — passed to network, MCTS, train loop. Defaults
target the 9x9 demo; switch board_size + adjust filters/blocks for 19x19.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoConfig:
    # Board / move encoding
    board_size: int = 9
    n_input_planes: int = 4   # 4 (no history) or 17 (8-step history × 2 + STM)
    komi: float = 7.5
    rules: str = "tromp-taylor"

    # Network
    n_res_blocks: int = 4     # 9x9 demo size; 19x19 paper-sized ~ 20
    n_filters: int = 64

    # MCTS
    sims_train: int = 100
    sims_eval: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.03   # AlphaGo-Zero uses 0.03 for 19x19; ~0.15 for 9x9
    dirichlet_eps: float = 0.25

    # Self-play / eval
    temp_moves: int = 30
    max_plies: int = 400           # 9x9 games rarely exceed 100; pad for safety

    # Training
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4

    @property
    def policy_size(self) -> int:
        """board_size² + 1 (pass)."""
        return self.board_size * self.board_size + 1

    @property
    def pass_move(self) -> int:
        return self.board_size * self.board_size
