"""wm_chess — shared chess core for the world-models experiments.

This package owns the canonical implementations of:

  - board encoding (19-plane and 103-plane history variants)
  - the AlphaZero-style ResNet (policy + value heads)
  - MCTS (single-threaded and batched-with-virtual-loss variants)
  - arena (vs-engine match player used by all evals)
  - Config dataclass holding shared hyperparameters

Three sibling experiments depend on it:

  experiments/selfplay/        — AlphaZero self-play RL
  experiments/distill-hard/    — supervised distillation, hard targets
  experiments/distill-soft/    — supervised distillation, multipv soft targets

Each experiment owns its own data-gen, training loop, and scripts;
shared chess machinery lives here.
"""
from .board import (
    N_HISTORY,
    N_INPUT_PLANES,
    N_INPUT_PLANES_HISTORY,
    N_MOVE_PLANES,
    N_POLICY,
    decode_move,
    encode_board,
    encode_history,
    encode_move,
    legal_move_mask,
)
from .config import Config
from .network import AlphaZeroNet, ResidualBlock, get_device

__all__ = [
    "N_HISTORY",
    "N_INPUT_PLANES",
    "N_INPUT_PLANES_HISTORY",
    "N_MOVE_PLANES",
    "N_POLICY",
    "decode_move",
    "encode_board",
    "encode_history",
    "encode_move",
    "legal_move_mask",
    "Config",
    "AlphaZeroNet",
    "ResidualBlock",
    "get_device",
]
