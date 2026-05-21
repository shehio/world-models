"""Full Go distillation pipeline.

Mirrors the chess pipeline:
  wm_chess.board   -> distill_go.board     (rules engine + plane encoding)
  wm_chess.network -> distill_go.network   (parameterized for Go shape)
  wm_chess.mcts    -> distill_go.mcts      (PUCT, same algorithm)
  stockfish_data   -> distill_go.katago_data (KataGo teacher subprocess)

The on-disk .npz schema is byte-compatible with the chess pipeline so
`wm_chess/scripts/merge_chunks.py` reads Go chunks unchanged.
"""
from .board import (
    BLACK,
    EMPTY,
    WHITE,
    GoBoard,
    board_to_planes,
    board_to_history_planes,
)
from .config import GoConfig

__all__ = [
    "BLACK",
    "WHITE",
    "EMPTY",
    "GoBoard",
    "GoConfig",
    "board_to_planes",
    "board_to_history_planes",
]
