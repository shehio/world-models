"""Board encoding — identical to 02-alphazero-chess/src/alphazero/board.py.

Copied (not imported) so each project is self-contained. If the encoders
diverge significantly in the future we'll factor a shared `chess_common`
package; for now duplication is cheaper than premature abstraction.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

import numpy as np
import chess


N_INPUT_PLANES = 19
N_POLICY = 8 * 8 * 73


def encode_board(board: chess.Board) -> np.ndarray:
    raise NotImplementedError


def encode_move(move: chess.Move, board: chess.Board) -> int:
    raise NotImplementedError


def decode_move(idx: int, board: chess.Board) -> chess.Move:
    raise NotImplementedError


def legal_move_mask(board: chess.Board) -> np.ndarray:
    raise NotImplementedError
