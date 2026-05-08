"""Board encoding: chess.Board <-> input planes; chess.Move <-> 4672-index.

Input planes (19 total, simplified from AlphaZero's full spec):
  0..5    : my pieces       (P, N, B, R, Q, K) as 0/1
  6..11   : opponent pieces (P, N, B, R, Q, K) as 0/1
  12      : side-to-move (all 1 if white to move, else all 0)
  13..16  : castling rights (mine kingside, mine queenside, opp kingside, opp queenside)
  17      : en-passant target square (one-hot if exists)
  18      : halfmove clock / 100 (no-progress count, normalized)

Move encoding (8 * 8 * 73 = 4672):
  56 "queen-like" planes  : 8 directions × 7 distances (sliding/king/pawn-push)
  8  knight planes        : 8 knight jumps
  9  underpromotion planes: 3 pieces (N/B/R) × 3 directions (forward, capture-left, capture-right)

When black is to move, the board is flipped vertically before encoding,
so the network always sees "my pieces moving up the board." This is the
canonical AlphaZero/Leela trick — without it, the network has to learn
two separate sets of pawn/castling rules.

NOT YET IMPLEMENTED. Sketch only.
"""
from __future__ import annotations

import numpy as np
import chess


N_INPUT_PLANES = 19
N_POLICY = 8 * 8 * 73


def encode_board(board: chess.Board) -> np.ndarray:
    """chess.Board -> (19, 8, 8) float32 array, oriented for side-to-move."""
    raise NotImplementedError


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """chess.Move -> int in [0, 4672). Orientation matches encode_board."""
    raise NotImplementedError


def decode_move(idx: int, board: chess.Board) -> chess.Move:
    """Inverse of encode_move. Returns a legal move or raises if idx is illegal."""
    raise NotImplementedError


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """(4672,) bool mask of legal moves at this position. Used to mask policy logits."""
    raise NotImplementedError
