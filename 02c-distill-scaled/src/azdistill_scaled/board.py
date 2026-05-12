"""Board encoding: chess.Board <-> input planes; chess.Move <-> 4672-index.

Conventions
-----------
Board orientation: always white's POV (no flipping). Side-to-move plane
tells the network whose turn it is.

Input planes (19 total):
  0..5    : white pieces (P, N, B, R, Q, K) as 0/1
  6..11   : black pieces (P, N, B, R, Q, K) as 0/1
  12      : side-to-move (all 1 if white to move, else all 0)
  13      : white kingside castle right
  14      : white queenside castle right
  15      : black kingside castle right
  16      : black queenside castle right
  17      : en-passant target (one-hot)
  18      : halfmove clock / 100  (no-progress count, normalized)

Move encoding (8 * 8 * 73 = 4672):
  planes 0..55  : queen-like = 8 directions x 7 distances
  planes 56..63 : 8 knight jumps
  planes 64..72 : underpromotion = 3 pieces (N/B/R) x 3 directions

  Direction order (queen-like and knight): see _QUEEN_DIRS / _KNIGHT_DIRS.
  Underpromotion direction: 0=capture-toward-a-file, 1=push, 2=capture-toward-h-file.

  Action index = plane * 64 + from_square.

Queen-promotion is encoded as a queen-like move (forward 1 square on the
last rank) — NOT in the underpromotion planes. Decoding handles this.
"""
from __future__ import annotations

import numpy as np
import chess

N_INPUT_PLANES = 19
N_MOVE_PLANES = 73
N_POLICY = 8 * 8 * N_MOVE_PLANES  # 4672

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# 8 directions, in order: N, NE, E, SE, S, SW, W, NW. (dr, df)
_QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
# 8 knight jumps, fixed order. (dr, df)
_KNIGHT_DIRS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
# Underpromotion piece order, indexed 0..2.
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def encode_board(board: chess.Board) -> np.ndarray:
    """chess.Board -> (19, 8, 8) float32 array, white's POV always."""
    planes = np.zeros((N_INPUT_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        r, f = chess.square_rank(sq), chess.square_file(sq)
        pt_idx = _PIECE_TYPES.index(piece.piece_type)
        plane = pt_idx + (0 if piece.color == chess.WHITE else 6)
        planes[plane, r, f] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0
    if board.ep_square is not None:
        r, f = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        planes[17, r, f] = 1.0
    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    return planes


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """chess.Move -> int in [0, 4672). Requires the board only for pawn-promotion ambiguity."""
    fr, ff = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    tr, tf = chess.square_rank(move.to_square), chess.square_file(move.to_square)
    dr, df = tr - fr, tf - ff

    # Underpromotion: pawn promotion to N/B/R.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # df ∈ {-1, 0, +1} (push or capture); piece N/B/R.
        piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        dir_idx = df + 1  # -1 -> 0, 0 -> 1, +1 -> 2
        plane = 64 + piece_idx * 3 + dir_idx
    elif (abs(dr), abs(df)) in [(1, 2), (2, 1)]:
        # Knight jump.
        plane = 56 + _KNIGHT_DIRS.index((dr, df))
    else:
        # Queen-like (includes queen promotion, which is pawn forward to last rank).
        d = max(abs(dr), abs(df))
        ndr = 0 if dr == 0 else dr // abs(dr)
        ndf = 0 if df == 0 else df // abs(df)
        dir_idx = _QUEEN_DIRS.index((ndr, ndf))
        plane = dir_idx * 7 + (d - 1)

    return plane * 64 + move.from_square


def decode_move(idx: int, board: chess.Board) -> chess.Move:
    """Inverse of encode_move. Auto-fills queen promotion when a pawn reaches the last rank.

    If the resulting move is illegal at `board`, returns it anyway —
    callers must validate via legal_move_mask if they want guarantees.
    """
    plane, from_sq = divmod(idx, 64)
    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)

    if plane < 56:
        dir_idx, dist_minus_one = divmod(plane, 7)
        dr, df = _QUEEN_DIRS[dir_idx]
        distance = dist_minus_one + 1
        tr, tf = fr + dr * distance, ff + df * distance
        promotion = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and tr in (0, 7):
            promotion = chess.QUEEN
        to_sq = chess.square(tf, tr)
        return chess.Move(from_sq, to_sq, promotion=promotion)

    if plane < 64:
        dr, df = _KNIGHT_DIRS[plane - 56]
        tr, tf = fr + dr, ff + df
        return chess.Move(from_sq, chess.square(tf, tr))

    # Underpromotion
    under = plane - 64
    piece_idx, dir_idx = divmod(under, 3)
    df = dir_idx - 1
    piece = board.piece_at(from_sq)
    color = piece.color if piece is not None else board.turn
    dr = 1 if color == chess.WHITE else -1
    tr, tf = fr + dr, ff + df
    return chess.Move(from_sq, chess.square(tf, tr), promotion=_UNDERPROMO_PIECES[piece_idx])


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """(4672,) bool mask of legal moves at this position."""
    mask = np.zeros(N_POLICY, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move, board)] = True
    return mask
