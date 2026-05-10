"""Board encoding: chess.Board <-> input planes; chess.Move <-> 4672-index.

Conventions
-----------
Board orientation: always white's POV (no flipping). Side-to-move plane
tells the network whose turn it is.

Two encoders live here:

* `encode_board(b)` — 19 planes, single position. Used by v1-v4 runs.
* `encode_history(boards)` — 103 planes, last 8 positions stacked plus
  current-position metadata. Used by v5+ (paper-faithful).

Input planes (19 total) — `encode_board`:
  0..5    : white pieces (P, N, B, R, Q, K) as 0/1
  6..11   : black pieces (P, N, B, R, Q, K) as 0/1
  12      : side-to-move (all 1 if white to move, else all 0)
  13      : white kingside castle right
  14      : white queenside castle right
  15      : black kingside castle right
  16      : black queenside castle right
  17      : en-passant target (one-hot)
  18      : halfmove clock / 100  (no-progress count, normalized)

Input planes (103 total) — `encode_history`:
  0..11   : t-7 piece planes (12 = 6 white + 6 black; all zero if game shorter)
  12..23  : t-6 piece planes
   ... (8 stacks of 12 piece planes = 96 planes; oldest first, current last)
  84..95  : t   piece planes (current position)
  96      : side-to-move (current)
  97..100 : castling rights (current; W-K, W-Q, B-K, B-Q)
  101     : en-passant target (current; one-hot)
  102     : halfmove clock / 100 (current)

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

# History encoder constants (paper-faithful)
N_HISTORY = 8
N_PIECE_PLANES_PER_BOARD = 12  # 6 piece types × 2 colors
N_META_PLANES = 7  # side-to-move + 4 castling + ep + halfmove
N_INPUT_PLANES_HISTORY = N_HISTORY * N_PIECE_PLANES_PER_BOARD + N_META_PLANES  # 103

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


def _fill_piece_planes(out: np.ndarray, board: chess.Board, offset: int) -> None:
    """Write 12 piece planes for `board` into out[offset:offset+12]."""
    for sq, piece in board.piece_map().items():
        r, f = chess.square_rank(sq), chess.square_file(sq)
        pt_idx = _PIECE_TYPES.index(piece.piece_type)
        plane = pt_idx + (0 if piece.color == chess.WHITE else 6) + offset
        out[plane, r, f] = 1.0


def encode_history(boards: list[chess.Board], n_history: int = N_HISTORY) -> np.ndarray:
    """Stack last `n_history` positions into a (103, 8, 8) input tensor.

    `boards` is ordered oldest-first, most-recent-last. The last element is
    the current position (the one whose move we're choosing). Earlier
    elements are the prior plies.

    If `len(boards) < n_history`, the leading slots are zero-padded —
    early in the game we have fewer prior positions, which is fine.

    Meta planes (side-to-move, castling, EP, halfmove clock) are taken
    from the *current* board only — we don't carry stale meta from prior
    plies.
    """
    if not boards:
        raise ValueError("encode_history needs at least one board")

    n_planes = n_history * N_PIECE_PLANES_PER_BOARD + N_META_PLANES
    planes = np.zeros((n_planes, 8, 8), dtype=np.float32)

    # Take the last n_history boards. If we have fewer, leave the leading
    # piece-plane stacks all zero (oldest-first padding).
    history = boards[-n_history:]
    pad = n_history - len(history)
    for i, b in enumerate(history):
        offset = (pad + i) * N_PIECE_PLANES_PER_BOARD
        _fill_piece_planes(planes, b, offset)

    # Meta planes from the current (most recent) board only.
    meta_offset = n_history * N_PIECE_PLANES_PER_BOARD  # 96
    cur = boards[-1]
    if cur.turn == chess.WHITE:
        planes[meta_offset, :, :] = 1.0
    if cur.has_kingside_castling_rights(chess.WHITE):
        planes[meta_offset + 1, :, :] = 1.0
    if cur.has_queenside_castling_rights(chess.WHITE):
        planes[meta_offset + 2, :, :] = 1.0
    if cur.has_kingside_castling_rights(chess.BLACK):
        planes[meta_offset + 3, :, :] = 1.0
    if cur.has_queenside_castling_rights(chess.BLACK):
        planes[meta_offset + 4, :, :] = 1.0
    if cur.ep_square is not None:
        r, f = chess.square_rank(cur.ep_square), chess.square_file(cur.ep_square)
        planes[meta_offset + 5, r, f] = 1.0
    planes[meta_offset + 6, :, :] = min(cur.halfmove_clock, 100) / 100.0
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
