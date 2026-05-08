"""Run with: uv run python tests/test_board.py

No pytest dep — plain assertions.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Make src/ importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import numpy as np

from alphazero.board import (
    N_INPUT_PLANES,
    N_POLICY,
    decode_move,
    encode_board,
    encode_move,
    legal_move_mask,
)


def _random_positions(n: int, max_plies: int = 80, seed: int = 0):
    """Yield n positions reached from random play. Includes some endgames."""
    rng = random.Random(seed)
    for _ in range(n):
        board = chess.Board()
        plies = rng.randint(0, max_plies)
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        yield board


def test_encode_board_shape():
    board = chess.Board()
    p = encode_board(board)
    assert p.shape == (N_INPUT_PLANES, 8, 8), p.shape
    assert p.dtype == np.float32
    # Side-to-move plane is all 1 at start (white).
    assert (p[12] == 1.0).all()
    # Castling rights all 4 at start.
    for plane in (13, 14, 15, 16):
        assert (p[plane] == 1.0).all()


def test_round_trip_legal_moves():
    """For every legal move on many positions, decode(encode(m)) == m."""
    n_positions, total_moves, fails = 0, 0, []
    for board in _random_positions(100, max_plies=80, seed=42):
        n_positions += 1
        for m in board.legal_moves:
            total_moves += 1
            idx = encode_move(m, board)
            assert 0 <= idx < N_POLICY, f"idx {idx} out of range"
            decoded = decode_move(idx, board)
            if decoded != m:
                fails.append((board.fen(), m.uci(), decoded.uci()))
    if fails:
        for fen, want, got in fails[:5]:
            print(f"  FEN: {fen}\n   want: {want}  got: {got}")
        raise AssertionError(f"{len(fails)}/{total_moves} round-trip failures")
    print(f"  ✓ {total_moves} legal moves over {n_positions} positions round-trip cleanly")


def test_legal_move_mask_matches():
    for board in _random_positions(50, seed=7):
        mask = legal_move_mask(board)
        legal_idxs = {encode_move(m, board) for m in board.legal_moves}
        assert mask.sum() == len(legal_idxs), (mask.sum(), len(legal_idxs))
        assert set(np.flatnonzero(mask).tolist()) == legal_idxs


def test_underpromotion_round_trip():
    """Construct a position with multiple promotion options and verify N/B/R round-trip."""
    # White pawn on a7, ready to promote on a8 or capture b8.
    board = chess.Board("1n6/P7/8/8/8/8/8/4K2k w - - 0 1")
    # Push promotion to knight, bishop, rook.
    for promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        m = chess.Move(chess.A7, chess.A8, promotion=promo)
        assert m in board.legal_moves
        idx = encode_move(m, board)
        decoded = decode_move(idx, board)
        assert decoded == m, (m, decoded)
    # Capture promotion a7xb8.
    for promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        m = chess.Move(chess.A7, chess.B8, promotion=promo)
        assert m in board.legal_moves
        idx = encode_move(m, board)
        decoded = decode_move(idx, board)
        assert decoded == m, (m, decoded)


def test_castling_round_trip():
    """Castling moves are encoded by king's from/to squares; round-trip check."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    for move in board.legal_moves:
        if board.is_castling(move):
            idx = encode_move(move, board)
            decoded = decode_move(idx, board)
            assert decoded == move


def main():
    tests = [
        test_encode_board_shape,
        test_round_trip_legal_moves,
        test_legal_move_mask_matches,
        test_underpromotion_round_trip,
        test_castling_round_trip,
    ]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
