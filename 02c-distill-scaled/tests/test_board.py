"""Tests for the board encoder / move encoder.

These are pure-Python, no engine, no GPU — fast and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from azdistill_scaled.board import (
    N_INPUT_PLANES,
    N_POLICY,
    decode_move,
    encode_board,
    encode_move,
    legal_move_mask,
)


class TestEncodeBoard:
    def test_shape_and_dtype(self):
        planes = encode_board(chess.Board())
        assert planes.shape == (N_INPUT_PLANES, 8, 8)
        assert planes.dtype == np.float32

    def test_starting_position_pawns(self):
        planes = encode_board(chess.Board())
        # White pawns on rank 1 (index 1), black pawns on rank 6.
        assert (planes[0, 1, :] == 1.0).all()
        assert (planes[6, 6, :] == 1.0).all()

    def test_starting_position_kings(self):
        planes = encode_board(chess.Board())
        # White king is plane 5 (P,N,B,R,Q,K) at e1 = rank 0, file 4.
        assert planes[5, 0, 4] == 1.0
        # Black king at e8 = rank 7, file 4.
        assert planes[11, 7, 4] == 1.0

    def test_side_to_move_plane(self):
        b = chess.Board()
        assert encode_board(b)[12].all()           # white to move
        b.push_san("e4")
        assert not encode_board(b)[12].any()       # black to move

    def test_castling_rights(self):
        b = chess.Board()
        planes = encode_board(b)
        # all four castling planes should be set at the start.
        assert planes[13].all() and planes[14].all()
        assert planes[15].all() and planes[16].all()

    def test_halfmove_clock_normalized(self):
        # Position with halfmove clock 50 → plane 18 = 0.5
        b = chess.Board("8/8/8/8/8/8/4K3/4k3 w - - 50 100")
        planes = encode_board(b)
        assert np.isclose(planes[18, 0, 0], 0.5)


class TestEncodeMoveRoundtrip:
    @pytest.mark.parametrize("uci", [
        "e2e4",   # double pawn push
        "g1f3",   # knight
        "d1h5",   # queen diagonal
        "e1g1",   # white kingside castle (handled as king e1→g1, distance 2)
    ])
    def test_startpos_moves_decode_legal(self, uci):
        b = chess.Board()
        # Build up to a position where the move is legal.
        if uci == "e1g1":
            for m in ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5"]:
                b.push_uci(m)
        m = chess.Move.from_uci(uci)
        idx = encode_move(m, b)
        assert 0 <= idx < N_POLICY
        decoded = decode_move(idx, b)
        # Decoded move should match (modulo promotion suffix, which startpos
        # moves don't have).
        assert decoded.from_square == m.from_square
        assert decoded.to_square == m.to_square

    def test_underpromotion_knight(self):
        # White pawn on a7, black king on h1, white king on h3, white to move.
        # a7-a8=N is an underpromotion.
        b = chess.Board("8/P7/8/8/8/7K/8/7k w - - 0 1")
        m = chess.Move.from_uci("a7a8n")
        idx = encode_move(m, b)
        decoded = decode_move(idx, b)
        assert decoded.promotion == chess.KNIGHT
        assert decoded.from_square == m.from_square
        assert decoded.to_square == m.to_square

    def test_queen_promotion_via_queenlike_plane(self):
        b = chess.Board("8/P7/8/8/8/7K/8/7k w - - 0 1")
        m = chess.Move.from_uci("a7a8q")
        idx = encode_move(m, b)
        # Should be in queen-like planes (0..55), not underpromotion (64+)
        plane = idx // 64
        assert plane < 56
        decoded = decode_move(idx, b)
        assert decoded.promotion == chess.QUEEN

    def test_all_legal_moves_roundtrip(self):
        """Stronger: at the start position, every legal move encodes
        uniquely and decodes back to itself."""
        b = chess.Board()
        seen_indices = set()
        for m in b.legal_moves:
            idx = encode_move(m, b)
            assert idx not in seen_indices, f"collision at {m.uci()}"
            seen_indices.add(idx)
            decoded = decode_move(idx, b)
            assert decoded.from_square == m.from_square
            assert decoded.to_square == m.to_square


class TestLegalMoveMask:
    def test_startpos(self):
        b = chess.Board()
        mask = legal_move_mask(b)
        assert mask.shape == (N_POLICY,)
        assert mask.dtype == bool
        # 20 legal moves at startpos.
        assert mask.sum() == 20

    def test_checkmate_position_has_no_legal_moves(self):
        # Fool's mate
        b = chess.Board()
        for m in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            b.push_uci(m)
        assert b.is_checkmate()
        assert legal_move_mask(b).sum() == 0
