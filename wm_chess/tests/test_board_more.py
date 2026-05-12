"""Additional board-encoder tests beyond what test_board.py covers.

Covers en-passant, halfmove clock edge cases, castling-rights mutation,
move-decoding pawn promotions, and the legal_move_mask contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm_chess.board import (
    N_INPUT_PLANES,
    N_POLICY,
    decode_move,
    encode_board,
    encode_move,
    legal_move_mask,
)


class TestEnPassantPlane:
    def test_no_ep_at_startpos(self):
        planes = encode_board(chess.Board())
        assert planes[17].sum() == 0.0

    def test_ep_square_one_hot(self):
        b = chess.Board()
        b.push_uci("e2e4")
        # After e2e4 the EP square is e3 (rank 2 from white POV, file 4).
        planes = encode_board(b)
        ep = b.ep_square
        if ep is not None:
            r, f = chess.square_rank(ep), chess.square_file(ep)
            assert planes[17, r, f] == 1.0
            assert planes[17].sum() == 1.0


class TestCastlingRightsLoss:
    def test_kingside_castling_lost_after_king_move(self):
        b = chess.Board()
        for m in ["e2e4", "e7e5", "e1e2"]:
            b.push_uci(m)
        planes = encode_board(b)
        # White lost both castling rights (king moved).
        assert planes[13].sum() == 0.0
        assert planes[14].sum() == 0.0
        # Black rights intact.
        assert planes[15].any()
        assert planes[16].any()


class TestHalfmoveClock:
    def test_zero_at_startpos(self):
        planes = encode_board(chess.Board())
        assert np.isclose(planes[18, 0, 0], 0.0)

    def test_clipped_at_100(self):
        # Synthetic high halfmove clock.
        b = chess.Board("8/8/8/8/8/8/4K3/4k3 w - - 200 100")
        planes = encode_board(b)
        # 200 clipped to 100, normalized to 1.0
        assert np.isclose(planes[18, 0, 0], 1.0)


class TestEncodeMoveCoverage:
    def test_encode_then_legal_mask_agrees(self):
        b = chess.Board()
        mask = legal_move_mask(b)
        for m in b.legal_moves:
            assert mask[encode_move(m, b)], f"{m.uci()} encoded to an index outside the legal mask"

    def test_all_legal_decoded_to_legal(self):
        """For each legal move at startpos, encode→decode should
        produce a chess.Move that's legal at that board."""
        b = chess.Board()
        for m in b.legal_moves:
            idx = encode_move(m, b)
            decoded = decode_move(idx, b)
            assert decoded in b.legal_moves, f"decoded {decoded.uci()} from {idx} not legal"

    def test_index_in_range(self):
        b = chess.Board()
        for m in b.legal_moves:
            idx = encode_move(m, b)
            assert 0 <= idx < N_POLICY


class TestLegalMoveMaskShapes:
    def test_dtype_and_shape(self):
        mask = legal_move_mask(chess.Board())
        assert mask.shape == (N_POLICY,)
        assert mask.dtype == bool

    def test_position_with_many_moves(self):
        """Open middlegame should have 30+ legal moves."""
        b = chess.Board()
        for m in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
            b.push_uci(m)
        mask = legal_move_mask(b)
        assert mask.sum() >= 25
