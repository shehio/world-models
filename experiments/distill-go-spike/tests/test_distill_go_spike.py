"""Unit tests for the KataGo spike.

Doesn't require a real KataGo binary — uses a mocked analysis response
to exercise the data-conversion path. The tests verify:
  - Board representation produces the right plane shape and content.
  - GTP coordinate parsing handles the 'skip I' quirk + pass + corners.
  - KataGo response → soft targets produces a valid log-probability
    distribution that softmax-sums to 1.0.
  - Top-K truncation pads correctly when KataGo returns fewer candidates.

Run from the spike root:
    uv run --project . pytest tests/ -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go_spike.board import (
    BLACK,
    BOARD_SIZE,
    GoBoard,
    N_INPUT_PLANES,
    WHITE,
    board_to_planes,
    gtp_move_to_xy,
    xy_to_gtp_move,
)
from distill_go_spike.katago_data import (
    DEFAULT_MULTIPV,
    _gtp_to_flat_idx,
    katago_analysis_to_soft_targets,
)


class TestGoBoard:
    def test_empty_board(self):
        b = GoBoard(size=9)
        assert b.size == 9
        assert b.to_move == BLACK
        assert b.grid.shape == (9, 9)
        assert (b.grid == 0).all()

    def test_play_flips_turn(self):
        b = GoBoard(size=9)
        b.play(3, 3)
        assert b.to_move == WHITE
        assert b.grid[3, 3] == BLACK
        b.play(4, 4)
        assert b.to_move == BLACK
        assert b.grid[4, 4] == WHITE

    def test_pass_flips_turn_without_placing(self):
        b = GoBoard(size=9)
        b.play(-1, -1)
        assert b.to_move == WHITE
        assert (b.grid == 0).all()

    def test_out_of_bounds_raises(self):
        b = GoBoard(size=9)
        with pytest.raises(ValueError):
            b.play(9, 0)
        with pytest.raises(ValueError):
            b.play(0, -2)

    def test_copy_isolates_state(self):
        b = GoBoard(size=9)
        b.play(3, 3)
        c = b.copy()
        c.play(0, 0)
        assert b.grid[0, 0] == 0


class TestPlanes:
    def test_plane_shape_and_dtype(self):
        b = GoBoard(size=19)
        planes = board_to_planes(b)
        assert planes.shape == (N_INPUT_PLANES, 19, 19)
        assert planes.dtype == np.float32

    def test_side_to_move_plane(self):
        b = GoBoard(size=9)
        planes = board_to_planes(b)
        assert (planes[2] == 1.0).all(), "Black to move → plane 2 = ones"
        b.play(0, 0)
        planes = board_to_planes(b)
        assert (planes[2] == 0.0).all(), "White to move → plane 2 = zeros"

    def test_stones_land_on_correct_plane(self):
        b = GoBoard(size=9)
        b.play(3, 3)  # black
        b.play(4, 4)  # white
        planes = board_to_planes(b)
        assert planes[0, 3, 3] == 1.0 and planes[1, 3, 3] == 0.0
        assert planes[1, 4, 4] == 1.0 and planes[0, 4, 4] == 0.0

    def test_ones_plane(self):
        b = GoBoard(size=9)
        planes = board_to_planes(b)
        assert (planes[3] == 1.0).all()


class TestGTPCoordinates:
    def test_basic_parse(self):
        # A1 → bottom-left (column 0, row 0 from bottom = y = size - 1)
        x, y = gtp_move_to_xy("A1", 9)
        assert (x, y) == (0, 8)

    def test_pass(self):
        assert gtp_move_to_xy("pass", 9) == (-1, -1)
        assert gtp_move_to_xy("PASS", 19) == (-1, -1)

    def test_skip_i_column(self):
        # J is the 9th column (skipping I). On a 19x19, J = column 8.
        x, y = gtp_move_to_xy("J10", 19)
        assert x == 8
        assert y == 19 - 10

    def test_reserved_i_raises(self):
        with pytest.raises(ValueError):
            gtp_move_to_xy("I10", 19)

    def test_xy_to_gtp_inverts(self):
        for raw in ["A1", "D4", "J10", "T19"]:
            x, y = gtp_move_to_xy(raw, 19)
            assert xy_to_gtp_move(x, y, 19) == raw

    def test_xy_to_gtp_pass(self):
        assert xy_to_gtp_move(-1, -1, 19) == "pass"


class TestFlatIdx:
    def test_pass_is_size_squared(self):
        assert _gtp_to_flat_idx("pass", 9) == 81
        assert _gtp_to_flat_idx("PASS", 19) == 361

    def test_a1_is_corner(self):
        # A1 = (x=0, y=size-1) = flat (size-1) * size + 0
        assert _gtp_to_flat_idx("A1", 9) == 8 * 9 + 0

    def test_d4_consistency(self):
        x, y = gtp_move_to_xy("D4", 9)
        assert _gtp_to_flat_idx("D4", 9) == y * 9 + x


class TestSoftTargets:
    """The core integration test: KataGo response → policy distribution."""

    def _mock_response(self, moves: list[tuple[str, int]], winrate: float = 0.55):
        """Build a minimal KataGo analysis-engine response."""
        return {
            "id": "test",
            "rootInfo": {"winrate": winrate, "scoreLead": 1.5, "visits": sum(v for _, v in moves)},
            "moveInfos": [
                {"move": gtp, "visits": visits, "winrate": winrate, "scoreLead": 0.0}
                for gtp, visits in moves
            ],
        }

    def test_top_k_padded_when_fewer_candidates(self):
        # Only 3 candidate moves, but we ask for top-8.
        resp = self._mock_response([("D4", 100), ("Q16", 60), ("Q4", 30)])
        idx, lp, val = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8)
        assert idx.shape == (8,)
        assert lp.shape == (8,)
        assert (idx[3:] == -1).all(), "Slots beyond candidates must be -1"
        assert np.isinf(lp[3:]).all(), "Padded slots must be -inf"

    def test_softmax_sums_to_one(self):
        resp = self._mock_response([("D4", 100), ("Q16", 60), ("Q4", 30), ("D16", 10)])
        idx, lp, _ = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8)
        mask = idx != -1
        probs = np.exp(lp[mask])
        assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5)

    def test_higher_visits_means_higher_probability(self):
        resp = self._mock_response([("D4", 1000), ("Q16", 100), ("Q4", 10)])
        _, lp, _ = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8)
        # lp is in candidate order (sorted desc by visits) — first 3 valid.
        assert lp[0] > lp[1] > lp[2]

    def test_temperature_sharpens_distribution(self):
        # Small visit counts so neither temperature saturates float32 softmax.
        # With visits 5/4/3: T=10 → near-uniform, T=1 → still mild peak.
        resp = self._mock_response([("D4", 5), ("Q16", 4), ("Q4", 3)])
        _, lp_hot, _ = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8, temperature=10.0)
        _, lp_cold, _ = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8, temperature=1.0)
        # Lower temperature → sharper distribution → higher top prob.
        assert np.exp(lp_cold[0]) > np.exp(lp_hot[0])

    def test_pass_in_top_k(self):
        resp = self._mock_response([("D4", 100), ("pass", 50)])
        idx, _, _ = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8)
        assert 9 * 9 in idx.tolist(), "Pass index (size**2) should appear"

    def test_value_passthrough(self):
        resp = self._mock_response([("D4", 100)], winrate=0.73)
        _, _, val = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8)
        assert math.isclose(val, 0.73)

    def test_empty_moveinfos_safe_fallback(self):
        resp = {"id": "x", "rootInfo": {"winrate": 0.5}, "moveInfos": []}
        idx, lp, val = katago_analysis_to_soft_targets(resp, board_size=9, top_k=8)
        assert idx.shape == (8,) and lp.shape == (8,)
        assert idx[0] == 81  # pass
        assert math.isclose(float(np.exp(lp[0])), 1.0)


class TestSchemaMatchesChessPipeline:
    """The whole point of the spike: produce data that fits chess training."""

    def test_npz_keys_match_chess_format(self):
        """The result dict must have keys: states, moves, multipv_indices, multipv_logprobs, zs."""
        from distill_go_spike.katago_data import play_one_game  # noqa: F401
        # We can't actually run play_one_game without KataGo, but we can
        # verify the contract by inspecting the source. (Sanity test only.)
        # If someone changes the dict keys, the chess-style merge_chunks.py
        # will fail downstream — fail loudly here at unit-test time.
        import inspect
        from distill_go_spike import katago_data
        src = inspect.getsource(katago_data.play_one_game)
        for key in ('"states"', '"moves"', '"multipv_indices"',
                    '"multipv_logprobs"', '"zs"'):
            assert key in src, f"play_one_game() must return key {key}"
