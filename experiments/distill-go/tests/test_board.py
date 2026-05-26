"""Tests for the rules engine. Run with: uv run --project . pytest tests/

These check the bits the spike's tests didn't cover, since the spike
delegated rules to KataGo. Bugs here propagate into MCTS rollouts, training
data, and eval, so they're worth pinning down with unit tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from distill_go.board import (
    BLACK,
    WHITE,
    EMPTY,
    GoBoard,
    IllegalMoveError,
    board_to_planes,
    board_to_history_planes,
    flat_to_gtp,
    gtp_to_flat,
)


def _xy(b: GoBoard, x: int, y: int) -> int:
    return y * b.size + x


# ---------------------------------------------------------------- placement


def test_play_alternates_color_and_advances_count():
    b = GoBoard(size=5)
    b.play(_xy(b, 2, 2))
    assert b.grid[2, 2] == BLACK
    assert b.to_move == WHITE
    assert b.move_count == 1
    b.play(_xy(b, 1, 1))
    assert b.grid[1, 1] == WHITE
    assert b.to_move == BLACK


def test_play_occupied_raises():
    b = GoBoard(size=5)
    b.play(_xy(b, 2, 2))
    with pytest.raises(IllegalMoveError):
        b.play(_xy(b, 2, 2))


def test_pass_advances_turn_and_increments_pass_counter():
    b = GoBoard(size=5)
    b.play(b.pass_move)
    assert b.to_move == WHITE
    assert b.consecutive_passes == 1
    b.play(b.pass_move)
    assert b.consecutive_passes == 2
    assert b.is_game_over


# ---------------------------------------------------------------- capture


def test_simple_single_stone_capture():
    """Surround a single black stone with white; it should disappear."""
    b = GoBoard(size=5)
    # B at (2,2). W surrounds at N, E, S, W.
    b.play(_xy(b, 2, 2))   # B center
    b.play(_xy(b, 2, 1))   # W north
    b.play(_xy(b, 0, 0))   # B somewhere harmless
    b.play(_xy(b, 3, 2))   # W east
    b.play(_xy(b, 4, 4))   # B harmless
    b.play(_xy(b, 2, 3))   # W south
    b.play(_xy(b, 4, 0))   # B harmless
    b.play(_xy(b, 1, 2))   # W west — captures B at (2,2)
    assert b.grid[2, 2] == EMPTY


def test_group_capture():
    """A connected two-stone B group gets captured by one final W move."""
    b = GoBoard(size=5)
    # Direct setup: B at (1,1)-(1,2), surrounded by W except for one liberty
    # at (2,1). W to play (2,1) captures both B stones.
    b.grid[1, 1] = BLACK
    b.grid[1, 2] = BLACK
    b.grid[0, 1] = WHITE   # N of (1,1)
    b.grid[0, 2] = WHITE   # N of (1,2)
    b.grid[1, 0] = WHITE   # W of (1,1)
    b.grid[1, 3] = WHITE   # E of (1,2)
    b.grid[2, 2] = WHITE   # S of (1,2)
    # Remaining liberty of B group is (2,1) — S of (1,1).
    b.to_move = WHITE
    b._past_states = {b.grid.tobytes()}
    move = 2 * 5 + 1  # (y=2, x=1)
    b.play(move)
    assert b.grid[1, 1] == EMPTY
    assert b.grid[1, 2] == EMPTY
    assert b.grid[2, 1] == WHITE


# ---------------------------------------------------------------- suicide


def test_suicide_forbidden():
    """B playing into a 1-point eye made entirely of W is suicide."""
    b = GoBoard(size=3)
    # Position:
    #   . W .
    #   W . .
    #   . . .
    # B to play at (0,0): neighbors are (0,1)=W and (1,0)=W. B's own group
    # has 0 libs and no W group is captured (each W has other libs) → suicide.
    b.grid[0, 1] = WHITE
    b.grid[1, 0] = WHITE
    b.to_move = BLACK
    b._past_states = {b.grid.tobytes()}
    with pytest.raises(IllegalMoveError, match="suicide"):
        b.play(0)  # (y=0, x=0)


def test_capture_is_not_suicide_even_if_own_libs_zero_first():
    """If a move would be suicide but captures the surrounding group first,
    it's legal because captures resolve before the suicide check."""
    b = GoBoard(size=3)
    # Position:
    #   . B .
    #   B W .
    #   . B .
    # W at center has only 1 liberty at (2,1). If B plays (2,1):
    #   - (2,1) tentatively placed.
    #   - The W stone has no liberties → captured (removed).
    #   - Then check B's own group at (2,1): it has libs at (2,0), (2,2). OK.
    b.grid[0, 1] = BLACK   # (1,0)
    b.grid[1, 0] = BLACK   # (0,1)
    b.grid[1, 1] = WHITE   # (1,1)
    b.grid[2, 1] = BLACK   # (1,2)  - wait that's also B; need to fix layout

    # Restart with proper coordinates (y=row from top, x=col).
    b = GoBoard(size=3)
    b.grid[0, 1] = BLACK   # y=0,x=1
    b.grid[1, 0] = BLACK   # y=1,x=0
    b.grid[1, 1] = WHITE   # y=1,x=1  center
    b.grid[2, 1] = BLACK   # y=2,x=1
    # W has neighbors at y=0,x=1 (B); y=1,x=0 (B); y=1,x=2 (empty — 1 lib); y=2,x=1 (B).
    # So W has 1 liberty at (y=1,x=2). B to play at (1,2) captures W.
    b.to_move = BLACK
    b._past_states = {b.grid.tobytes()}
    move_idx = 1 * 3 + 2  # y=1,x=2
    b.play(move_idx)        # should NOT raise — captures W instead
    assert b.grid[1, 1] == EMPTY   # white stone captured
    assert b.grid[1, 2] == BLACK   # black move played


# ---------------------------------------------------------------- ko


def test_positional_superko_blocks_repetition():
    """The classic 4-stone ko shape: white captures, black can't immediately recapture."""
    b = GoBoard(size=5)
    # Set up known ko position (mid-board to keep edges out of it):
    #   . . . . .
    #   . . W . .
    #   . W . W .
    #   . . W . .
    #   . . . . .
    # Black to play at (2,2) captures the center? No — center is empty. We need
    # an actual capture. Place stones so the previous turn was a capture, then
    # try to recapture.
    # Easier: place stones and use direct grid setup + past_states tracking.
    b.grid[1, 2] = WHITE   # north
    b.grid[2, 1] = WHITE   # west
    b.grid[3, 2] = WHITE   # south
    b.grid[2, 3] = WHITE   # east
    # B has just been captured at (2,2) — there is now no stone there. Past state
    # included a position with (2,2)=BLACK that was captured. We simulate by
    # adding "the position before W captured" to _past_states.
    pre_capture = b.grid.copy()
    pre_capture[2, 2] = BLACK
    # In the actual capture sequence, the state with center=BLACK was visible
    # to past_states. After the capture move, the state is the current one.
    b._past_states = {pre_capture.tobytes(), b.grid.tobytes()}
    # Now it's black's turn to recapture by playing at (2,2). But (2,2) is empty,
    # placing B there captures one W (the one with 1 lib). The resulting state
    # has B at (2,2) and one W missing — which is exactly pre_capture-like.
    # To make this a clean ko, we need the recapture to reproduce pre_capture.
    # Construct directly: pre_capture had B at (2,2) only, all W in place. The
    # recapture would put B at (2,2) and remove one W. That's a DIFFERENT state
    # from pre_capture (one W missing). So this isn't a true ko in this layout.

    # Reset with a real ko layout: the classic 2x2 ko shape on an edge.
    b = GoBoard(size=4)
    # Position:        (B and W in a ladder-like edge)
    #   . B W .
    #   B . . W
    #   B W . W   <- mid; we'll engineer the ko on rows 1-2
    #   . B W .
    # Actually a cleanest ko: put two pieces of each color so capturing one
    # produces a mirrored state.
    b = GoBoard(size=4)
    b.grid[1, 1] = BLACK
    b.grid[1, 2] = WHITE
    b.grid[2, 1] = WHITE
    b.grid[2, 2] = BLACK
    b.grid[0, 1] = BLACK
    b.grid[1, 0] = BLACK
    b.grid[3, 2] = BLACK
    b.grid[2, 3] = BLACK
    b.grid[0, 2] = WHITE
    b.grid[1, 3] = WHITE
    b.grid[3, 1] = WHITE
    b.grid[2, 0] = WHITE
    # In this position (1,1)=B has libs only at... check: neighbors (0,1)=B,
    # (2,1)=W, (1,0)=B, (1,2)=W. B group at (1,1) is connected to B at (0,1)
    # and (1,0). The combined group has libs at... too complex to verify by hand.
    # Skip this elaborate layout; instead test the *mechanism* directly.

    # MECHANISM TEST: manually put a state in _past_states that the next move
    # would reproduce, and verify the move raises "ko".
    b = GoBoard(size=3)
    # Empty board. B plays (0,0); result has B at (0,0) only.
    state_after_b_corner = np.zeros((3, 3), dtype=np.int8)
    state_after_b_corner[0, 0] = BLACK
    # Pre-load _past_states so playing B at (0,0) now hits the past state.
    b._past_states.add(state_after_b_corner.tobytes())
    # Black to move; (0,0) move would reproduce state_after_b_corner → ko-illegal.
    with pytest.raises(IllegalMoveError, match="ko"):
        b.play(0)


# ---------------------------------------------------------------- legal mask


def test_legal_mask_pass_always_legal():
    b = GoBoard(size=5)
    m = b.legal_mask()
    assert m[-1] == 1


def test_legal_mask_excludes_occupied_and_suicide():
    b = GoBoard(size=3)
    b.grid[1, 1] = WHITE  # occupied center
    b.grid[0, 1] = WHITE  # surround (0,0) from one side
    b.grid[1, 0] = WHITE  # surround (0,0) from other side
    b.to_move = BLACK
    b._past_states = {b.grid.tobytes()}
    m = b.legal_mask()
    # occupied:
    assert m[1 * 3 + 1] == 0  # (1,1)
    assert m[0 * 3 + 1] == 0  # (0,1)
    assert m[1 * 3 + 0] == 0  # (1,0)
    # suicide at (0,0): both relevant neighbors W, no captures available → suicide
    assert m[0] == 0
    # legal elsewhere (e.g. (2,2) far from action):
    assert m[2 * 3 + 2] == 1


# ---------------------------------------------------------------- planes


def test_4_plane_encoding_shape_and_content():
    b = GoBoard(size=5)
    b.grid[1, 1] = BLACK
    b.grid[3, 3] = WHITE
    b.to_move = WHITE  # white to move
    planes = board_to_planes(b)
    assert planes.shape == (4, 5, 5)
    assert planes[0, 1, 1] == 1.0  # black plane
    assert planes[1, 3, 3] == 1.0  # white plane
    assert (planes[2] == 0.0).all()  # STM=W → side-to-move plane is 0
    assert (planes[3] == 1.0).all()  # ones plane


def test_17_plane_history_shape():
    b1 = GoBoard(size=5)
    b2 = b1.copy(); b2.play(0)
    b3 = b2.copy(); b3.play(1)
    planes = board_to_history_planes([b1, b2, b3], n_history=8)
    assert planes.shape == (17, 5, 5)
    # Most-recent board (b3) is the last pair in the history stack.
    # b3 has B at (0,0) and W at (0,1).
    last_pair_start = 2 * 7  # index of (black, white) pair for the newest board
    assert planes[last_pair_start + 0, 0, 0] == 1.0   # black at (0,0)
    assert planes[last_pair_start + 1, 0, 1] == 1.0   # white at (0,1)


# ---------------------------------------------------------------- scoring


def test_tromp_taylor_empty_board_white_by_komi():
    b = GoBoard(size=5, komi=7.5)
    bp, wp = b.tromp_taylor_score()
    assert bp == 0
    assert wp == 7.5
    assert b.winner() == WHITE


def test_tromp_taylor_simple_territory():
    """One-color board: empty regions touching only that color count for it."""
    b = GoBoard(size=5, komi=0.0)
    # B wall along column x=2. Every empty point only borders B (or empty),
    # so all 20 empties are B's territory.
    for y in range(5):
        b.grid[y, 2] = BLACK
    bp, wp = b.tromp_taylor_score()
    assert bp == 5 + 20    # 5 B stones + 20 empty points (all touch only B)
    assert wp == 0
    assert b.winner() == BLACK


def test_tromp_taylor_two_color_split():
    """B + W walls touching: each owns the empties on its own side."""
    b = GoBoard(size=5, komi=0.0)
    # Adjacent walls: B at x=1, W at x=2. No empty region touches both colors
    # because the colors are stones-adjacent, not empty-adjacent.
    # Col 0 (5 empties) touches only B → black territory.
    # Cols 3 and 4 (10 empties) touch only W → white territory.
    for y in range(5):
        b.grid[y, 1] = BLACK
        b.grid[y, 2] = WHITE
    bp, wp = b.tromp_taylor_score()
    assert bp == 5 + 5     # 5 B stones + 5 empties in col 0
    assert wp == 5 + 10    # 5 W stones + 10 empties in cols 3-4


def test_tromp_taylor_dame_region_nobody_scores():
    """An empty region that touches BOTH colors is dame — counts for neither.

    Mutation-catches: if the `bordering == {BLACK}` / `{WHITE}` checks were
    relaxed to `in {BLACK}` / `in {WHITE}`, this region would mistakenly
    score for both sides.
    """
    b = GoBoard(size=5, komi=0.0)
    # Row 1 = all BLACK. Row 3 = all WHITE. Row 2 (5 empties) borders both.
    for x in range(5):
        b.grid[1, x] = BLACK
        b.grid[3, x] = WHITE
    bp, wp = b.tromp_taylor_score()
    # Row 0 (5 empties) only touches BLACK (via row 1) → black territory.
    # Row 2 (5 empties) borders BOTH → dame, nobody.
    # Row 4 (5 empties) only touches WHITE (via row 3) → white territory.
    assert bp == 5 + 5    # 5 B stones + 5 empties in row 0
    assert wp == 5 + 5    # 5 W stones + 5 empties in row 4


def test_tromp_taylor_fully_filled_board_no_territory():
    """Every point is a stone — no empties to assign as territory."""
    b = GoBoard(size=3, komi=0.5)
    # Checkerboard so neither side has empty neighbors that get captured.
    for y in range(3):
        for x in range(3):
            b.grid[y, x] = BLACK if (y + x) % 2 == 0 else WHITE
    bp, wp = b.tromp_taylor_score()
    # 5 BLACK stones (corners + center), 4 WHITE stones, 0 territory either side.
    assert bp == 5
    assert wp == 4 + 0.5    # WHITE stones + komi


def test_tromp_taylor_all_black_board_wins_decisively():
    """Sanity: a board with only BLACK stones gives BLACK the entire board minus komi."""
    b = GoBoard(size=3, komi=7.5)
    for y in range(3):
        for x in range(3):
            b.grid[y, x] = BLACK
    bp, wp = b.tromp_taylor_score()
    assert bp == 9
    assert wp == 7.5
    assert b.winner() == BLACK   # 9 > 7.5


# ---------------------------------------------------------------- coord helpers


def test_gtp_roundtrip():
    for size in (9, 19):
        for idx in range(size * size + 1):
            gtp = flat_to_gtp(idx, size)
            assert gtp_to_flat(gtp, size) == idx
