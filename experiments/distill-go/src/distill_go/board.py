"""Minimal Go rules engine + plane encoding.

Differs from the spike's board.py: the spike only tracked stone placement
because KataGo upstream enforced rules. For MCTS rollouts, the student
needs its own legal-move generation + capture + ko + suicide handling.

Implemented:
  - Liberty counting via flood fill
  - Capture: opposing groups with 0 liberties after a move are removed
  - Suicide: forbidden (Tromp-Taylor rules — no suicide)
  - Positional superko: the *resulting board* must not repeat any past position
  - Pass: always legal; two consecutive passes end the game
  - Tromp-Taylor scoring: stones + territory + komi

Move encoding: flat int. 0..size²-1 = board points (y * size + x). size² = pass.

Plane encoding: two variants —
  - 4 planes (default): black stones, white stones, side-to-move, ones.
  - 17 planes (AlphaGo-Zero-style): 8 board states × (black, white) + STM.

Both produce shape (n_input_planes, size, size) float32.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


EMPTY = 0
BLACK = 1
WHITE = 2

DEFAULT_SIZE = 9
DEFAULT_KOMI = 7.5


class IllegalMoveError(ValueError):
    """Raised when play() receives an illegal move (occupied / suicide / ko)."""


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


@dataclass
class GoBoard:
    """Go board with full rules — capture, ko, suicide, pass, scoring.

    Use `play(move_idx)` to advance, `legal_moves()` to query, `is_game_over`
    to check for two consecutive passes, `tromp_taylor_score()` to score.

    `move_idx` is flat: 0..size²-1 = board points (y*size+x, y=0 is top row),
    size² = pass.
    """

    size: int = DEFAULT_SIZE
    komi: float = DEFAULT_KOMI
    grid: np.ndarray = field(init=False)
    to_move: int = BLACK
    consecutive_passes: int = 0
    move_count: int = 0
    # Past board states (as bytes) for positional superko. Includes the
    # CURRENT state once a move is played; ko-illegal moves are ones whose
    # resulting state matches anything in this set.
    _past_states: set = field(default_factory=set, repr=False)
    # Stack of (grid_bytes, to_move, consecutive_passes) snapshots for
    # cheap undo. Used by legal_moves() to test trial plays.
    _history_stack: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self._past_states.add(self.grid.tobytes())

    @property
    def pass_move(self) -> int:
        return self.size * self.size

    # ------------------------------------------------------------------ moves

    def play(self, move_idx: int) -> None:
        """Play a move (legal or it raises). Modifies the board in place."""
        if move_idx == self.pass_move:
            self.to_move = _other(self.to_move)
            self.consecutive_passes += 1
            self.move_count += 1
            self._past_states.add(self.grid.tobytes())
            return
        y, x = divmod(move_idx, self.size)
        if not (0 <= y < self.size and 0 <= x < self.size):
            raise IllegalMoveError(f"out of bounds: {move_idx}")
        if self.grid[y, x] != EMPTY:
            raise IllegalMoveError(f"occupied: ({x},{y})")

        new_grid = self._grid_after(y, x, self.to_move)
        if new_grid is None:
            raise IllegalMoveError(f"suicide: ({x},{y})")
        h = new_grid.tobytes()
        if h in self._past_states:
            raise IllegalMoveError(f"ko (positional superko): ({x},{y})")

        self.grid = new_grid
        self._past_states.add(h)
        self.to_move = _other(self.to_move)
        self.consecutive_passes = 0
        self.move_count += 1

    def _grid_after(self, y: int, x: int, color: int) -> np.ndarray | None:
        """Return the grid after placing color at (y,x), removing captures.

        Returns None if the move is suicide (own group has 0 liberties after
        captures resolve).
        """
        new_grid = self.grid.copy()
        new_grid[y, x] = color
        opp = _other(color)
        for ny, nx in self._neighbors(y, x):
            if new_grid[ny, nx] == opp:
                group = self._connected_group(new_grid, ny, nx)
                if self._group_liberties(new_grid, group) == 0:
                    for gy, gx in group:
                        new_grid[gy, gx] = EMPTY
        own_group = self._connected_group(new_grid, y, x)
        if self._group_liberties(new_grid, own_group) == 0:
            return None
        return new_grid

    def legal_mask(self) -> np.ndarray:
        """Return a (size² + 1,) int8 mask: 1 if move is legal, 0 if not.

        Pass is always legal.
        """
        mask = np.zeros(self.size * self.size + 1, dtype=np.int8)
        mask[-1] = 1  # pass always legal
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y, x] != EMPTY:
                    continue
                new_grid = self._grid_after(y, x, self.to_move)
                if new_grid is None:
                    continue
                if new_grid.tobytes() in self._past_states:
                    continue
                mask[y * self.size + x] = 1
        return mask

    # ------------------------------------------------------------------ groups

    def _neighbors(self, y: int, x: int):
        if y > 0:
            yield (y - 1, x)
        if y < self.size - 1:
            yield (y + 1, x)
        if x > 0:
            yield (y, x - 1)
        if x < self.size - 1:
            yield (y, x + 1)

    def _connected_group(self, grid: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
        """BFS from (y,x) over same-color stones. Returns list of coords."""
        color = grid[y, x]
        if color == EMPTY:
            return []
        seen = {(y, x)}
        out = [(y, x)]
        q = deque([(y, x)])
        while q:
            cy, cx = q.popleft()
            for ny, nx in self._neighbors(cy, cx):
                if (ny, nx) in seen:
                    continue
                if grid[ny, nx] == color:
                    seen.add((ny, nx))
                    out.append((ny, nx))
                    q.append((ny, nx))
        return out

    def _group_liberties(self, grid: np.ndarray, group: list[tuple[int, int]]) -> int:
        """Count empty neighbors of all stones in `group` (de-duplicated)."""
        libs = set()
        for y, x in group:
            for ny, nx in self._neighbors(y, x):
                if grid[ny, nx] == EMPTY:
                    libs.add((ny, nx))
        return len(libs)

    # ------------------------------------------------------------------ scoring

    @property
    def is_game_over(self) -> bool:
        return self.consecutive_passes >= 2

    def tromp_taylor_score(self) -> tuple[float, float]:
        """Return (black_score, white_score) under Tromp-Taylor rules.

        - Each stone of color X = 1 point for X.
        - Each empty point whose flood-fill reaches only X stones = 1 point for X.
        - Komi is added to white.

        Ties are possible only with integer komi; default 7.5 forbids them.
        """
        black_pts = 0.0
        white_pts = 0.0
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y, x] == BLACK:
                    black_pts += 1
                elif self.grid[y, x] == WHITE:
                    white_pts += 1

        seen_empty: set = set()
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y, x] != EMPTY or (y, x) in seen_empty:
                    continue
                # Flood-fill this empty region; track which colors border it.
                region = []
                bordering = set()
                q = deque([(y, x)])
                seen_empty.add((y, x))
                while q:
                    cy, cx = q.popleft()
                    region.append((cy, cx))
                    for ny, nx in self._neighbors(cy, cx):
                        if self.grid[ny, nx] == EMPTY and (ny, nx) not in seen_empty:
                            seen_empty.add((ny, nx))
                            q.append((ny, nx))
                        elif self.grid[ny, nx] in (BLACK, WHITE):
                            bordering.add(int(self.grid[ny, nx]))
                if bordering == {BLACK}:
                    black_pts += len(region)
                elif bordering == {WHITE}:
                    white_pts += len(region)
                # Mixed-border regions are dame; nobody scores.
        white_pts += self.komi
        return black_pts, white_pts

    def winner(self) -> int:
        """Return BLACK or WHITE (no ties with non-integer komi)."""
        b, w = self.tromp_taylor_score()
        return BLACK if b > w else WHITE

    # ------------------------------------------------------------------ copy

    def copy(self) -> "GoBoard":
        new = GoBoard(size=self.size, komi=self.komi)
        new.grid = self.grid.copy()
        new.to_move = self.to_move
        new.consecutive_passes = self.consecutive_passes
        new.move_count = self.move_count
        new._past_states = set(self._past_states)
        return new


# ====================================================================
# Plane encoders — both 4-plane and 17-plane variants
# ====================================================================


def board_to_planes(board: GoBoard) -> np.ndarray:
    """4-plane encoding: black, white, side-to-move, ones.

    Spike-compatible. Use this for the small-net 9x9 demo.
    """
    planes = np.zeros((4, board.size, board.size), dtype=np.float32)
    planes[0] = (board.grid == BLACK).astype(np.float32)
    planes[1] = (board.grid == WHITE).astype(np.float32)
    planes[2] = 1.0 if board.to_move == BLACK else 0.0
    planes[3] = 1.0
    return planes


def board_to_history_planes(
    boards: list[GoBoard],
    n_history: int = 8,
) -> np.ndarray:
    """17-plane AlphaGo-Zero encoding: 8 history × (black, white) + STM.

    `boards` is the list of game positions, most-recent last. Older
    positions are zero-padded if the game is shorter than `n_history`.
    Side-to-move plane reads from the most-recent board.

    Output shape: (2*n_history + 1, size, size) — 17 with n_history=8.
    """
    if not boards:
        raise ValueError("boards must contain at least the current position")
    size = boards[-1].size
    planes = np.zeros((2 * n_history + 1, size, size), dtype=np.float32)
    # Take the last n_history boards; pad with zeros at the front if short.
    history = boards[-n_history:]
    offset = n_history - len(history)
    for i, b in enumerate(history):
        idx = offset + i
        planes[2 * idx + 0] = (b.grid == BLACK).astype(np.float32)
        planes[2 * idx + 1] = (b.grid == WHITE).astype(np.float32)
    planes[-1] = 1.0 if boards[-1].to_move == BLACK else 0.0
    return planes


# ====================================================================
# GTP coordinate helpers (KataGo I/O)
# ====================================================================


def gtp_to_flat(move: str, size: int) -> int:
    """Parse 'D4' / 'pass' / 'resign' to flat index. Resign = pass."""
    m = move.strip().lower()
    if m in ("pass", "resign"):
        return size * size
    col_char = m[0]
    if col_char == "i":
        raise ValueError(f"GTP move '{move}' uses reserved column 'I'")
    col = ord(col_char) - ord("a")
    if col_char > "i":
        col -= 1
    row_one_indexed = int(m[1:])
    y = size - row_one_indexed
    x = col
    return y * size + x


def flat_to_gtp(idx: int, size: int) -> str:
    """Convert flat index to GTP move string. size² → 'pass'."""
    if idx == size * size:
        return "pass"
    y, x = divmod(idx, size)
    col_letter = chr(ord("a") + (x if x < 8 else x + 1))
    row = size - y
    return f"{col_letter.upper()}{row}"
