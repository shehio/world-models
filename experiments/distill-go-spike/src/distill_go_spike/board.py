"""Minimal Go board representation for the KataGo spike.

Keeps the input-plane shape similar to AlphaGo Zero's representation so
the existing ResNet body (in `wm_chess.network`) transfers cleanly:
  - plane 0: black stones on the board
  - plane 1: white stones on the board
  - plane 2: side to move (all 1s if black, all 0s if white)
  - plane 3: ones plane (helps the conv kernel detect board edges)

That's only 4 planes — AlphaGo Zero used 17 (8 history × 2 + side-to-move).
For a spike we drop history; it can be added incrementally without changing
the training shape (just bump N_INPUT_PLANES + extend board_to_planes).

The board is 19×19 (full Go) by default. The spike accepts 9 / 13 / 19 so
we can iterate fast on the smaller boards before paying for full-size data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


BOARD_SIZE = 19
N_INPUT_PLANES = 4  # see module docstring

EMPTY = 0
BLACK = 1
WHITE = 2


@dataclass
class GoBoard:
    """Tiny Go board for the spike.

    We don't implement legality (KataGo handles that). The board only
    tracks stone placement so we can serialize state for the network.
    """

    size: int = BOARD_SIZE
    grid: np.ndarray = field(init=False)
    to_move: int = BLACK

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

    def play(self, x: int, y: int) -> None:
        """Place a stone for the current side and flip turn.

        KataGo enforces legality upstream — this is just bookkeeping.
        """
        if x == -1 and y == -1:  # pass
            self.to_move = WHITE if self.to_move == BLACK else BLACK
            return
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError(f"out-of-bounds move ({x}, {y}) on size {self.size}")
        self.grid[y, x] = self.to_move
        self.to_move = WHITE if self.to_move == BLACK else BLACK

    def copy(self) -> "GoBoard":
        new = GoBoard(size=self.size)
        new.grid = self.grid.copy()
        new.to_move = self.to_move
        return new


def board_to_planes(board: GoBoard) -> np.ndarray:
    """Serialize a board to (N_INPUT_PLANES, size, size) float32 planes."""
    planes = np.zeros((N_INPUT_PLANES, board.size, board.size), dtype=np.float32)
    planes[0] = (board.grid == BLACK).astype(np.float32)
    planes[1] = (board.grid == WHITE).astype(np.float32)
    planes[2] = 1.0 if board.to_move == BLACK else 0.0
    planes[3] = 1.0  # ones plane
    return planes


def gtp_move_to_xy(move: str, size: int = BOARD_SIZE) -> tuple[int, int]:
    """Parse a GTP coordinate like 'D4' or 'pass' into (x, y) or (-1, -1).

    GTP skips column 'I' (visually similar to '1'). Rows are 1-indexed
    from the bottom; we convert to top-down (y=0 is top row) for numpy.
    """
    m = move.strip().lower()
    if m in ("pass", "resign"):
        return (-1, -1)
    if not m:
        raise ValueError("empty GTP move")
    col_char = m[0]
    if col_char == "i":
        raise ValueError(f"GTP move '{move}' uses reserved column 'I'")
    col = ord(col_char) - ord("a")
    if col_char > "i":
        col -= 1  # skip 'I'
    row_one_indexed = int(m[1:])
    y_from_top = size - row_one_indexed
    return (col, y_from_top)


def xy_to_gtp_move(x: int, y: int, size: int = BOARD_SIZE) -> str:
    if x == -1 and y == -1:
        return "pass"
    col_letter = chr(ord("a") + (x if x < 8 else x + 1))  # skip 'I'
    row = size - y
    return f"{col_letter.upper()}{row}"
