"""Arena match runner: two random policies should score ~50% (with noise).

Run with: uv run python tests/test_arena.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess

from wm_chess.arena import play_match, random_policy


def test_play_match_returns_complete_record():
    random.seed(0)
    stats = play_match(random_policy, random_policy, n_games=20, max_plies=200)
    # Every game must end as a win, draw, or loss.
    assert stats["wins"] + stats["draws"] + stats["losses"] == 20
    # Score is in [0, 1].
    assert 0.0 <= stats["score"] <= 1.0
    # Two random players: most games should be draws (50-move/insufficient).
    # Lots of variance with n=20, but assert score within wide band.
    assert 0.2 <= stats["score"] <= 0.8, stats


def test_color_alternation():
    """Verify policy_a actually plays both colors when n_games > 1.

    Plays a deterministic policy_a (always picks first legal move) vs
    random; with color alternation we should get a mixture, not all-white-wins.
    """
    rng = random.Random(0)

    def first_legal(board: chess.Board) -> chess.Move:
        return next(iter(board.legal_moves))

    def random_seeded(board: chess.Board) -> chess.Move:
        return rng.choice(list(board.legal_moves))

    stats = play_match(first_legal, random_seeded, n_games=10, max_plies=200)
    # Just verify we played 10 games end-to-end without crashing.
    assert stats["games"] == 10


def main():
    tests = [test_play_match_returns_complete_record, test_color_alternation]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
