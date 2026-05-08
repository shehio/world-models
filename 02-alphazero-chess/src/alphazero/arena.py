"""Match two policies head-to-head, or evaluate vs a fixed baseline.

Three useful baselines:
  - Random: uniform over legal moves. Should be trivially beaten.
  - Stockfish at depth 1: weak but materially aware. Real signal.
  - Prior checkpoint: monotonic-improvement check during training.

Each match plays N games with colors alternating. Reports win/draw/loss
and a 95% Wilson CI on win rate. With N=20 games you can distinguish
"clearly better" from "noise" but not 50% vs 55%.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations


def play_match(policy_a, policy_b, n_games: int) -> dict:
    """Returns {'wins': int, 'draws': int, 'losses': int, 'win_rate': float}."""
    raise NotImplementedError


def random_policy(board):
    """Uniform random over legal moves. Used as floor baseline."""
    raise NotImplementedError


def stockfish_policy(depth: int):
    """Returns a policy callable that wraps Stockfish at fixed depth."""
    raise NotImplementedError
