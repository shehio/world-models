"""Smoke tests for the MuZero vs Stockfish eval path.

These tests need stockfish on PATH. Skipped automatically when it's not."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import MuZeroConfig
from muzero_chess.eval import evaluate_vs_stockfish, muzero_policy
from muzero_chess.networks import MuZeroNet


STOCKFISH_AVAILABLE = shutil.which("stockfish") is not None
needs_stockfish = pytest.mark.skipif(
    not STOCKFISH_AVAILABLE,
    reason="stockfish binary not on PATH (cloud has it; local optional)",
)


def _tiny_cfg(**overrides) -> MuZeroConfig:
    base = dict(
        latent_channels=8,
        repr_n_res_blocks=1, repr_n_filters=8,
        dyn_n_res_blocks=1, dyn_n_filters=8,
        pred_n_res_blocks=1, pred_n_filters=8,
        num_simulations=4,
        max_plies=12,
        temp_moves=0,
    )
    base.update(overrides)
    return MuZeroConfig(**base)


def test_muzero_policy_returns_legal_moves():
    """The MuZero policy is what we feed into arena.play_match. Calling it
    on a fresh board should always return one of the legal moves at the root.
    No stockfish required for this test."""
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    net.eval()
    pol = muzero_policy(net, cfg)
    board = chess.Board()
    for _ in range(5):
        move = pol(board)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break


@needs_stockfish
def test_evaluate_vs_stockfish_returns_well_formed_stats():
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    net.eval()
    stats = evaluate_vs_stockfish(
        net, cfg, device=torch.device("cpu"),
        stockfish_elo=1320, n_games=2, max_plies=20,
    )
    assert set(stats) >= {"games", "wins", "draws", "losses", "score"}
    assert stats["games"] == 2
    assert stats["wins"] + stats["draws"] + stats["losses"] == 2
    assert 0.0 <= stats["score"] <= 1.0
