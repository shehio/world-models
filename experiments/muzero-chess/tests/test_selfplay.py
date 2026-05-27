"""Smoke test: self-play runs a real chess game using MCTS over learned dynamics."""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import ACTION_DIM, MuZeroConfig
from muzero_chess.networks import MuZeroNet
from muzero_chess.selfplay import play_game

from wm_chess.board import decode_move


def _tiny_cfg(**overrides) -> MuZeroConfig:
    base = dict(
        latent_channels=8,
        repr_n_res_blocks=1, repr_n_filters=8,
        dyn_n_res_blocks=1, dyn_n_filters=8,
        pred_n_res_blocks=1, pred_n_filters=8,
        num_simulations=4,
        max_plies=12,
        temp_moves=4,
    )
    base.update(overrides)
    return MuZeroConfig(**base)


def test_play_game_terminates_under_max_plies():
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    net.eval()
    game = play_game(net, cfg, device=torch.device("cpu"))
    assert game.length <= cfg.max_plies
    assert game.length >= 1, "should have played at least one move"
    # Lengths line up between the parallel arrays.
    assert len(game.observations) == game.length
    assert len(game.actions) == game.length
    assert len(game.pis) == game.length
    assert len(game.z_per_ply) == game.length
    assert len(game.r_per_action) == game.length


def test_play_game_only_takes_legal_moves():
    """Replay the recorded actions on a fresh board — every move must be legal."""
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    net.eval()
    game = play_game(net, cfg, device=torch.device("cpu"))
    board = chess.Board()
    for action in game.actions:
        move = decode_move(action, board)
        assert move in board.legal_moves, (
            f"Recorded action {action} → move {move} was illegal at {board.fen()}"
        )
        board.push(move)


def test_play_game_outcome_consistent_with_terminal_reward():
    """If the game ended in checkmate the terminal reward should be 1.0;
    otherwise it should be 0.0. (Z targets are derived from these.)"""
    cfg = _tiny_cfg(max_plies=8)
    net = MuZeroNet(cfg)
    net.eval()
    game = play_game(net, cfg, device=torch.device("cpu"))
    # Replay to inspect the final board state.
    board = chess.Board()
    for action in game.actions:
        board.push(decode_move(action, board))
    if board.is_checkmate():
        assert game.r_per_action[-1] == 1.0
    else:
        assert game.r_per_action[-1] == 0.0


def test_play_game_pis_are_distributions_over_action_dim():
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    net.eval()
    game = play_game(net, cfg, device=torch.device("cpu"))
    for pi in game.pis:
        assert pi.shape == (ACTION_DIM,)
        # MCTS visit distribution sums to 1 over the legal moves.
        assert abs(pi.sum() - 1.0) < 1e-5
        # Non-negative.
        assert (pi >= 0).all()
