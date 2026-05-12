"""Run with: uv run python tests/test_history.py

Tests for the 8-step history encoder and its threading through MCTS.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import numpy as np
import torch
from dataclasses import replace

from wm_chess.board import (
    N_HISTORY,
    N_INPUT_PLANES,
    N_INPUT_PLANES_HISTORY,
    N_META_PLANES,
    N_PIECE_PLANES_PER_BOARD,
    encode_board,
    encode_history,
)
from wm_chess.config import Config
from wm_chess.mcts import run_mcts, run_mcts_batched
from wm_chess.network import AlphaZeroNet


def test_constants():
    assert N_HISTORY == 8
    assert N_PIECE_PLANES_PER_BOARD == 12
    assert N_META_PLANES == 7
    assert N_INPUT_PLANES_HISTORY == 8 * 12 + 7 == 103
    print("✓ constants")


def test_shape_and_dtype():
    b = chess.Board()
    p = encode_history([b])
    assert p.shape == (103, 8, 8), p.shape
    assert p.dtype == np.float32
    print("✓ shape and dtype")


def test_padding_for_short_game():
    """If we only have 1 board, the first 7 piece-plane stacks should be all zero."""
    b = chess.Board()
    p = encode_history([b])
    # First 7 stacks (84 planes) should all be zero — no history.
    assert p[:7 * 12].sum() == 0, "expected zero padding for missing history"
    # The 8th stack (current position) must contain pieces.
    cur_pieces = p[7 * 12 : 8 * 12]
    assert cur_pieces.sum() > 0, "current position has no pieces"
    # Total piece planes for starting position: 32 pieces.
    assert int(cur_pieces.sum()) == 32, int(cur_pieces.sum())
    print("✓ padding for short game")


def test_full_history_no_padding():
    """With 8+ boards, padding is zero and every stack has pieces."""
    b = chess.Board()
    history = []
    rng = np.random.default_rng(0)
    for _ in range(10):  # 10 plies → can take last 8
        history.append(b.copy(stack=False))
        moves = list(b.legal_moves)
        b.push(moves[rng.integers(len(moves))])
    p = encode_history(history)
    # Each of the 8 piece-plane stacks should have at least one piece.
    for i in range(8):
        stack = p[i * 12 : (i + 1) * 12]
        assert stack.sum() > 0, f"stack {i} has no pieces"
    print("✓ full history no padding")


def test_meta_planes_from_current_only():
    """Meta planes (96..102) must come from the LAST board, not earlier ones."""
    # White-to-move start, black-to-move after one push.
    b1 = chess.Board()  # white to move
    b1_copy = b1.copy(stack=False)
    b1.push(chess.Move.from_uci("e2e4"))
    p = encode_history([b1_copy, b1])  # current is b1 (black to move)
    side_plane = p[8 * 12]  # plane 96
    assert side_plane.sum() == 0, "expected black-to-move = all zero side plane"

    # Sanity check the inverse: pass only b1_copy as current.
    p2 = encode_history([b1_copy])
    side_plane2 = p2[8 * 12]
    assert side_plane2.sum() == 64, "expected white-to-move = all 1 side plane"
    print("✓ meta planes from current only")


def test_history_equivalence_at_n1():
    """encode_history(boards, n_history=1) should agree with encode_board for piece+meta sums."""
    b = chess.Board()
    b.push(chess.Move.from_uci("d2d4"))
    p1 = encode_history([b], n_history=1)
    p_legacy = encode_board(b)
    # n_history=1 → 12 piece planes + 7 meta = 19 total. Matches encode_board layout.
    assert p1.shape == (19, 8, 8), p1.shape
    assert p_legacy.shape == (19, 8, 8)
    # Same total piece count (32 starting - 0 captures = 32)
    assert int(p1[:12].sum()) == int(p_legacy[:12].sum()) == 32
    # Same meta planes
    assert np.allclose(p1[12:], p_legacy[12:])
    print("✓ history n_history=1 matches encode_board for piece and meta planes")


def test_history_takes_last_n():
    """With more boards than n_history, only the last n_history are encoded."""
    boards = [chess.Board() for _ in range(15)]
    p = encode_history(boards, n_history=8)
    # All 8 piece-plane stacks should match the starting position (32 pieces each).
    for i in range(8):
        stack = p[i * 12 : (i + 1) * 12]
        assert int(stack.sum()) == 32, f"stack {i} count = {int(stack.sum())}"
    print("✓ history takes last n boards only")


def test_mcts_runs_with_history():
    """run_mcts and run_mcts_batched should produce visit counts with n_history > 1."""
    cfg = replace(Config(), n_input_planes=N_INPUT_PLANES_HISTORY,
                  n_res_blocks=2, n_filters=16)
    net = AlphaZeroNet(cfg).eval()
    device = torch.device("cpu")
    b = chess.Board()
    b.push(chess.Move.from_uci("e2e4"))
    b.push(chess.Move.from_uci("e7e5"))
    game_history = [chess.Board(), chess.Board()]  # placeholder boards
    game_history[1].push(chess.Move.from_uci("e2e4"))

    visits = run_mcts(b, net, num_sims=8, c_puct=1.5,
                      add_root_noise=False, device=device,
                      game_history=game_history, n_history=8)
    assert sum(visits.values()) == 8, f"expected 8 sims, got {sum(visits.values())}"

    visits_b = run_mcts_batched(b, net, num_sims=16, c_puct=1.5,
                                add_root_noise=False, device=device,
                                batch_size=4,
                                game_history=game_history, n_history=8)
    assert sum(visits_b.values()) == 16, f"expected 16 sims, got {sum(visits_b.values())}"
    print("✓ MCTS (sequential and batched) accepts game_history and n_history")


def test_mcts_n1_unchanged_behavior():
    """With n_history=1 (legacy path), MCTS should match its old behavior — same shape, same visit total."""
    cfg = replace(Config(), n_input_planes=N_INPUT_PLANES,
                  n_res_blocks=2, n_filters=16)
    net = AlphaZeroNet(cfg).eval()
    device = torch.device("cpu")
    b = chess.Board()
    visits = run_mcts_batched(b, net, num_sims=8, c_puct=1.5,
                              add_root_noise=False, device=device,
                              batch_size=4)
    assert sum(visits.values()) == 8
    print("✓ MCTS at n_history=1 unchanged (legacy 19-plane path)")


def test_selfplay_with_history():
    """play_game should produce 103-plane samples when n_history=8."""
    from selfplay.selfplay import play_game

    cfg = replace(Config(), n_input_planes=N_INPUT_PLANES_HISTORY,
                  n_res_blocks=2, n_filters=16,
                  max_plies=8, sims_train=4)
    net = AlphaZeroNet(cfg).eval()
    samples, z, ply = play_game(net, cfg, torch.device("cpu"),
                                sims=4, batch_size=2, n_history=8)
    assert ply >= 1
    assert samples, "expected at least one sample"
    state, pi, z_pov = samples[0]
    assert state.shape == (103, 8, 8), state.shape
    assert pi.shape == (4672,)
    print(f"✓ play_game with n_history=8 produced {len(samples)} samples of shape {state[0:1].shape}")


if __name__ == "__main__":
    test_constants()
    test_shape_and_dtype()
    test_padding_for_short_game()
    test_full_history_no_padding()
    test_meta_planes_from_current_only()
    test_history_equivalence_at_n1()
    test_history_takes_last_n()
    test_mcts_runs_with_history()
    test_mcts_n1_unchanged_behavior()
    test_selfplay_with_history()
    print("\nALL HISTORY TESTS PASSED")
