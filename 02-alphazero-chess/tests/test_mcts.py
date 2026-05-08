"""MCTS behavior tests.

Verifies:
  - visit counts at root sum to num_sims (exactly one new visit per sim)
  - all visited children are LEGAL moves
  - terminal positions don't crash and propagate proper values
  - select_move with τ=0 picks the most-visited move
  - Dirichlet noise actually changes priors

Run with: uv run python tests/test_mcts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import numpy as np
import torch

from alphazero.config import Config
from alphazero.mcts import (
    Node,
    _add_dirichlet_noise,
    _expand,
    _terminal_value,
    run_mcts,
    select_move,
    visits_to_distribution,
)
from alphazero.network import AlphaZeroNet


def _make_net():
    cfg = Config()
    net = AlphaZeroNet(cfg).cpu()
    net.eval()
    return cfg, net, torch.device("cpu")


def test_visit_counts_sum_to_num_sims():
    cfg, net, dev = _make_net()
    board = chess.Board()
    n = 30
    visits = run_mcts(board, net, num_sims=n, c_puct=cfg.c_puct,
                     add_root_noise=False, device=dev)
    total = sum(visits.values())
    assert total == n, f"expected {n} root visits, got {total}"


def test_all_visited_children_are_legal():
    cfg, net, dev = _make_net()
    board = chess.Board()
    visits = run_mcts(board, net, num_sims=20, c_puct=cfg.c_puct,
                     add_root_noise=False, device=dev)
    legal = set(board.legal_moves)
    for move in visits:
        assert move in legal, f"{move.uci()} not legal at start position"


def test_terminal_value_signs():
    """Black just got mated; from white's POV (white-to-move at the prev ply)
    we want value = +1. From black's POV (side-to-move at the terminal node)
    we want -1 because the side-to-move just got mated."""
    # Fool's mate: 1.f3 e5 2.g4 Qh4# (black mates white).
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate(), board.fen()
    # Side to move is white (just got mated). _terminal_value returns from
    # POV of side-to-move, so should be -1.
    assert _terminal_value(board) == -1.0
    # Stalemate / draw: empty kings.
    draw_board = chess.Board("8/8/8/8/8/k7/p7/K7 w - - 0 1")
    # Force stalemate by playing king to a tight spot — easier: hand a pos
    stale = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    if stale.is_stalemate():
        assert _terminal_value(stale) == 0.0
    # Just check the empty-game-over branch returns 0 too.
    not_terminal = chess.Board()
    assert _terminal_value(not_terminal) == 0.0


def test_run_mcts_handles_terminal_root():
    """If root is already terminal, MCTS should still return some result without crashing."""
    cfg, net, dev = _make_net()
    # Mate position from above.
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    # Root has no legal moves. MCTS expansion creates 0 children; sims do nothing useful.
    visits = run_mcts(board, net, num_sims=5, c_puct=cfg.c_puct,
                     add_root_noise=False, device=dev)
    assert visits == {}, visits


def test_select_move_temperature_zero_is_argmax():
    visits = {
        chess.Move.from_uci("e2e4"): 7,
        chess.Move.from_uci("d2d4"): 12,
        chess.Move.from_uci("g1f3"): 4,
    }
    for _ in range(20):
        m = select_move(visits, temperature=0.0)
        assert m == chess.Move.from_uci("d2d4")


def test_select_move_temperature_one_is_proportional():
    """With τ=1, sampled probabilities should match visit fractions in expectation."""
    visits = {
        chess.Move.from_uci("e2e4"): 80,
        chess.Move.from_uci("d2d4"): 20,
    }
    counts = {chess.Move.from_uci("e2e4"): 0, chess.Move.from_uci("d2d4"): 0}
    np.random.seed(0)
    for _ in range(2000):
        m = select_move(visits, temperature=1.0)
        counts[m] += 1
    frac_e4 = counts[chess.Move.from_uci("e2e4")] / 2000
    assert 0.72 < frac_e4 < 0.88, frac_e4  # within sampling noise of 0.8


def test_dirichlet_noise_changes_priors():
    cfg, net, dev = _make_net()
    board = chess.Board()
    root = Node()
    _expand(root, board, net, dev)
    before = [c.prior for c in root.children.values()]
    np.random.seed(123)
    _add_dirichlet_noise(root, alpha=0.3, eps=0.25)
    after = [c.prior for c in root.children.values()]
    # Some prior must move; sum still ≈ 1.
    assert any(abs(a - b) > 1e-3 for a, b in zip(before, after))
    assert abs(sum(after) - 1.0) < 1e-3


def test_visits_to_distribution_is_a_distribution():
    cfg, net, dev = _make_net()
    board = chess.Board()
    visits = run_mcts(board, net, num_sims=30, c_puct=cfg.c_puct,
                     add_root_noise=True, device=dev)
    pi = visits_to_distribution(visits, board)
    assert pi.shape == (4672,)
    assert abs(pi.sum() - 1.0) < 1e-5
    assert (pi >= 0).all()


def main():
    tests = [
        test_visit_counts_sum_to_num_sims,
        test_all_visited_children_are_legal,
        test_terminal_value_signs,
        test_run_mcts_handles_terminal_root,
        test_select_move_temperature_zero_is_argmax,
        test_select_move_temperature_one_is_proportional,
        test_dirichlet_noise_changes_priors,
        test_visits_to_distribution_is_a_distribution,
    ]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
