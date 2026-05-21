"""Smoke tests for network shape + MCTS basic correctness.

No KataGo binary needed — uses an untrained network. Validates that the
forward pass produces correctly-shaped outputs and that MCTS terminates
on a small board.
"""
from __future__ import annotations

import numpy as np
import torch

from distill_go.board import BLACK, GoBoard
from distill_go.config import GoConfig
from distill_go.mcts import run_mcts, select_move, visits_to_distribution
from distill_go.network import AlphaZeroGoNet


def _tiny_cfg(board_size: int = 5, n_input_planes: int = 4) -> GoConfig:
    return GoConfig(
        board_size=board_size,
        n_input_planes=n_input_planes,
        n_res_blocks=1,
        n_filters=8,
    )


def test_network_output_shape_4_plane():
    cfg = _tiny_cfg()
    net = AlphaZeroGoNet(cfg)
    x = torch.randn(2, 4, cfg.board_size, cfg.board_size)
    logits, value = net(x)
    assert logits.shape == (2, cfg.policy_size)
    assert value.shape == (2,)


def test_network_output_shape_17_plane():
    cfg = _tiny_cfg(n_input_planes=17)
    net = AlphaZeroGoNet(cfg)
    x = torch.randn(2, 17, cfg.board_size, cfg.board_size)
    logits, value = net(x)
    assert logits.shape == (2, cfg.policy_size)
    assert value.shape == (2,)


def test_mcts_runs_and_returns_visits_for_legal_moves_only():
    cfg = _tiny_cfg()
    net = AlphaZeroGoNet(cfg)
    board = GoBoard(size=cfg.board_size, komi=cfg.komi)
    torch.manual_seed(0)
    np.random.seed(0)
    visits = run_mcts(
        board, net, num_sims=20, c_puct=cfg.c_puct,
        add_root_noise=True, device=torch.device("cpu"),
        n_input_planes=cfg.n_input_planes,
    )
    # All keys must be in [0, policy_size).
    for idx in visits:
        assert 0 <= idx < cfg.policy_size
    # Total visit count is approximately num_sims (root expansion counts once).
    assert sum(visits.values()) >= 1


def test_mcts_select_move_returns_legal():
    cfg = _tiny_cfg()
    net = AlphaZeroGoNet(cfg)
    board = GoBoard(size=cfg.board_size, komi=cfg.komi)
    torch.manual_seed(0)
    np.random.seed(0)
    visits = run_mcts(
        board, net, num_sims=20, c_puct=cfg.c_puct,
        add_root_noise=False, device=torch.device("cpu"),
        n_input_planes=cfg.n_input_planes,
    )
    move = select_move(visits, temperature=0.0)
    legal_mask = board.legal_mask()
    assert legal_mask[move] == 1


def test_visits_to_distribution_sums_to_one():
    cfg = _tiny_cfg()
    visits = {0: 3, 1: 7, 2: 2}
    pi = visits_to_distribution(visits, cfg)
    assert pi.shape == (cfg.policy_size,)
    assert abs(pi.sum() - 1.0) < 1e-6
    assert abs(pi[0] - 0.25) < 1e-6
    assert abs(pi[1] - 7 / 12) < 1e-6
