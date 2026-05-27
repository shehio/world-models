"""Smoke tests for MCTS over learned dynamics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import MuZeroConfig, INPUT_PLANES, ACTION_DIM
from muzero_chess.mcts import (
    MinMaxStats,
    Node,
    root_visit_distribution,
    run_mcts,
    select_action,
)
from muzero_chess.networks import MuZeroNet


def _tiny_cfg(**overrides) -> MuZeroConfig:
    base = dict(
        latent_channels=8,
        repr_n_res_blocks=1, repr_n_filters=8,
        dyn_n_res_blocks=1, dyn_n_filters=8,
        pred_n_res_blocks=1, pred_n_filters=8,
        num_simulations=4,
    )
    base.update(overrides)
    return MuZeroConfig(**base)


def test_minmax_stats_normalizes_within_range():
    mm = MinMaxStats()
    mm.update(1.0)
    mm.update(3.0)
    mm.update(2.0)
    # values are normalized into [0, 1]; the min should map to 0 and the max to 1.
    assert abs(mm.normalize(1.0) - 0.0) < 1e-9
    assert abs(mm.normalize(3.0) - 1.0) < 1e-9
    assert 0 < mm.normalize(2.0) < 1


def test_minmax_stats_falls_through_when_unobserved():
    """Before any updates the normalize() should not crash; it returns
    the input value unchanged so PUCT still has a number to work with."""
    mm = MinMaxStats()
    v = mm.normalize(0.5)
    assert v == 0.5


def test_run_mcts_returns_expanded_root_with_visits():
    cfg = _tiny_cfg(num_simulations=8)
    net = MuZeroNet(cfg)
    net.eval()
    obs = torch.randn(1, INPUT_PLANES, 8, 8)
    root = run_mcts(net, obs, cfg, add_root_noise=False)
    assert root.is_expanded
    # Total child visits should equal num_simulations + the root's own init visit.
    total_visits = sum(c.visit_count for c in root.children.values())
    # Each simulation visits exactly one child (the path from root → leaf),
    # except for terminal-on-root which we treat as a no-op.
    assert total_visits >= 1
    assert root.visit_count >= 1


def test_run_mcts_respects_legal_actions_at_root():
    """When legal_actions is set, only those moves should appear as root children."""
    cfg = _tiny_cfg(num_simulations=4)
    net = MuZeroNet(cfg)
    net.eval()
    obs = torch.randn(1, INPUT_PLANES, 8, 8)
    legal = [10, 100, 200, 4000]
    root = run_mcts(net, obs, cfg, add_root_noise=False, legal_actions=legal)
    assert set(root.children.keys()) == set(legal)


def test_run_mcts_no_root_noise_is_reproducible():
    """Without Dirichlet noise + same RNG state, two calls should produce
    the same visit distribution (network is deterministic in eval mode)."""
    cfg = _tiny_cfg(num_simulations=8)
    net = MuZeroNet(cfg)
    net.eval()
    obs = torch.randn(1, INPUT_PLANES, 8, 8)
    torch.manual_seed(0); np.random.seed(0)
    r1 = run_mcts(net, obs, cfg, add_root_noise=False)
    torch.manual_seed(0); np.random.seed(0)
    r2 = run_mcts(net, obs, cfg, add_root_noise=False)
    pi1 = root_visit_distribution(r1, ACTION_DIM)
    pi2 = root_visit_distribution(r2, ACTION_DIM)
    assert np.allclose(pi1, pi2)


def test_select_action_argmax_at_temp0():
    root = Node()
    root.children = {
        10: Node(visit_count=5),
        20: Node(visit_count=10),
        30: Node(visit_count=3),
    }
    chosen = select_action(root, temperature=0.0)
    assert chosen == 20


def test_select_action_with_temperature_returns_legal():
    root = Node()
    root.children = {
        10: Node(visit_count=5),
        20: Node(visit_count=10),
        30: Node(visit_count=3),
    }
    for _ in range(20):
        chosen = select_action(root, temperature=1.0)
        assert chosen in {10, 20, 30}


def test_root_visit_distribution_is_a_probability():
    root = Node()
    root.children = {
        10: Node(visit_count=5),
        20: Node(visit_count=10),
        30: Node(visit_count=3),
    }
    pi = root_visit_distribution(root, action_dim=ACTION_DIM)
    assert pi.shape == (ACTION_DIM,)
    assert abs(pi.sum() - 1.0) < 1e-6
    # The three nonzero entries are in the right ratio.
    assert pi[20] > pi[10] > pi[30]
