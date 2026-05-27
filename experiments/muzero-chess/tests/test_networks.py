"""Smoke tests for the MuZero networks — shapes + interaction."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import MuZeroConfig, INPUT_PLANES, ACTION_DIM, BOARD_SIZE
from muzero_chess.networks import (
    DynamicsNet,
    MuZeroNet,
    PredictionNet,
    RepresentationNet,
)


def _tiny_cfg() -> MuZeroConfig:
    """Smallest meaningful config for fast tests."""
    return MuZeroConfig(
        latent_channels=8,
        repr_n_res_blocks=1, repr_n_filters=8,
        dyn_n_res_blocks=1, dyn_n_filters=8,
        pred_n_res_blocks=1, pred_n_filters=8,
    )


def test_representation_net_shapes():
    cfg = _tiny_cfg()
    h = RepresentationNet(cfg)
    obs = torch.randn(2, INPUT_PLANES, 8, 8)
    s = h(obs)
    assert s.shape == (2, cfg.latent_channels, BOARD_SIZE, BOARD_SIZE)


def test_dynamics_net_shapes():
    cfg = _tiny_cfg()
    g = DynamicsNet(cfg)
    s = torch.randn(2, cfg.latent_channels, BOARD_SIZE, BOARD_SIZE)
    a = torch.randint(0, ACTION_DIM, (2,))
    s_next, r = g(s, a)
    assert s_next.shape == s.shape
    assert r.shape == (2,)
    # Reward is tanh-bounded.
    assert (r.abs() <= 1.0).all()


def test_prediction_net_shapes():
    cfg = _tiny_cfg()
    f = PredictionNet(cfg)
    s = torch.randn(2, cfg.latent_channels, BOARD_SIZE, BOARD_SIZE)
    p, v = f(s)
    assert p.shape == (2, ACTION_DIM)
    assert v.shape == (2,)
    assert (v.abs() <= 1.0).all()


def test_muzero_net_initial_then_recurrent():
    """The two-stage inference pattern MCTS actually uses."""
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    obs = torch.randn(2, INPUT_PLANES, 8, 8)
    s0, p0, v0 = net.initial_inference(obs)
    assert s0.shape == (2, cfg.latent_channels, BOARD_SIZE, BOARD_SIZE)
    assert p0.shape == (2, ACTION_DIM)
    assert v0.shape == (2,)
    a = torch.randint(0, ACTION_DIM, (2,))
    s1, r1, p1, v1 = net.recurrent_inference(s0, a)
    # Recurrent step preserves latent shape — so we can keep unrolling.
    assert s1.shape == s0.shape
    assert r1.shape == (2,)
    assert p1.shape == (2, ACTION_DIM)
    assert v1.shape == (2,)


def test_latent_doesnt_collapse_to_zero():
    """Sanity: a freshly-initialized h_θ should produce non-zero outputs.

    Catches BN scale being clamped wrong or weight init zeroing the net.
    """
    cfg = _tiny_cfg()
    h = RepresentationNet(cfg)
    h.eval()    # use running stats so output is deterministic in this test
    obs = torch.randn(4, INPUT_PLANES, 8, 8)
    s = h(obs)
    assert s.abs().sum() > 1.0, "representation output is degenerate"


def test_param_counts_split_evenly_ish_across_three_nets():
    """Sanity: the three nets should each have a meaningful number of params.

    Catches a case where one head is misconfigured to zero (e.g. all 1x1
    convs and no trunk).
    """
    cfg = MuZeroConfig()      # paper-ish defaults
    net = MuZeroNet(cfg)
    h_params = sum(p.numel() for p in net.h.parameters())
    g_params = sum(p.numel() for p in net.g.parameters())
    f_params = sum(p.numel() for p in net.f.parameters())
    assert h_params > 1000
    assert g_params > 1000
    assert f_params > 1000
