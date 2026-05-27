"""Smoke tests for the MuZero networks — shapes + interaction."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import MuZeroConfig, INPUT_PLANES, ACTION_DIM, BOARD_SIZE, N_MOVE_PLANES
from muzero_chess.networks import (
    DynamicsNet,
    MuZeroNet,
    PredictionNet,
    RepresentationNet,
    _action_to_plane,
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


def test_action_plane_one_hot_at_correct_cell():
    """The action plane should have exactly one set cell per batch element,
    located at (plane_idx, from_rank, from_file) — matching encode_move."""
    # action = plane * 64 + from_sq; pick a few specific actions and check.
    cases = [
        # (action, expected plane_idx, from_sq)
        (0, 0, 0),                  # plane 0, square a1
        (63, 0, 63),                # plane 0, square h8
        (64, 1, 0),                 # plane 1, square a1
        (72 * 64 + 27, 72, 27),     # plane 72, square d4
    ]
    a = torch.tensor([c[0] for c in cases], dtype=torch.long)
    plane = _action_to_plane(a, device=torch.device("cpu"), dtype=torch.float32)
    assert plane.shape == (len(cases), N_MOVE_PLANES, BOARD_SIZE, BOARD_SIZE)
    # Exactly one set cell per batch element.
    assert (plane.sum(dim=(1, 2, 3)) == 1).all()
    for i, (_, plane_idx, from_sq) in enumerate(cases):
        rank, file = divmod(from_sq, BOARD_SIZE)
        assert plane[i, plane_idx, rank, file] == 1.0


def test_dynamics_distinguishes_source_square_AND_move_type():
    """Regression guard for the action-encoding bug: two actions that share
    move-type but differ in source square should produce DIFFERENT next_latents.
    Likewise two actions that share source square but differ in move-type
    must also differ. The old `action // 73` encoding collided in both cases."""
    cfg = _tiny_cfg()
    g = DynamicsNet(cfg)
    g.eval()
    s = torch.randn(1, cfg.latent_channels, BOARD_SIZE, BOARD_SIZE)
    s = s.expand(3, -1, -1, -1).contiguous()
    # Three actions:
    #   A: plane 0, sq 0
    #   B: plane 0, sq 27   (same plane, different sq — different next latent)
    #   C: plane 5, sq 0    (same sq, different plane — different next latent)
    a = torch.tensor([0 * 64 + 0, 0 * 64 + 27, 5 * 64 + 0], dtype=torch.long)
    with torch.no_grad():
        s_next, _ = g(s, a)
    # A vs B (source-square differs): must differ.
    assert not torch.allclose(s_next[0], s_next[1], atol=1e-5), \
        "DynamicsNet collapsed two different source squares to the same next latent"
    # A vs C (move-type differs): must differ.
    assert not torch.allclose(s_next[0], s_next[2], atol=1e-5), \
        "DynamicsNet collapsed two different move-types to the same next latent"


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
