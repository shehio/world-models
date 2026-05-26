"""Smoke test for the Go self-play loop's play_game_pcr.

Don't run the full multi-process loop here — that's covered by manual
smoke runs on a real machine. This test just verifies:
  - play_game_pcr completes on a tiny config
  - Returns valid (state, π, z) samples with correct shapes
  - Recorded targets respect p_full (full positions only, not reduced)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from distill_go.config import GoConfig
from distill_go.network import AlphaZeroGoNet
from selfplay_loop import play_game_pcr


def test_play_game_pcr_basic_shapes():
    cfg = GoConfig(
        board_size=5,        # tiny board for speed
        n_input_planes=4,
        n_res_blocks=1,
        n_filters=8,
        c_puct=1.5,
        temp_moves=10,
        max_plies=40,
    )
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    net.eval()

    samples, z, ply, stats = play_game_pcr(
        net, cfg, device,
        sims_full=4,
        sims_reduced=2,
        p_full=1.0,         # every move records
        max_moves=40,
    )

    assert ply > 0, "game should have played at least one move"
    assert stats["full_moves"] >= 1
    assert stats["reduced_moves"] == 0   # p_full=1.0
    assert stats["recorded_targets"] == len(samples)
    assert len(samples) == stats["full_moves"]
    assert z in (-1.0, 0.0, 1.0)

    expected_policy_dim = cfg.policy_size  # 5² + 1 = 26
    for state, pi, sz in samples:
        assert state.shape == (cfg.n_input_planes, cfg.board_size, cfg.board_size)
        assert state.dtype == np.float32
        assert pi.shape == (expected_policy_dim,)
        assert pi.dtype == np.float32
        assert abs(pi.sum() - 1.0) < 1e-3 or pi.sum() == 0.0
        assert isinstance(sz, float)
        assert sz in (-1.0, 0.0, 1.0)


def test_play_game_pcr_zero_p_full_records_nothing():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8,
                   max_plies=20)
    net = AlphaZeroGoNet(cfg).to(torch.device("cpu"))
    net.eval()
    samples, _, ply, stats = play_game_pcr(
        net, cfg, torch.device("cpu"),
        sims_full=2, sims_reduced=2, p_full=0.0, max_moves=20,
    )
    assert ply > 0
    assert stats["full_moves"] == 0
    assert stats["reduced_moves"] >= 1
    assert len(samples) == 0
