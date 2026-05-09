"""Playout Cap Randomization tests.

Run with: uv run python tests/test_pcr.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace
import numpy as np
import torch

from alphazero.config import Config
from alphazero.network import AlphaZeroNet
from alphazero.selfplay import play_game_pcr


def _make_net(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Config()
    net = AlphaZeroNet(cfg).cpu()
    net.eval()
    return cfg, net, torch.device("cpu")


def test_pcr_returns_valid_structure():
    cfg, net, dev = _make_net()
    cfg = replace(cfg, max_plies=40)
    samples, z, ply, stats = play_game_pcr(
        net, cfg, dev,
        sims_full=10, sims_reduced=4, p_full=0.5, batch_size=8,
    )
    assert ply > 0
    assert z in (-1.0, 0.0, 1.0)
    assert stats["full_moves"] + stats["reduced_moves"] == ply
    assert stats["recorded_targets"] == len(samples)
    assert stats["recorded_targets"] == stats["full_moves"]
    for state, pi, z_pov in samples:
        assert state.shape == (19, 8, 8)
        assert pi.shape == (4672,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert z_pov in (-1.0, 0.0, 1.0)


def test_pcr_p_full_zero_records_nothing():
    """With p_full=0, no positions should be recorded — game still completes."""
    cfg, net, dev = _make_net()
    cfg = replace(cfg, max_plies=30)
    samples, z, ply, stats = play_game_pcr(
        net, cfg, dev,
        sims_full=10, sims_reduced=4, p_full=0.0, batch_size=8,
    )
    assert stats["full_moves"] == 0
    assert stats["reduced_moves"] == ply
    assert len(samples) == 0


def test_pcr_p_full_one_records_everything():
    """With p_full=1, every move is a recorded position."""
    cfg, net, dev = _make_net()
    cfg = replace(cfg, max_plies=20)
    samples, z, ply, stats = play_game_pcr(
        net, cfg, dev,
        sims_full=10, sims_reduced=4, p_full=1.0, batch_size=8,
    )
    assert stats["reduced_moves"] == 0
    assert stats["full_moves"] == ply
    assert len(samples) == ply


def test_pcr_proportions_match_p_full():
    """Average over many games: recorded fraction ≈ p_full."""
    cfg, net, dev = _make_net()
    cfg = replace(cfg, max_plies=30)
    random.seed(0)
    np.random.seed(0)
    total_full = total_plies = 0
    for _ in range(8):
        _, _, ply, stats = play_game_pcr(
            net, cfg, dev,
            sims_full=8, sims_reduced=2, p_full=0.5, batch_size=8,
        )
        total_full += stats["full_moves"]
        total_plies += ply
    frac = total_full / total_plies
    assert 0.35 < frac < 0.65, frac  # wide window, just checking not 0 or 1


def main():
    tests = [
        test_pcr_returns_valid_structure,
        test_pcr_p_full_zero_records_nothing,
        test_pcr_p_full_one_records_everything,
        test_pcr_proportions_match_p_full,
    ]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
