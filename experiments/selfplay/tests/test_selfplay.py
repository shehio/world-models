"""Self-play one short game; verify samples and outcome are consistent.

Run with: uv run python tests/test_selfplay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace

import torch

from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet
from selfplay.selfplay import play_game


def test_play_game_emits_consistent_samples():
    # Tight time budget: 5 sims, max 50 plies.
    cfg = replace(Config(), sims_train=5, max_plies=50)
    net = AlphaZeroNet(cfg).cpu()
    net.eval()
    samples, z, ply = play_game(net, cfg, torch.device("cpu"), sims=5)

    assert ply > 0, "no moves were played"
    assert len(samples) == ply, (len(samples), ply)
    assert z in (-1.0, 0.0, 1.0), z

    for state, pi, z_pov in samples:
        assert state.shape == (19, 8, 8)
        assert pi.shape == (4672,)
        assert abs(pi.sum() - 1.0) < 1e-5 or pi.sum() == 0.0
        assert z_pov in (-1.0, 0.0, 1.0)

    # The sign relationship: samples alternate between white-to-move and
    # black-to-move plies. White-to-move's z_pov should equal z; black-to-
    # move's should equal -z. Index 0 is start position (white to move).
    assert samples[0][2] == z
    if len(samples) > 1:
        assert samples[1][2] == -z


def main():
    print("running test_play_game_emits_consistent_samples ...")
    test_play_game_emits_consistent_samples()
    print("ALL OK")


if __name__ == "__main__":
    main()
