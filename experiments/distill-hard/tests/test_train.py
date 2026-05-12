"""Tests for supervised training step.

Run with: uv run python tests/test_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace
import numpy as np
import torch

from wm_chess.board import N_INPUT_PLANES, N_POLICY
from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet
from distill_hard.train_supervised import train_step


def _toy_batch(n=8):
    states = np.random.randn(n, N_INPUT_PLANES, 8, 8).astype(np.float32)
    moves = np.random.randint(0, N_POLICY, size=n).astype(np.int64)
    zs = (2 * np.random.randint(0, 2, size=n) - 1).astype(np.float32)  # ±1
    return states, moves, zs


def test_train_step_returns_finite_losses():
    cfg = replace(Config(), n_res_blocks=2, n_filters=16)  # tiny for speed
    net = AlphaZeroNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = train_step(net, opt, _toy_batch(), torch.device("cpu"))
    assert all(v == v for v in losses.values()), losses  # not NaN
    assert losses["loss"] > 0
    assert 0 <= losses["top1_acc"] <= 1


def test_train_step_decreases_loss_on_repeated_data():
    """Memorization sanity: loss should drop after several updates on the same batch."""
    cfg = replace(Config(), n_res_blocks=2, n_filters=16)
    net = AlphaZeroNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    batch = _toy_batch(n=16)
    initial = train_step(net, opt, batch, torch.device("cpu"))["loss"]
    for _ in range(20):
        train_step(net, opt, batch, torch.device("cpu"))
    final = train_step(net, opt, batch, torch.device("cpu"))["loss"]
    assert final < initial * 0.9, (initial, final)


def test_top1_acc_is_perfect_after_overfitting_one_sample():
    """If we train hard on a single (state, move) pair, top1 should hit 1.0."""
    cfg = replace(Config(), n_res_blocks=2, n_filters=16)
    net = AlphaZeroNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    states = np.random.randn(1, N_INPUT_PLANES, 8, 8).astype(np.float32)
    moves = np.array([42], dtype=np.int64)
    zs = np.array([0.0], dtype=np.float32)
    for _ in range(100):
        train_step(net, opt, (states, moves, zs), torch.device("cpu"))
    final = train_step(net, opt, (states, moves, zs), torch.device("cpu"))
    assert final["top1_acc"] == 1.0


def main():
    tests = [
        test_train_step_returns_finite_losses,
        test_train_step_decreases_loss_on_repeated_data,
        test_top1_acc_is_perfect_after_overfitting_one_sample,
    ]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
