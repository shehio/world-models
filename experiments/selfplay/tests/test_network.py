"""Forward-pass + parameter-count sanity for AlphaZeroNet.

Run with: uv run python tests/test_network.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from wm_chess.board import N_INPUT_PLANES, N_POLICY
from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet


def test_forward_shapes():
    cfg = Config()
    net = AlphaZeroNet(cfg)
    net.eval()
    for batch in (1, 4, 16):
        x = torch.randn(batch, N_INPUT_PLANES, 8, 8)
        p, v = net(x)
        assert p.shape == (batch, N_POLICY), p.shape
        assert v.shape == (batch,), v.shape
        # Values clamped to [-1, 1] by tanh.
        assert v.min().item() >= -1.0 - 1e-6
        assert v.max().item() <= 1.0 + 1e-6


def test_param_count_reasonable():
    """5 res blocks × 64 ch should land in 100k–1M params."""
    cfg = Config()
    net = AlphaZeroNet(cfg)
    n = sum(p.numel() for p in net.parameters())
    assert 100_000 < n < 1_000_000, n
    print(f"  param count: {n:,}")


def test_gradient_flow():
    """A single backward pass: every parameter receives a gradient."""
    cfg = Config()
    net = AlphaZeroNet(cfg)
    net.train()
    x = torch.randn(4, N_INPUT_PLANES, 8, 8)
    pi_target = torch.zeros(4, N_POLICY)
    pi_target[:, 0] = 1.0
    z_target = torch.zeros(4)

    p, v = net(x)
    log_p = torch.log_softmax(p, dim=-1)
    loss = -(pi_target * log_p).sum(dim=-1).mean() + ((v - z_target) ** 2).mean()
    loss.backward()

    no_grad = [name for name, prm in net.named_parameters() if prm.grad is None]
    assert not no_grad, f"params with no grad: {no_grad[:5]}"


def test_eval_train_mode_consistency():
    """eval() with batch=1 must not crash (BatchNorm running stats path)."""
    cfg = Config()
    net = AlphaZeroNet(cfg)
    net.eval()
    x = torch.randn(1, N_INPUT_PLANES, 8, 8)
    with torch.no_grad():
        p, v = net(x)
    assert p.shape == (1, N_POLICY)
    assert v.shape == (1,)


def main():
    tests = [test_forward_shapes, test_param_count_reasonable, test_gradient_flow, test_eval_train_mode_consistency]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
