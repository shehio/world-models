"""Tests for the ResNet network — forward shapes, value range, device map."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm_chess.board import N_INPUT_PLANES, N_POLICY
from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet, ResidualBlock, get_device


CPU = torch.device("cpu")


def _tiny_cfg():
    return replace(Config(), n_res_blocks=1, n_filters=8)


class TestNetworkForward:
    def test_output_shapes(self):
        net = AlphaZeroNet(_tiny_cfg())
        net.eval()
        x = torch.zeros(2, N_INPUT_PLANES, 8, 8)
        logits, value = net(x)
        assert logits.shape == (2, N_POLICY)
        assert value.shape == (2,)

    def test_value_in_tanh_range(self):
        net = AlphaZeroNet(_tiny_cfg())
        net.eval()
        x = torch.randn(8, N_INPUT_PLANES, 8, 8)
        _, value = net(x)
        # tanh bounds.
        assert (value >= -1.0).all() and (value <= 1.0).all()

    def test_batchnorm_train_vs_eval(self):
        """BN stats differ between train and eval mode — both should at
        least run without error and produce finite outputs."""
        net = AlphaZeroNet(_tiny_cfg())
        x = torch.randn(4, N_INPUT_PLANES, 8, 8)

        net.train()
        l1, v1 = net(x)
        assert torch.isfinite(l1).all() and torch.isfinite(v1).all()

        net.eval()
        l2, v2 = net(x)
        assert torch.isfinite(l2).all() and torch.isfinite(v2).all()

    def test_scales_with_filters(self):
        """Doubling n_filters more than doubles param count (Conv kernels
        scale ~quadratically). Sanity check the architecture is wired
        through Config."""
        n1 = sum(p.numel() for p in
                 AlphaZeroNet(replace(Config(), n_res_blocks=2, n_filters=16)).parameters())
        n2 = sum(p.numel() for p in
                 AlphaZeroNet(replace(Config(), n_res_blocks=2, n_filters=32)).parameters())
        assert n2 > 2 * n1


class TestResidualBlock:
    def test_preserves_shape(self):
        block = ResidualBlock(channels=16)
        block.eval()
        x = torch.randn(3, 16, 8, 8)
        y = block(x)
        assert y.shape == x.shape


class TestGetDevice:
    def test_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)
        assert d.type in {"cpu", "cuda", "mps"}
