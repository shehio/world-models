"""Tests for the supervised training surface — the soft/hard target loss
and the sparse→dense scatter that makes it possible.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm_chess.board import N_INPUT_PLANES, N_POLICY
from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet
from distill_soft.train_supervised import (
    MultipvDataset,
    _scatter_sparse_target_to_dense,
    train_step,
)


CPU = torch.device("cpu")


class TestScatter:
    def test_basic_normalized(self):
        # Two positions, K=3. p = [0.5, 0.3, 0.2] on indices [10, 20, 30].
        logp = torch.log(torch.tensor([[0.5, 0.3, 0.2],
                                       [0.6, 0.3, 0.1]]))
        idx = torch.tensor([[10, 20, 30],
                            [ 5, 15, 25]])
        dense = _scatter_sparse_target_to_dense(idx, logp, n_actions=64,
                                                 device=CPU)
        assert dense.shape == (2, 64)
        assert torch.allclose(dense.sum(dim=-1), torch.tensor([1.0, 1.0]),
                              atol=1e-5)
        assert torch.isclose(dense[0, 10], torch.tensor(0.5))
        assert torch.isclose(dense[1, 5], torch.tensor(0.6))

    def test_padding_zeros(self):
        # K=4 with padding in last two slots (-1 index, -inf logprob)
        logp = torch.tensor([[math.log(0.7), math.log(0.3), -math.inf, -math.inf]])
        idx = torch.tensor([[3, 7, -1, -1]])
        dense = _scatter_sparse_target_to_dense(idx, logp, n_actions=16,
                                                 device=CPU)
        assert torch.isclose(dense.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(dense[0, 3], torch.tensor(0.7), atol=1e-5)
        assert torch.isclose(dense[0, 7], torch.tensor(0.3), atol=1e-5)
        # Index 0 is the dummy fill for padded slots — must be zero.
        assert dense[0, 0].item() == 0.0

    def test_all_padding(self):
        # Defensive: a position with zero valid PVs (shouldn't happen in
        # practice but we must not crash or produce NaN).
        logp = torch.full((1, 4), -math.inf)
        idx = torch.tensor([[-1, -1, -1, -1]])
        dense = _scatter_sparse_target_to_dense(idx, logp, n_actions=8,
                                                 device=CPU)
        assert torch.isfinite(dense).all()
        assert dense.sum().item() == 0.0  # nothing to distribute


class TestMultipvDataset:
    def test_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "fake.npz")
        N, K = 5, 4
        np.savez_compressed(
            path,
            states=np.zeros((N, N_INPUT_PLANES, 8, 8), dtype=np.float32),
            moves=np.arange(N, dtype=np.int64),
            multipv_indices=np.zeros((N, K), dtype=np.int64),
            multipv_logprobs=np.zeros((N, K), dtype=np.float32),
            zs=np.zeros(N, dtype=np.float32),
            K=np.array(K, dtype=np.int32),
        )
        ds = MultipvDataset(path)
        assert len(ds) == N
        assert ds.K == K
        s, mpv_idx, mpv_logp, mv, z = ds[2]
        assert s.shape == (N_INPUT_PLANES, 8, 8)
        assert int(mv) == 2

    def test_K_inferred_when_missing(self, tmp_path):
        """Legacy NPZ files without an explicit K field — K should be
        inferred from multipv_indices.shape[1]."""
        path = str(tmp_path / "legacy.npz")
        np.savez_compressed(
            path,
            states=np.zeros((1, N_INPUT_PLANES, 8, 8), dtype=np.float32),
            moves=np.zeros(1, dtype=np.int64),
            multipv_indices=np.zeros((1, 6), dtype=np.int64),
            multipv_logprobs=np.zeros((1, 6), dtype=np.float32),
            zs=np.zeros(1, dtype=np.float32),
        )
        assert MultipvDataset(path).K == 6


class TestTrainStep:
    """Smoke-level integration: build a tiny net, run one train step on
    a fake batch, verify it doesn't crash and the loss is finite.

    Uses a 1-block / 8-channel net so this stays well under 1s on CPU.
    """

    def _tiny_net_and_batch(self, hard_targets: bool, K: int = 4, B: int = 3):
        cfg = Config()
        # Override to a tiny net.
        from dataclasses import replace
        cfg = replace(cfg, n_res_blocks=1, n_filters=8)
        net = AlphaZeroNet(cfg).to(CPU)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        states = np.random.randn(B, N_INPUT_PLANES, 8, 8).astype(np.float32)
        mpv_idx = np.random.randint(0, N_POLICY, size=(B, K)).astype(np.int64)
        # Build valid logprobs (uniform over K) → sums to 1.
        mpv_logp = np.full((B, K), math.log(1.0 / K), dtype=np.float32)
        moves = mpv_idx[:, 0]  # played move = top-1 candidate
        zs = np.zeros(B, dtype=np.float32)
        return net, opt, (states, mpv_idx, mpv_logp, moves, zs)

    def test_soft_targets_loss_finite(self):
        net, opt, batch = self._tiny_net_and_batch(hard_targets=False)
        out = train_step(net, opt, batch, CPU, hard_targets=False)
        assert math.isfinite(out["loss"])
        assert math.isfinite(out["policy_loss"])
        assert math.isfinite(out["value_loss"])
        assert 0.0 <= out["top1_acc"] <= 1.0
        assert 0.0 <= out["topk_acc"] <= 1.0
        # Soft target entropy with uniform-over-K=4 should be ≈ ln(4) ≈ 1.386
        assert abs(out["tgt_entropy"] - math.log(4)) < 0.05

    def test_hard_targets_loss_finite(self):
        net, opt, batch = self._tiny_net_and_batch(hard_targets=True)
        out = train_step(net, opt, batch, CPU, hard_targets=True)
        assert math.isfinite(out["loss"])
        # Hard targets → target dist is one-hot → entropy 0.
        assert out["tgt_entropy"] < 1e-5

    def test_one_step_changes_weights(self):
        """Gradient actually flows: weights move after a single step."""
        net, opt, batch = self._tiny_net_and_batch(hard_targets=False)
        before = next(net.parameters()).detach().clone()
        train_step(net, opt, batch, CPU, hard_targets=False)
        after = next(net.parameters()).detach()
        # Some parameter has changed.
        assert not torch.allclose(before, after)
