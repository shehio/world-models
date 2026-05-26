"""Tests for the chess self-play replay buffers.

Both ReplayBuffer (flat deque) and ShardedReplayBuffer (sliding window
of per-iter shards) are critical to the self-play loop. They were
previously untested at this layer — the bug they protect against
(sampling an empty buffer raises ValueError from np.random.randint)
was only guarded at call-site in selfplay_loop_mp.py.

These tests pin down:
  - empty buffer rejects sample() cleanly (instead of cryptic numpy error)
  - basic add → sample round-trip returns tensors with correct shapes
  - ShardedReplayBuffer evicts the oldest shard when at capacity
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfplay.replay import ReplayBuffer, ShardedReplayBuffer


# 19-plane chess board, 4672-dim policy — match the real selfplay shapes.
STATE_SHAPE = (19, 8, 8)
PI_DIM = 4672


def _fake_sample(seed: int):
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(STATE_SHAPE).astype(np.float32)
    pi = rng.dirichlet(np.ones(PI_DIM)).astype(np.float32)
    z = float(rng.choice([-1.0, 0.0, 1.0]))
    return (state, pi, z)


# ---- ReplayBuffer ----

def test_flat_buffer_starts_empty():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0


def test_flat_buffer_sample_on_empty_raises_clearly():
    buf = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError, match="empty buffer"):
        buf.sample(batch_size=4, device=torch.device("cpu"))


def test_flat_buffer_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    buf.add_game([_fake_sample(i) for i in range(20)])
    s, p, z = buf.sample(batch_size=8, device=torch.device("cpu"))
    assert s.shape == (8, *STATE_SHAPE)
    assert p.shape == (8, PI_DIM)
    assert z.shape == (8,)


def test_flat_buffer_capacity_evicts_oldest():
    buf = ReplayBuffer(capacity=5)
    # Add 10 distinct samples; only last 5 should remain.
    for i in range(10):
        buf.add_game([(np.full(STATE_SHAPE, float(i), dtype=np.float32),
                       np.ones(PI_DIM, dtype=np.float32) / PI_DIM,
                       float(i))])
    assert len(buf) == 5
    # Sample heavily and check no value < 5 (the oldest 5 were evicted).
    _, _, zs = buf.sample(batch_size=200, device=torch.device("cpu"))
    assert zs.min().item() >= 5.0


# ---- ShardedReplayBuffer ----

def test_sharded_buffer_sample_on_empty_raises_clearly():
    buf = ShardedReplayBuffer(max_shards=3)
    with pytest.raises(ValueError, match="empty buffer"):
        buf.sample(batch_size=4, device=torch.device("cpu"))


def test_sharded_buffer_window_evicts_oldest_shard():
    buf = ShardedReplayBuffer(max_shards=2)
    buf.add_iteration([_fake_sample(0) for _ in range(3)])
    buf.add_iteration([_fake_sample(1) for _ in range(5)])
    buf.add_iteration([_fake_sample(2) for _ in range(7)])
    assert buf.n_shards == 2
    assert len(buf) == 12   # 5 + 7; the 3 from iter 0 evicted


def test_sharded_buffer_sample_pools_across_shards():
    """If any sample function only picked from one shard, this catches it."""
    buf = ShardedReplayBuffer(max_shards=5)
    for shard_id in range(3):
        buf.add_iteration([
            (np.zeros(STATE_SHAPE, dtype=np.float32),
             np.ones(PI_DIM, dtype=np.float32) / PI_DIM,
             float(shard_id))
            for _ in range(50)
        ])
    _, _, zs = buf.sample(batch_size=300, device=torch.device("cpu"))
    assert set(zs.tolist()) == {0.0, 1.0, 2.0}


def test_add_game_on_sharded_creates_one_shard_per_call():
    buf = ShardedReplayBuffer(max_shards=10)
    buf.add_game([_fake_sample(0)])
    buf.add_game([_fake_sample(1)])
    assert buf.n_shards == 2
    assert len(buf) == 2
