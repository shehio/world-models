"""Tests for the Go self-play loop's exposed building blocks.

The full multi-process loop is exercised by a manual smoke run on real
infra. These tests cover the bits we can run in <30s in CI:
  - play_game_pcr: shape/dtype/PCR-ratio invariants
  - ShardedReplayBuffer: per-iter shards + sliding window + sampling
  - train_step: one SGD step actually moves the loss
  - eval_vs_random: terminates, returns sane stats, doesn't crash on
    an empty-visits / pass-only edge case
  - _outcome_black_pov: returns +1/-1/0 for all three outcome types
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
from selfplay_loop import (
    ShardedReplayBuffer,
    _outcome_black_pov,
    eval_vs_random,
    play_game_pcr,
    train_step,
)
from distill_go.board import BLACK, WHITE, GoBoard


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


# ----------------------------------------------------------------------------
# _outcome_black_pov
# ----------------------------------------------------------------------------

class _FakeBoard:
    """Stub GoBoard that returns a fixed winner() value — used only here."""
    def __init__(self, winner_val):
        self._w = winner_val
    def winner(self):
        return self._w


def test_outcome_black_pov_returns_plus1_for_black_win():
    assert _outcome_black_pov(_FakeBoard(BLACK)) == 1.0


def test_outcome_black_pov_returns_minus1_for_white_win():
    assert _outcome_black_pov(_FakeBoard(WHITE)) == -1.0


def test_outcome_black_pov_returns_zero_for_no_winner():
    assert _outcome_black_pov(_FakeBoard(None)) == 0.0


# ----------------------------------------------------------------------------
# ShardedReplayBuffer
# ----------------------------------------------------------------------------

def _fake_sample(seed: int, state_shape=(4, 5, 5), pi_dim=26):
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(state_shape).astype(np.float32)
    pi = rng.dirichlet(np.ones(pi_dim)).astype(np.float32)
    z = float(rng.choice([-1.0, 0.0, 1.0]))
    return (state, pi, z)


def test_replay_buffer_starts_empty():
    buf = ShardedReplayBuffer(max_shards=3)
    assert len(buf) == 0
    assert buf.n_shards == 0


def test_replay_buffer_add_iteration_grows_correctly():
    buf = ShardedReplayBuffer(max_shards=3)
    buf.add_iteration([_fake_sample(i) for i in range(5)])
    assert buf.n_shards == 1
    assert len(buf) == 5
    buf.add_iteration([_fake_sample(i + 100) for i in range(7)])
    assert buf.n_shards == 2
    assert len(buf) == 12


def test_replay_buffer_window_evicts_oldest_shard():
    buf = ShardedReplayBuffer(max_shards=2)
    buf.add_iteration([_fake_sample(0) for _ in range(3)])
    buf.add_iteration([_fake_sample(1) for _ in range(5)])
    buf.add_iteration([_fake_sample(2) for _ in range(7)])
    # The first shard (3 samples) should have been evicted; only 5+7=12 remain.
    assert buf.n_shards == 2
    assert len(buf) == 12


def test_replay_buffer_sample_returns_tensors_on_device():
    buf = ShardedReplayBuffer(max_shards=2)
    buf.add_iteration([_fake_sample(i) for i in range(20)])
    device = torch.device("cpu")
    states, pis, zs = buf.sample(batch_size=4, device=device)
    assert states.shape == (4, 4, 5, 5)
    assert pis.shape == (4, 26)
    assert zs.shape == (4,)
    assert states.device.type == "cpu"
    assert pis.dtype == torch.float32
    assert zs.dtype == torch.float32


def test_replay_buffer_sample_uses_all_shards():
    """Sampling should pool across shards, not just one."""
    buf = ShardedReplayBuffer(max_shards=5)
    # Make z values per shard distinguishable.
    for shard_id in range(3):
        buf.add_iteration([
            (np.zeros((4, 5, 5), dtype=np.float32),
             np.ones(26, dtype=np.float32) / 26,
             float(shard_id))
            for _ in range(50)
        ])
    _, _, zs = buf.sample(batch_size=300, device=torch.device("cpu"))
    seen = set(zs.tolist())
    assert seen == {0.0, 1.0, 2.0}, f"sampling missed shards; saw {seen}"


# ----------------------------------------------------------------------------
# train_step
# ----------------------------------------------------------------------------

def test_train_step_runs_and_returns_floats():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8)
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    states = torch.randn(8, 4, 5, 5, device=device)
    pis = torch.softmax(torch.randn(8, cfg.policy_size, device=device), dim=-1)
    zs = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0], device=device)
    stats = train_step(net, opt, (states, pis, zs))
    assert isinstance(stats["loss"], float)
    assert isinstance(stats["policy_loss"], float)
    assert isinstance(stats["value_loss"], float)
    assert stats["loss"] > 0
    # value_loss should be in a sensible range for tanh outputs vs [-1, 1] targets
    assert stats["value_loss"] < 5.0


def test_train_step_actually_updates_weights():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8)
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    states = torch.randn(8, 4, 5, 5, device=device)
    pis = torch.softmax(torch.randn(8, cfg.policy_size, device=device), dim=-1)
    zs = torch.zeros(8, device=device)
    train_step(net, opt, (states, pis, zs))
    n_changed = sum(
        1 for name, p in net.named_parameters()
        if not torch.allclose(before[name], p)
    )
    assert n_changed > 0, "no parameter changed after one SGD step — backprop broken?"


# ----------------------------------------------------------------------------
# eval_vs_random
# ----------------------------------------------------------------------------

def test_eval_vs_random_basic_run_terminates_with_sane_stats():
    cfg = GoConfig(
        board_size=5,
        n_input_planes=4,
        n_res_blocks=1,
        n_filters=8,
        max_plies=30,
    )
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    net.eval()
    r = eval_vs_random(net, cfg, device, games=2, sims=4)
    assert r["games"] == 2
    assert r["wins"] + r["draws"] + r["losses"] == 2
    assert 0.0 <= r["score"] <= 1.0


def test_eval_vs_random_zero_games_returns_zero_score():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8)
    net = AlphaZeroGoNet(cfg).to(torch.device("cpu"))
    net.eval()
    r = eval_vs_random(net, cfg, torch.device("cpu"), games=0, sims=1)
    assert r["games"] == 0
    assert r["wins"] == 0
    assert r["score"] == 0.0
