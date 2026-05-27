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
    S = cfg.board_size
    for sample in samples:
        # New 6-tuple format: (state, pi, z, ownership, score, opp_pi)
        assert len(sample) == 6
        state, pi, sz, ownership, score, opp_pi = sample
        assert state.shape == (cfg.n_input_planes, S, S)
        assert state.dtype == np.float32
        assert pi.shape == (expected_policy_dim,)
        assert pi.dtype == np.float32
        assert abs(pi.sum() - 1.0) < 1e-3 or pi.sum() == 0.0
        assert isinstance(sz, float)
        assert sz in (-1.0, 0.0, 1.0)
        # Aux targets always present, even if game cut off:
        assert ownership.shape == (S, S)
        assert ownership.dtype == np.int8
        assert ((ownership == 0) | (ownership == 1) | (ownership == 2)).all()
        assert isinstance(score, float)
        assert opp_pi.shape == (expected_policy_dim,)
        assert opp_pi.dtype == np.float32
        # opp_pi is either all-zeros (no opp move) or sums to 1 (one-hot).
        assert opp_pi.sum() == 0.0 or abs(opp_pi.sum() - 1.0) < 1e-6


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
    """Buffer's sample() returns a dict of stacked tensors.

    Tests backward-compat: 3-tuple samples (no aux) get auto-upcast to
    6-tuples with zero aux fields, so old shards in a buffer still work.
    """
    buf = ShardedReplayBuffer(max_shards=2)
    buf.add_iteration([_fake_sample(i) for i in range(20)])
    device = torch.device("cpu")
    batch = buf.sample(batch_size=4, device=device)
    assert isinstance(batch, dict)
    assert batch["state"].shape == (4, 4, 5, 5)
    assert batch["pi"].shape == (4, 26)
    assert batch["z"].shape == (4,)
    assert batch["state"].device.type == "cpu"
    assert batch["pi"].dtype == torch.float32
    assert batch["z"].dtype == torch.float32
    # Aux fields auto-promoted to zeros for 3-tuple samples.
    assert batch["ownership"].shape == (4, 5, 5)
    assert batch["score"].shape == (4,)
    assert batch["opp_pi"].shape == (4, 26)
    assert (batch["ownership"] == 0).all()
    assert (batch["score"] == 0).all()
    assert (batch["opp_pi"] == 0).all()


def test_replay_buffer_sample_on_empty_raises_clearly():
    """np.random.randint(0, 0, size=N) gives a cryptic numpy error; our guard
    raises a ValueError that names the precondition the caller must satisfy."""
    import pytest
    buf = ShardedReplayBuffer(max_shards=3)
    with pytest.raises(ValueError, match="empty buffer"):
        buf.sample(batch_size=4, device=torch.device("cpu"))


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
    batch = buf.sample(batch_size=300, device=torch.device("cpu"))
    seen = set(batch["z"].tolist())
    assert seen == {0.0, 1.0, 2.0}, f"sampling missed shards; saw {seen}"


# ----------------------------------------------------------------------------
# train_step
# ----------------------------------------------------------------------------

def _zero_aux_batch(B: int, policy_size: int, S: int, device):
    """Helper: dict batch with zero aux targets — exercises the base path."""
    return {
        "state": torch.randn(B, 4, S, S, device=device),
        "pi": torch.softmax(torch.randn(B, policy_size, device=device), dim=-1),
        "z": torch.zeros(B, device=device),
        "ownership": torch.zeros(B, S, S, dtype=torch.long, device=device),
        "score": torch.zeros(B, device=device),
        "opp_pi": torch.zeros(B, policy_size, device=device),
    }


def test_train_step_runs_and_returns_floats():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8)
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = _zero_aux_batch(B=8, policy_size=cfg.policy_size, S=cfg.board_size, device=device)
    batch["z"] = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0], device=device)
    stats = train_step(net, opt, batch)
    assert isinstance(stats["loss"], float)
    assert isinstance(stats["policy_loss"], float)
    assert isinstance(stats["value_loss"], float)
    assert stats["loss"] > 0
    assert stats["value_loss"] < 5.0


def test_train_step_actually_updates_weights():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8)
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    batch = _zero_aux_batch(B=8, policy_size=cfg.policy_size, S=cfg.board_size, device=device)
    train_step(net, opt, batch)
    n_changed = sum(
        1 for name, p in net.named_parameters()
        if not torch.allclose(before[name], p)
    )
    assert n_changed > 0, "no parameter changed after one SGD step — backprop broken?"


def test_train_step_with_aux_ownership_returns_extra_losses():
    """When use_aux_ownership=True, train_step computes the aux losses
    and includes them in the stats dict."""
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8,
                   use_aux_ownership=True)
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = _zero_aux_batch(B=4, policy_size=cfg.policy_size, S=cfg.board_size, device=device)
    stats = train_step(net, opt, batch)
    assert "ownership_loss" in stats
    assert "score_loss" in stats
    assert stats["ownership_loss"] >= 0
    assert stats["score_loss"] >= 0


def test_train_step_with_aux_opp_policy_runs():
    """use_aux_opp_policy=True path returns the extra opp_policy_loss."""
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8,
                   use_aux_opp_policy=True)
    device = torch.device("cpu")
    net = AlphaZeroGoNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = _zero_aux_batch(B=4, policy_size=cfg.policy_size, S=cfg.board_size, device=device)
    # Need at least one non-zero opp_pi to exercise the loss path.
    batch["opp_pi"][0, 5] = 1.0
    stats = train_step(net, opt, batch)
    assert "opp_policy_loss" in stats
    assert stats["opp_policy_loss"] >= 0


def test_aux_heads_load_from_plain_prior_via_strict_false():
    """ownership + opp_policy heads only ADD modules; the trunk stays the
    same. So a plain-prior state_dict loads into an aux-enabled net via
    strict=False — the new heads start randomly, the rest inherits."""
    cfg_plain = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1,
                          n_filters=8)
    cfg_aux = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1,
                        n_filters=8,
                        use_aux_ownership=True, use_aux_opp_policy=True)
    net_plain = AlphaZeroGoNet(cfg_plain)
    net_aux = AlphaZeroGoNet(cfg_aux)
    # strict=False should accept the plain state_dict.
    missing, unexpected = net_aux.load_state_dict(net_plain.state_dict(), strict=False)
    # The new aux heads are the missing keys; nothing should be unexpected.
    assert len(unexpected) == 0
    assert any("ownership" in m or "score" in m or "opp_policy" in m for m in missing)


def test_global_pool_changes_trunk_shape_so_cannot_load_from_plain():
    """use_global_pool=True changes conv shapes in the trunk. Even
    strict=False cannot bridge shape mismatches — pytorch raises on them.
    Document this so users know a global-pool run needs to retrain
    distillation rather than warm-start from the existing prior."""
    cfg_plain = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1,
                          n_filters=8, use_global_pool=False)
    cfg_pool = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1,
                         n_filters=8, use_global_pool=True)
    net_plain = AlphaZeroGoNet(cfg_plain)
    net_pool = AlphaZeroGoNet(cfg_pool)
    # Parameter count differs — different trunk modules.
    assert (sum(p.numel() for p in net_plain.parameters())
            != sum(p.numel() for p in net_pool.parameters()))
    # Loading should raise due to shape mismatch even with strict=False.
    import pytest
    with pytest.raises(RuntimeError, match="size mismatch"):
        net_pool.load_state_dict(net_plain.state_dict(), strict=False)


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
