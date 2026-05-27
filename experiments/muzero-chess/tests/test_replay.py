"""Tests for the replay buffer + GameRecord."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import ACTION_DIM, INPUT_PLANES, MuZeroConfig
from muzero_chess.replay import GameRecord, ReplayBuffer


def _fake_game(T: int, outcome_for_starter: float = 1.0,
               terminal_reward_for_mover: float = 1.0) -> GameRecord:
    """Synthesize a GameRecord with deterministic actions and uniform π."""
    obs = [np.full((INPUT_PLANES, 8, 8), float(t), dtype=np.float32) for t in range(T)]
    actions = [t * 7 % ACTION_DIM for t in range(T)]
    pi = [np.full(ACTION_DIM, 1.0 / ACTION_DIM, dtype=np.float32) for _ in range(T)]
    return GameRecord.from_trajectory(
        observations=obs,
        actions=actions,
        pis=pi,
        outcome_for_starter=outcome_for_starter,
        terminal_reward_for_mover=terminal_reward_for_mover,
    )


def test_z_targets_flip_sign_by_ply_parity():
    g = _fake_game(T=5, outcome_for_starter=1.0)
    # mover at ply 0 won → z_0 = +1, z_1 = -1, z_2 = +1, z_3 = -1, z_4 = +1
    assert g.z_per_ply == [1.0, -1.0, 1.0, -1.0, 1.0]


def test_z_targets_zero_for_draw():
    g = _fake_game(T=4, outcome_for_starter=0.0, terminal_reward_for_mover=0.0)
    assert g.z_per_ply == [0.0, 0.0, 0.0, 0.0]


def test_terminal_reward_in_last_slot_only():
    g = _fake_game(T=5, terminal_reward_for_mover=1.0)
    # Only the LAST action carries the terminal reward.
    assert g.r_per_action[:-1] == [0.0, 0.0, 0.0, 0.0]
    assert g.r_per_action[-1] == 1.0


def test_buffer_capacity_evicts_oldest():
    cfg = MuZeroConfig()
    buf = ReplayBuffer(capacity=3, cfg=cfg)
    games = [_fake_game(T=2) for _ in range(5)]
    for g in games:
        buf.push(g)
    assert len(buf) == 3
    # The oldest two should be gone; surviving games are the last three pushed.
    surviving_obs0 = [g.observations[0][0, 0, 0] for g in buf._games]
    # All games have obs[t][0,0,0] = t, but we want to check identity:
    assert buf._games[0] is games[2]
    assert buf._games[-1] is games[-1]


def test_sample_shapes_match_train_step_spec():
    cfg = MuZeroConfig()
    buf = ReplayBuffer(capacity=10, cfg=cfg)
    for _ in range(3):
        buf.push(_fake_game(T=6))
    batch = buf.sample(batch_size=4, K=3)
    assert batch["observation"].shape == (4, INPUT_PLANES, 8, 8)
    assert batch["actions"].shape == (4, 3)
    assert batch["pi_targets"].shape == (4, 4, ACTION_DIM)
    assert batch["z_targets"].shape == (4, 4)
    assert batch["r_targets"].shape == (4, 3)
    # All values finite, actions in-range.
    import torch
    assert torch.isfinite(batch["pi_targets"]).all()
    assert torch.isfinite(batch["z_targets"]).all()
    assert torch.isfinite(batch["r_targets"]).all()
    assert (batch["actions"] >= 0).all() and (batch["actions"] < ACTION_DIM).all()


def test_past_terminal_padding_uses_absorbing_values():
    """Sample with K large enough to spill past the end of the game and verify
    that pi-rows are all-zero, z=0, r=0 at those positions. (Random actions
    are fine — we only need to confirm pi/z/r are absorbing.)"""
    import random as pyrandom
    cfg = MuZeroConfig()
    buf = ReplayBuffer(capacity=2, cfg=cfg)
    buf.push(_fake_game(T=3, outcome_for_starter=1.0, terminal_reward_for_mover=1.0))

    # Force start position to be the last ply so the K-step window spills off.
    rng = pyrandom.Random(0)
    # Sample many to find one that started at ply 2 (the last ply).
    found_spill = False
    for _ in range(50):
        batch = buf.sample(batch_size=1, K=4, rng=rng)
        # If pi at the last positions is all-zero, we know we spilled.
        last_pi_sum = batch["pi_targets"][0, -1].sum().item()
        if last_pi_sum == 0.0:
            assert batch["z_targets"][0, -1].item() == 0.0
            assert batch["r_targets"][0, -1].item() == 0.0
            found_spill = True
            break
    assert found_spill, "Could not find a sample that spilled past terminal"


def test_n_positions_sums_game_lengths():
    cfg = MuZeroConfig()
    buf = ReplayBuffer(capacity=10, cfg=cfg)
    buf.push(_fake_game(T=3))
    buf.push(_fake_game(T=5))
    buf.push(_fake_game(T=2))
    assert buf.n_positions == 10
