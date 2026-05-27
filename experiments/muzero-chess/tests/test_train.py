"""Smoke tests for the K-step unrolled training loss."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import MuZeroConfig, INPUT_PLANES, ACTION_DIM
from muzero_chess.networks import MuZeroNet
from muzero_chess.train import train_step


def _tiny_cfg() -> MuZeroConfig:
    return MuZeroConfig(
        latent_channels=8,
        repr_n_res_blocks=1, repr_n_filters=8,
        dyn_n_res_blocks=1, dyn_n_filters=8,
        pred_n_res_blocks=1, pred_n_filters=8,
        num_unroll_steps=3,
    )


def _make_batch(B: int, K: int, device) -> dict:
    """One batch of K-step training trajectories with random labels."""
    pi_logits = torch.randn(B, K + 1, ACTION_DIM, device=device)
    pi_targets = torch.softmax(pi_logits, dim=-1)
    return {
        "observation": torch.randn(B, INPUT_PLANES, 8, 8, device=device),
        "actions": torch.randint(0, ACTION_DIM, (B, K), device=device),
        "pi_targets": pi_targets,
        "z_targets": torch.empty(B, K + 1, device=device).uniform_(-1, 1),
        "r_targets": torch.zeros(B, K, device=device),    # chess has 0 reward at non-terminal
    }


def test_train_step_runs_and_returns_loss_components():
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = _make_batch(B=4, K=3, device=torch.device("cpu"))
    stats = train_step(net, opt, batch)
    assert isinstance(stats["loss"], float)
    assert isinstance(stats["policy_loss"], float)
    assert isinstance(stats["value_loss"], float)
    assert isinstance(stats["reward_loss"], float)
    assert stats["loss"] > 0


def test_train_step_actually_updates_all_three_networks():
    """Critical correctness check: a single SGD step should update params
    in h, g, AND f. If gradients only flow through f (e.g. if the unroll
    detaches the latent), we'd see h + g unchanged."""
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    opt = torch.optim.SGD(net.parameters(), lr=1e-1)
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    batch = _make_batch(B=4, K=3, device=torch.device("cpu"))
    train_step(net, opt, batch)

    h_changed = any(
        not torch.allclose(before[n], p)
        for n, p in net.named_parameters() if n.startswith("h.")
    )
    g_changed = any(
        not torch.allclose(before[n], p)
        for n, p in net.named_parameters() if n.startswith("g.")
    )
    f_changed = any(
        not torch.allclose(before[n], p)
        for n, p in net.named_parameters() if n.startswith("f.")
    )
    assert h_changed, "h_θ did not receive gradient — representation isn't training"
    assert g_changed, "g_θ did not receive gradient — dynamics isn't training"
    assert f_changed, "f_θ did not receive gradient — prediction isn't training"


def test_train_step_reduces_loss_on_overfit_batch():
    """Train repeatedly on ONE batch — loss should drop measurably.

    Policy loss is over 4,672 actions, so it dominates and is slow to drop;
    we just require the total loss to go DOWN, not how fast. The
    "no-gradient flow" failure mode would have loss flat or rising."""
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch = _make_batch(B=4, K=2, device=torch.device("cpu"))
    initial_loss = train_step(net, opt, batch)["loss"]
    for _ in range(100):
        stats = train_step(net, opt, batch)
    final_loss = stats["loss"]
    # Loss must drop strictly. 5% is more than enough to rule out flat
    # gradients while not requiring a fully-converged overfit.
    assert final_loss < initial_loss * 0.95, (
        f"loss didn't drop on a single-batch overfit: {initial_loss} → {final_loss}"
    )


def test_value_targets_at_extremes_compute_finite_loss():
    """z=±1 is the natural target at game end. Make sure loss is well-defined."""
    cfg = _tiny_cfg()
    net = MuZeroNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = _make_batch(B=4, K=2, device=torch.device("cpu"))
    batch["z_targets"] = torch.tensor([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    batch["r_targets"] = torch.tensor([
        [0.0, 1.0],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    stats = train_step(net, opt, batch)
    import math
    for k in ("loss", "policy_loss", "value_loss", "reward_loss"):
        assert math.isfinite(stats[k]), f"{k} is not finite: {stats[k]}"
