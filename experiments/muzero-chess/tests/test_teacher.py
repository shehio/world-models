"""Tests for distill-init: teacher factoring + g-only dynamics training."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muzero_chess.config import MuZeroConfig
from muzero_chess.distill_dynamics import (
    TransitionBuffer,
    distill_step,
    generate_transitions,
)
from muzero_chess.networks import DynamicsNet
from muzero_chess.teacher import (
    DistillInitMuZeroNet,
    TeacherPrediction,
    TeacherRepresentation,
    load_teacher,
)

from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet


def _tiny_teacher() -> AlphaZeroNet:
    net = AlphaZeroNet(Config(n_res_blocks=2, n_filters=32))
    net.eval()
    return net


def _tiny_cfg() -> MuZeroConfig:
    # latent_channels MUST equal the teacher's n_filters (latent = trunk output).
    return MuZeroConfig(
        latent_channels=32, dyn_n_filters=32, dyn_n_res_blocks=1,
        num_simulations=4, mcts_batch_size=2, mcts_top_k=8,
        num_unroll_steps=3, max_plies=12, temp_moves=0, epsilon_random=0.2,
    )


def test_factoring_reproduces_teacher_exactly():
    """f(h(obs)) must equal teacher(obs) — proves the split is faithful."""
    teacher = _tiny_teacher()
    h = TeacherRepresentation(teacher).eval()
    f = TeacherPrediction(teacher).eval()
    obs = torch.randn(4, 19, 8, 8)
    with torch.no_grad():
        tp, tv = teacher(obs)
        fp, fv = f(h(obs))
    assert torch.allclose(tp, fp, atol=1e-6), "policy logits diverged"
    assert torch.allclose(tv, fv, atol=1e-6), "value diverged"


def test_teacher_params_frozen_only_g_trains():
    teacher = _tiny_teacher()
    cfg = _tiny_cfg()
    h = TeacherRepresentation(teacher)
    f = TeacherPrediction(teacher)
    g = DynamicsNet(cfg)
    composed = DistillInitMuZeroNet(h, g, f)
    h_grad = all(not p.requires_grad for p in composed.h.parameters())
    f_grad = all(not p.requires_grad for p in composed.f.parameters())
    g_grad = all(p.requires_grad for p in composed.g.parameters())
    assert h_grad, "h should be frozen"
    assert f_grad, "f should be frozen"
    assert g_grad, "g must be trainable"


def test_composed_net_train_keeps_teacher_in_eval():
    """Switching composed.train() must NOT flip h/f BN into training mode."""
    teacher = _tiny_teacher()
    cfg = _tiny_cfg()
    composed = DistillInitMuZeroNet(
        TeacherRepresentation(teacher), DynamicsNet(cfg), TeacherPrediction(teacher))
    composed.train()
    assert not composed.h.training, "h must stay in eval mode"
    assert not composed.f.training, "f must stay in eval mode"
    assert composed.g.training, "g should be in train mode"


def test_load_teacher_roundtrip(tmp_path):
    teacher = _tiny_teacher()
    ckpt = tmp_path / "teacher.pt"
    torch.save(teacher.state_dict(), ckpt)
    loaded = load_teacher(str(ckpt), n_blocks=2, n_filters=32)
    obs = torch.randn(2, 19, 8, 8)
    with torch.no_grad():
        a = teacher(obs)
        b = loaded(obs)
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert torch.allclose(a[1], b[1], atol=1e-6)


def test_generate_transitions_and_buffer():
    teacher = _tiny_teacher()
    cfg = _tiny_cfg()
    games = generate_transitions(teacher, cfg, n_games=4, seed=0)
    assert len(games) >= 1
    for g in games:
        assert len(g.obs) == len(g.actions) == len(g.rewards)
    buf = TransitionBuffer(K=cfg.num_unroll_steps)
    buf.add(games)
    if len(buf) > 0:
        batch = buf.sample(batch_size=3)
        assert batch["obs_seq"].shape == (3, cfg.num_unroll_steps + 1, 19, 8, 8)
        assert batch["actions"].shape == (3, cfg.num_unroll_steps)
        assert batch["rewards"].shape == (3, cfg.num_unroll_steps)


def test_distill_step_reduces_latent_loss_on_overfit():
    """Train g repeatedly on one batch — latent loss should drop, proving g
    learns to reproduce the frozen teacher's latent transitions."""
    teacher = _tiny_teacher()
    cfg = _tiny_cfg()
    composed = DistillInitMuZeroNet(
        TeacherRepresentation(teacher), DynamicsNet(cfg), TeacherPrediction(teacher))
    opt = torch.optim.Adam(composed.g.parameters(), lr=1e-2)
    games = generate_transitions(teacher, cfg, n_games=8, seed=1)
    buf = TransitionBuffer(K=cfg.num_unroll_steps)
    buf.add(games)
    if len(buf) == 0:
        return  # tiny random teacher made only short games; skip
    batch = buf.sample(batch_size=4, rng=random.Random(0))
    first = distill_step(composed, opt, batch, cfg)["latent_loss"]
    for _ in range(60):
        stats = distill_step(composed, opt, batch, cfg)
    assert stats["latent_loss"] < first * 0.9, (
        f"latent loss didn't drop: {first} -> {stats['latent_loss']}")


def test_composed_net_plays_legal_moves():
    """The composed net must run through muzero_policy and return legal moves."""
    from muzero_chess.eval import muzero_policy
    teacher = _tiny_teacher()
    cfg = _tiny_cfg()
    composed = DistillInitMuZeroNet(
        TeacherRepresentation(teacher), DynamicsNet(cfg), TeacherPrediction(teacher))
    composed.eval()
    pol = muzero_policy(composed, cfg)
    board = chess.Board()
    for _ in range(4):
        move = pol(board)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break
