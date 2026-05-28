"""Factor a distilled AlphaZeroNet into MuZero's h (representation) + f
(prediction), so only the dynamics g has to be trained.

The distilled net (wm_chess.network.AlphaZeroNet) is:
    obs (B,19,8,8) → stem → trunk → ┬→ policy_conv → (B,4672)
                                    └→ value head  → (B,)
which factors exactly onto MuZero:
    h_θ = stem + trunk      → latent (B, n_filters, 8, 8)
    f_θ = policy + value heads on that latent
    g_θ = the only NEW network: (latent, action) → (next_latent, reward)

Because TeacherPrediction reuses the teacher's *exact* head submodules and
forward math, f(h(obs)) reproduces the teacher's (policy, value) bit-for-bit
— so root MCTS evaluation starts at the teacher's strength. h and f are
frozen; training touches only g.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet


def _unwrap_state_dict(obj: Any) -> Mapping[str, Any]:
    """Accept either a bare state_dict (distill-soft train.py) or the wrapped
    selfplay format ({'net': state_dict, ...}). Inlined so muzero-chess doesn't
    take a dependency on the distill-soft package for three lines."""
    if isinstance(obj, dict) and "net" in obj and isinstance(obj["net"], dict):
        return obj["net"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"unsupported checkpoint format: {type(obj).__name__}")


def load_teacher(ckpt_path: str, *, n_blocks: int = 20, n_filters: int = 256,
                 map_location: str | torch.device = "cpu") -> AlphaZeroNet:
    """Build an AlphaZeroNet of the teacher's shape and load its weights."""
    cfg = Config(n_res_blocks=n_blocks, n_filters=n_filters)
    net = AlphaZeroNet(cfg)
    obj = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    net.load_state_dict(_unwrap_state_dict(obj))
    net.eval()
    return net


class TeacherRepresentation(nn.Module):
    """h_θ — the teacher's stem + trunk. Output is the latent state."""

    def __init__(self, teacher: AlphaZeroNet):
        super().__init__()
        self.stem = teacher.stem
        self.trunk = teacher.trunk

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.stem(obs))


class TeacherPrediction(nn.Module):
    """f_θ — the teacher's policy + value heads. Forward math copied verbatim
    from AlphaZeroNet.forward so f(h(obs)) == teacher(obs)."""

    def __init__(self, teacher: AlphaZeroNet):
        super().__init__()
        self.policy_conv = teacher.policy_conv
        self.value_conv = teacher.value_conv
        self.value_bn = teacher.value_bn
        self.value_fc1 = teacher.value_fc1
        self.value_fc2 = teacher.value_fc2

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.policy_conv(latent).flatten(start_dim=1)        # (B, 4672)
        v = F.relu(self.value_bn(self.value_conv(latent)))
        v = v.flatten(start_dim=1)                               # (B, 64)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)            # (B,)
        return p, v


class DistillInitMuZeroNet(nn.Module):
    """Composed MuZero net: frozen teacher h + f, trainable dynamics g.

    Exposes initial_inference / recurrent_inference with the SAME signatures
    as muzero_chess.networks.MuZeroNet, so run_mcts_batched + muzero_policy
    work against it unchanged.
    """

    def __init__(self, h: TeacherRepresentation, g: nn.Module, f: TeacherPrediction,
                 *, freeze_teacher: bool = True):
        super().__init__()
        self.h = h
        self.g = g
        self.f = f
        if freeze_teacher:
            for p in self.h.parameters():
                p.requires_grad_(False)
            for p in self.f.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep the frozen teacher in eval mode (running BN stats, deterministic)
        even when the composed module is switched to train() for g's sake."""
        super().train(mode)
        self.h.eval()
        self.f.eval()
        return self

    def initial_inference(self, obs: torch.Tensor):
        s = self.h(obs)
        p, v = self.f(s)
        return s, p, v

    def recurrent_inference(self, latent: torch.Tensor, action: torch.Tensor):
        s_next, r = self.g(latent, action)
        p, v = self.f(s_next)
        return s_next, r, p, v
