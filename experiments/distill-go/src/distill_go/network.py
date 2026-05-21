"""ResNet trunk with policy + value heads for Go.

Same structure as wm_chess.network.AlphaZeroNet, but parameterized for
arbitrary board size and input-plane count.

Input  (B, n_input_planes, S, S)
Trunk  Conv(filters, 3x3) + BN + ReLU + n_res_blocks × ResidualBlock
Policy Conv(2, 1x1) + BN + ReLU + flatten + Linear(2*S², S² + 1)
Value  Conv(1, 1x1) + BN + ReLU + flatten + Linear(S², 64) + ReLU
       + Linear(64, 1) + tanh

Policy output dim = S² + 1 (board points + pass).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GoConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class AlphaZeroGoNet(nn.Module):
    def __init__(self, cfg: GoConfig):
        super().__init__()
        self.cfg = cfg
        s = cfg.board_size
        f = cfg.n_filters

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.n_input_planes, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(f) for _ in range(cfg.n_res_blocks)]
        )

        # Policy head: 1x1 conv → 2 ch → flatten → linear to S² + 1.
        self.policy_conv = nn.Conv2d(f, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * s * s, s * s + 1)

        # Value head: 1x1 conv → 1 ch → flatten → linear → tanh.
        self.value_conv = nn.Conv2d(f, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(s * s, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.trunk(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(start_dim=1)
        p = self.policy_fc(p)                  # (B, S² + 1) — raw logits

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)   # (B,)
        return p, v


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
