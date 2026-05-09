"""ResNet trunk with policy + value heads.

input  (B, 19, 8, 8)
  └─> Conv(filters, 3x3) + BN + ReLU                         "stem"
  └─> n_res_blocks x ResidualBlock                            "trunk"
  ├─> Policy: Conv(73, 1x1) -> flatten -> (B, 4672)           (no softmax; masked at inference)
  └─> Value:  Conv(1, 1x1) + BN + ReLU + FC(64) + ReLU + FC(1) + tanh -> (B,)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .board import N_INPUT_PLANES, N_MOVE_PLANES, N_POLICY
from .config import Config


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


class AlphaZeroNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        f = cfg.n_filters
        self.stem = nn.Sequential(
            nn.Conv2d(N_INPUT_PLANES, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(f) for _ in range(cfg.n_res_blocks)])

        # Policy head: 1x1 conv -> 73 channels, flatten to 4672.
        self.policy_conv = nn.Conv2d(f, N_MOVE_PLANES, 1)

        # Value head: 1x1 conv -> 1 channel -> 64 -> 1 -> tanh.
        self.value_conv = nn.Conv2d(f, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.trunk(h)

        p = self.policy_conv(h)              # (B, 73, 8, 8)
        p = p.flatten(start_dim=1)           # (B, 4672) -- plane*64 + sq layout
        assert p.shape[-1] == N_POLICY

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)           # (B, 64)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,)
        return p, v


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
