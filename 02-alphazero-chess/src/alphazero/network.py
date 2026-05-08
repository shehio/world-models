"""ResNet trunk with policy + value heads.

Architecture:
    input  (B, 19, 8, 8)
      └─> Conv(filters=64, 3x3) + BN + ReLU         "stem"
      └─> n_res_blocks × ResidualBlock              "trunk"
      ├─> Policy head: Conv(2, 1x1)+BN+ReLU -> Flatten -> Linear(4672)
      └─> Value head:  Conv(1, 1x1)+BN+ReLU -> Flatten -> Linear(64) -> ReLU -> Linear(1) -> tanh

Each ResidualBlock = Conv-BN-ReLU-Conv-BN + skip + ReLU.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AlphaZeroNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 19, 8, 8) -> (policy_logits (B, 4672), value (B,))"""
        raise NotImplementedError
