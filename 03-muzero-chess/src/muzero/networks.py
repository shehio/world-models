"""Three networks: representation h, dynamics g, prediction f.

Action encoding into the dynamics network: an action is encoded as a
"move plane" — a (n_move_planes=73, 8, 8) tensor that's all zero except
for a single 1 at the (plane, from_rank, from_file) of the chosen move.
This is concatenated with the latent state along the channel axis
before being fed to the dynamics ResNet.

representation h:
    input  (B, 19, 8, 8)
      └─> Stem + n_res_blocks_repr trunk
      └─> output (B, latent_channels, 8, 8)

dynamics g:
    input  (B, latent_channels + n_move_planes, 8, 8)
      └─> Stem + n_res_blocks_dyn trunk
      ├─> next_state head: 1x1 conv -> (B, latent_channels, 8, 8)
      └─> reward head:     global pool -> Linear -> (B,) scalar

prediction f:
    input  (B, latent_channels, 8, 8)
      └─> n_res_blocks_pred trunk
      ├─> policy head: same as AlphaZero -> (B, 4672)
      └─> value head:  same as AlphaZero -> (B,) tanh

Min-max normalization of latent: paper scales s to [0, 1] elementwise
within batch before feeding to dynamics/prediction. We'll add it once
training is running; not strictly required for chess.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


class RepresentationNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DynamicsNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError

    def forward(
        self, state: torch.Tensor, action_plane: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class PredictionNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MuZeroNet(nn.Module):
    """Bundles h, g, f. Exposes initial_inference(obs) and recurrent_inference(s, a)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.h = RepresentationNet(cfg)
        self.g = DynamicsNet(cfg)
        self.f = PredictionNet(cfg)

    def initial_inference(self, obs):
        """obs -> (s, policy, value). Reward is implicit zero for the root."""
        raise NotImplementedError

    def recurrent_inference(self, s, action_plane):
        """(s, a) -> (s', reward, policy, value)."""
        raise NotImplementedError
