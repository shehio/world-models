"""The three MuZero networks: representation (h), dynamics (g), prediction (f).

```
   real observation o_t  (B, 19, 8, 8)
            │
            ▼  h_θ
       latent state s⁰_t  (B, C, 8, 8)
            │
            ▼   recursive descent via g_θ during MCTS:
                g_θ(s_t^k, a_{t+1}) = (s_t^{k+1}, r_t^{k+1})
            ▼
       latent state s_t^k  (B, C, 8, 8)
            │
            ▼  f_θ
       (policy_logits, value)
            (B, 4672)      (B,)
```

The fixed-size latent (`C` channels × 8×8 = 64×64 = 4,096 scalar units
by default) is the only carrier of information from t=0 forward into
the search tree. The dynamics network is the only thing that ever sees
the action — it conditions on a one-hot action plane broadcast across
the 8x8 grid.

The reward output is a scalar (B,) in chess where every move yields 0
reward and the terminal yields ±1. We use a scalar regression head
instead of a categorical reward head (MuZero paper Table 1 used 601 bins);
for chess's two-valued reward, scalar regression is fine.

The prediction value head is also a scalar (B,) in [−1, +1] — tanh
output, same as AlphaZero. Value at non-root latent is the
bootstrap target during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MuZeroConfig, BOARD_SIZE


class ResidualBlock(nn.Module):
    """Plain residual block, same shape as in wm_chess.network.

    Identical channels in and out; two 3×3 convs with BN + ReLU.
    """

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


def _resnet_stack(in_ch: int, mid_ch: int, n_blocks: int) -> nn.Sequential:
    """Stem (3x3 conv → BN → ReLU) + n_blocks residual blocks."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
    ]
    layers += [ResidualBlock(mid_ch) for _ in range(n_blocks)]
    return nn.Sequential(*layers)


class RepresentationNet(nn.Module):
    """h_θ: real observation → latent state.

    Input:  (B, 19, 8, 8) — wm_chess.board.encode_board output.
    Output: (B, C, 8, 8) — initial latent state for MCTS root.
    """

    def __init__(self, cfg: MuZeroConfig):
        super().__init__()
        self.trunk = _resnet_stack(
            in_ch=cfg.input_planes,
            mid_ch=cfg.repr_n_filters,
            n_blocks=cfg.repr_n_res_blocks,
        )
        # Project to latent_channels so the rest of the pipeline has a fixed C.
        self.project = nn.Conv2d(cfg.repr_n_filters, cfg.latent_channels, 1, bias=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        return self.project(h)


class DynamicsNet(nn.Module):
    """g_θ: (latent, action) → (next_latent, reward).

    The action is encoded as a (B, 1, 8, 8) plane: 1 at the source-square
    of the move, broadcast to fill (could be richer; this is the smallest
    encoding that still tells g_θ "where on the board the move happened").
    A full MuZero implementation would use a more expressive action plane
    (e.g. one channel per move-direction); this is a simpler educational
    choice that's enough to train on chess.

    The reward head produces a scalar in [−1, +1] (tanh): per-move reward
    is 0 in chess except at terminal where it's ±1, but the dynamics net
    has to learn this — it doesn't observe whether a move ended the game.

    Input:  latent (B, C, 8, 8), action (B,) int64 indices into [0, 4672)
    Output: next_latent (B, C, 8, 8), reward (B,)
    """

    def __init__(self, cfg: MuZeroConfig):
        super().__init__()
        self.action_dim = cfg.action_dim
        # latent concatenated with action plane → conv stack → next_latent.
        in_ch = cfg.latent_channels + 1
        self.trunk = _resnet_stack(
            in_ch=in_ch,
            mid_ch=cfg.dyn_n_filters,
            n_blocks=cfg.dyn_n_res_blocks,
        )
        self.next_latent_proj = nn.Conv2d(cfg.dyn_n_filters, cfg.latent_channels, 1, bias=False)
        # Reward head: 1×1 conv → global mean pool → linear → tanh.
        self.reward_conv = nn.Conv2d(cfg.dyn_n_filters, 1, 1, bias=False)
        self.reward_bn = nn.BatchNorm2d(1)
        self.reward_fc = nn.Linear(1, 1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor):
        B = latent.shape[0]
        # Encode action as a source-square plane (best single-channel choice
        # for the AlphaZero 8×8×73 move encoding: source square = action // 73).
        action_source = (action // 73).long()    # (B,) in [0, 64)
        plane = torch.zeros(B, 1, BOARD_SIZE * BOARD_SIZE, device=latent.device, dtype=latent.dtype)
        plane.scatter_(2, action_source.view(B, 1, 1), 1.0)
        plane = plane.view(B, 1, BOARD_SIZE, BOARD_SIZE)
        # Concat along channel dim.
        x = torch.cat([latent, plane], dim=1)
        h = self.trunk(x)
        next_latent = self.next_latent_proj(h)
        r = F.relu(self.reward_bn(self.reward_conv(h)))
        r = r.mean(dim=(2, 3))                # (B, 1) global average
        reward = torch.tanh(self.reward_fc(r)).squeeze(-1)
        return next_latent, reward


class PredictionNet(nn.Module):
    """f_θ: latent → (policy_logits, value).

    Input:  latent (B, C, 8, 8)
    Output: policy_logits (B, 4672), value (B,)
    """

    def __init__(self, cfg: MuZeroConfig):
        super().__init__()
        self.trunk = _resnet_stack(
            in_ch=cfg.latent_channels,
            mid_ch=cfg.pred_n_filters,
            n_blocks=cfg.pred_n_res_blocks,
        )
        # Policy: 1x1 conv → 73 channels → flatten → 4672 raw logits.
        # (Mirrors wm_chess.network's AlphaZero policy head.)
        self.policy_conv = nn.Conv2d(cfg.pred_n_filters, 73, 1, bias=False)
        # Value: 1x1 conv → 1 ch → flatten → 64-dim → 1 → tanh.
        self.value_conv = nn.Conv2d(cfg.pred_n_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, latent: torch.Tensor):
        h = self.trunk(latent)
        p = self.policy_conv(h)                                  # (B, 73, 8, 8)
        p = p.flatten(start_dim=1)                                # (B, 4672)
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p, v


class MuZeroNet(nn.Module):
    """Convenience wrapper that owns h, g, f together.

    The MCTS uses these in different combinations:
      - root expansion: h(obs) → s, then f(s) → (policy, value)
      - non-root expansion: g(s, a) → (s', r), then f(s') → (policy, value)

    So the three networks are needed independently rather than as a single
    forward() call.
    """

    def __init__(self, cfg: MuZeroConfig):
        super().__init__()
        self.cfg = cfg
        self.h = RepresentationNet(cfg)
        self.g = DynamicsNet(cfg)
        self.f = PredictionNet(cfg)

    def initial_inference(self, obs: torch.Tensor):
        """Root step: encode observation, predict policy + value, reward=0."""
        s = self.h(obs)
        p, v = self.f(s)
        # No reward at root — caller can treat as 0.
        return s, p, v

    def recurrent_inference(self, latent: torch.Tensor, action: torch.Tensor):
        """Non-root step: apply dynamics, predict policy + value at the new state."""
        s_next, r = self.g(latent, action)
        p, v = self.f(s_next)
        return s_next, r, p, v
