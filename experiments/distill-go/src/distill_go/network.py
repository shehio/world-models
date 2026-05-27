"""ResNet trunk with policy + value heads for Go, plus KataGo aux heads.

Base architecture (matches the original distill-go demo):

  Input  (B, n_input_planes, S, S)
  Trunk  Conv(filters, 3x3) + BN + ReLU + n_res_blocks × ResidualBlock
  Policy Conv(2, 1x1) + BN + ReLU + flatten + Linear(2·S², S²+1)
  Value  Conv(1, 1x1) + BN + ReLU + flatten + Linear(S², 64) + ReLU + Linear(64, 1) + tanh

KataGo efficiency tricks (off by default; toggle via GoConfig flags):

  cfg.use_global_pool       — Replaces standard residual blocks with KataGo-style
                              "global-pool-residual" blocks. After the second
                              conv, channel-wise mean + max pools over the spatial
                              dims are broadcast back across the board so subsequent
                              convs can condition on global context. ~1.60× faster
                              training per the KataGo paper.
  cfg.use_aux_ownership     — Adds two extra heads in parallel to policy/value:
                                  ownership: (B, 3, S, S) softmax over {B, W, E}
                                  score:     (B,) scalar — final score margin
                                             (positive = BLACK ahead by ⋯ points)
                              At game end, ownership target = final occupancy of
                              each point under Tromp-Taylor scoring, score target =
                              (black_pts − white_pts) clipped to ±board_size².
                              Per the paper, this is the densest learning signal —
                              every point becomes a label. ~1.65× faster.
  cfg.use_aux_opp_policy    — Adds an opponent-policy head: predicts the move the
                              opponent will play NEXT turn given the current state.
                              Same 4672-dim policy shape as the main head, trained
                              against the actually-played next move. ~1.30× faster.

Loading a checkpoint trained without these flags into a config with them ON: use
`net.load_state_dict(state, strict=False)` — the trunk + policy/value heads load
fine; the new aux heads / pooled layers start from random init and learn through
the self-play loop.

forward(x) signature stays (policy_logits, value) for backward compat with all
existing callers (MCTS, h2h, eval). The aux outputs are accessed via the
companion `forward_aux(x)` method that returns a dict — keeps the hot path of
inference free of aux work it doesn't need.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GoConfig


class ResidualBlock(nn.Module):
    """Plain residual block — used when use_global_pool=False."""

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


class GlobalPoolResidualBlock(nn.Module):
    """KataGo-style residual block with a global-pooling branch.

    Half the trunk channels feed a "regular" conv path; the other half feed
    a pooling path whose output (mean + max + per-S^2 normalization
    constants) gets broadcast across the board and added back into the
    regular branch before the second conv. The resulting block has the
    same input/output channel count as a plain residual block but lets
    the convolutions condition on global board context.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Split channels evenly between the regular and pool branches.
        # KataGo paper uses C_pool = C/2; we follow.
        c_reg = channels - channels // 2
        c_pool = channels // 2
        self.c_reg = c_reg
        self.c_pool = c_pool

        # First conv splits into (reg, pool) channels.
        self.conv1 = nn.Conv2d(channels, c_reg + c_pool, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_reg + c_pool)

        # Pool branch: mean + max → 2*c_pool global features → linear back to c_reg.
        self.pool_fc = nn.Linear(2 * c_pool, c_reg)

        self.conv2 = nn.Conv2d(c_reg, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h_reg = h[:, : self.c_reg]
        h_pool = h[:, self.c_reg :]
        # Pool over spatial dims: mean + max.
        b = h_pool.shape[0]
        mean = h_pool.mean(dim=(2, 3))                 # (B, c_pool)
        max_ = h_pool.amax(dim=(2, 3))                 # (B, c_pool)
        global_feat = torch.cat([mean, max_], dim=1)   # (B, 2*c_pool)
        global_feat = self.pool_fc(global_feat)        # (B, c_reg)
        # Broadcast across spatial dims and add to regular branch.
        h_reg = h_reg + global_feat[..., None, None]
        h_reg = F.relu(h_reg)
        h2 = self.bn2(self.conv2(h_reg))
        return F.relu(x + h2)


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

        # Trunk: either plain or global-pool residual blocks.
        block_cls = GlobalPoolResidualBlock if cfg.use_global_pool else ResidualBlock
        self.trunk = nn.Sequential(*[block_cls(f) for _ in range(cfg.n_res_blocks)])

        # Policy head: 1x1 conv → 2 ch → flatten → linear to S² + 1.
        self.policy_conv = nn.Conv2d(f, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * s * s, s * s + 1)

        # Value head: 1x1 conv → 1 ch → flatten → linear → tanh.
        self.value_conv = nn.Conv2d(f, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(s * s, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # KataGo aux heads — created only if flagged on, so prior loads don't
        # pull in unused weights.
        if cfg.use_aux_ownership:
            # Ownership head: 1x1 conv → 3 channels (B / W / EMPTY) → softmax per point.
            # Score head: 1x1 conv → 1 ch + global mean pool → linear → scalar.
            self.ownership_conv = nn.Conv2d(f, 3, 1, bias=False)
            self.score_conv = nn.Conv2d(f, 1, 1, bias=False)
            self.score_bn = nn.BatchNorm2d(1)
            self.score_fc = nn.Linear(1, 1)   # from global-pooled scalar to scalar
        if cfg.use_aux_opp_policy:
            # Opponent-policy head: mirror the policy head shape.
            self.opp_policy_conv = nn.Conv2d(f, 2, 1, bias=False)
            self.opp_policy_bn = nn.BatchNorm2d(2)
            self.opp_policy_fc = nn.Linear(2 * s * s, s * s + 1)

    # ------------------------------------------------------------------ forward

    def _trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.stem(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Main policy + value forward — unchanged signature.

        MCTS / eval / h2h all use this. Aux heads not computed on this path.
        """
        h = self._trunk_features(x)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(start_dim=1)
        p = self.policy_fc(p)                  # (B, S² + 1) — raw logits

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)   # (B,)
        return p, v

    def forward_aux(self, x: torch.Tensor) -> dict:
        """Train-time forward that computes EVERY head present on this net.

        Returns a dict:
          {
            "policy":  (B, S²+1) logits
            "value":   (B,) tanh-bounded
            "ownership": (B, 3, S, S) raw logits     [only if use_aux_ownership]
            "score":     (B,) scalar                 [only if use_aux_ownership]
            "opp_policy":(B, S²+1) logits            [only if use_aux_opp_policy]
          }
        """
        h = self._trunk_features(x)
        out: dict = {}

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(start_dim=1)
        out["policy"] = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        out["value"] = torch.tanh(self.value_fc2(v)).squeeze(-1)

        if self.cfg.use_aux_ownership:
            # Ownership: per-point softmax over 3 classes.
            out["ownership"] = self.ownership_conv(h)        # (B, 3, S, S)
            # Score: per-board global feature → scalar.
            sc = F.relu(self.score_bn(self.score_conv(h)))   # (B, 1, S, S)
            sc = sc.mean(dim=(2, 3))                         # (B, 1) global avg
            out["score"] = self.score_fc(sc).squeeze(-1)     # (B,)

        if self.cfg.use_aux_opp_policy:
            op = F.relu(self.opp_policy_bn(self.opp_policy_conv(h)))
            op = op.flatten(start_dim=1)
            out["opp_policy"] = self.opp_policy_fc(op)

        return out


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
