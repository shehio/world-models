"""C — controller.

Tiny linear policy on `[z, h]`. The point of the world model is that the
hard work of perception (V) and short-term prediction (M) is already
done, so the policy can be very small. Ha & Schmidhuber's controller is
a single linear layer with ~1k parameters that gets trained with
**evolution**, not gradient descent — the env is non-differentiable so
black-box optimization (CMA-ES) is a natural fit.

The controller maps to CarRacing's continuous action box:
    steer in [-1, 1]  (tanh)
    gas   in [0, 1]   (sigmoid)
    brake in [0, 1]   (sigmoid)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from world_models.config import CFG


class Controller(nn.Module):
    def __init__(
        self,
        z_dim: int = CFG.z_dim,
        h_dim: int = CFG.rnn_hidden,
        action_dim: int = CFG.action_dim,
        hidden: int = CFG.controller_hidden,
    ) -> None:
        super().__init__()
        in_dim = z_dim + h_dim
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, action_dim),
            )
        else:
            self.net = nn.Linear(in_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, h], dim=-1)
        raw = self.net(x)
        # CarRacing action layout: [steer, gas, brake].
        steer = torch.tanh(raw[..., 0:1])
        gas = torch.sigmoid(raw[..., 1:2])
        brake = torch.sigmoid(raw[..., 2:3])
        return torch.cat([steer, gas, brake], dim=-1)

    @torch.no_grad()
    def act_numpy(self, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        """One-step inference helper used by the env loop."""
        device = next(self.parameters()).device
        zt = torch.from_numpy(z).float().unsqueeze(0).to(device)
        ht = torch.from_numpy(h).float().unsqueeze(0).to(device)
        a = self.forward(zt, ht).squeeze(0).cpu().numpy()
        return a

    # ---- Flat-vector interface for CMA-ES ----------------------------
    def get_flat(self) -> np.ndarray:
        return torch.cat([p.detach().flatten() for p in self.parameters()]).cpu().numpy()

    def set_flat(self, flat: np.ndarray) -> None:
        flat_t = torch.tensor(flat, dtype=torch.float32)
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat_t[offset : offset + n].view_as(p))
            offset += n

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
