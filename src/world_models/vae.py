"""Convolutional VAE — the V (Vision) module.

Architecture is the original Ha & Schmidhuber 2018 spec, which is small
enough to train in a few minutes on MPS:

    encoder: (3, 64, 64) -> 4 stride-2 convs -> (256, 2, 2) -> Linear -> z (32)
    decoder: z (32) -> Linear -> (1024, 1, 1) -> 4 stride-2 deconvs -> (3, 64, 64)

The trick: the encoder outputs *two* vectors — a mean `mu` and a log
variance `logvar`. We sample `z ~ N(mu, exp(logvar))` via the reparam
trick (so gradients can flow through the sampling). The decoder maps that
sample back to a frame. The KL term keeps the latent distribution close
to a standard Gaussian, so the latent space is smooth and the RNN can
learn dynamics on it without adversarial weirdness.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from world_models.config import CFG


class Encoder(nn.Module):
    def __init__(self, z_dim: int = CFG.z_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)    # 64 -> 31
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)   # 31 -> 14
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)  # 14 -> 6
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2) # 6  -> 2
        self.fc_mu = nn.Linear(256 * 2 * 2, z_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim: int = CFG.z_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim, 1024)
        # Deconv kernels chosen so 1 -> 5 -> 13 -> 30 -> 64.
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 1024, 1, 1)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        return torch.sigmoid(self.deconv4(h))


@dataclass
class VAEOutput:
    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class VAE(nn.Module):
    def __init__(self, z_dim: int = CFG.z_dim) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar, z=z)

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding: skip the sampling, just take mu.

        Used at inference / for encoding rollouts before training the RNN.
        """
        mu, _ = self.encoder(x)
        return mu


def vae_loss(out: VAEOutput, target: torch.Tensor, kl_weight: float = CFG.vae_kl_weight) -> dict:
    """β-VAE loss: pixel BCE + KL(q(z|x) || N(0, I)).

    Returns a dict so train scripts can log each piece separately.
    """
    # Sum-then-mean-over-batch matches the original World Models repo.
    recon = F.binary_cross_entropy(out.recon, target, reduction="sum") / target.size(0)
    kl = -0.5 * torch.sum(1 + out.logvar - out.mu.pow(2) - out.logvar.exp()) / target.size(0)
    total = recon + kl_weight * kl
    return {"loss": total, "recon": recon.detach(), "kl": kl.detach()}
