"""MDN-RNN — the M (Memory) module.

Given the current latent `z_t` and action `a_t`, the LSTM updates its
hidden state and emits parameters for a *Gaussian mixture* over the next
latent `z_{t+1}`:

        p(z_{t+1} | z_t, a_t, h_t) = sum_k pi_k * N(mu_k, sigma_k^2)

We use a mixture (not a single Gaussian) because the next state is
genuinely uncertain: rounding a corner in CarRacing, the next frame
could be "tree on left" or "tree on right" — averaging those modes
would predict a tree in the middle, which is wrong. A mixture lets the
model commit to multiple possible futures with separate weights.

The hidden state `h_t` is also what the controller reads — it carries
"recent context" beyond the single current frame.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from world_models.config import CFG


class MDNRNN(nn.Module):
    def __init__(
        self,
        z_dim: int = CFG.z_dim,
        action_dim: int = CFG.action_dim,
        hidden: int = CFG.rnn_hidden,
        num_layers: int = CFG.rnn_layers,
        n_mixtures: int = CFG.mdn_components,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.K = n_mixtures

        self.lstm = nn.LSTM(
            input_size=z_dim + action_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        # MDN head: K mixing logits + K means + K log-sigmas, each per latent dim.
        self.mdn = nn.Linear(hidden, n_mixtures * (1 + 2 * z_dim))

    def init_hidden(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch, self.hidden, device=device)
        c = torch.zeros(self.num_layers, batch, self.hidden, device=device)
        return h, c

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """`z` is (B, T, z_dim), `a` is (B, T, action_dim)."""
        x = torch.cat([z, a], dim=-1)
        out, hidden_out = self.lstm(x, hidden)
        params = self.mdn(out)  # (B, T, K * (1 + 2*z_dim))

        B, T, _ = params.shape
        params = params.view(B, T, self.K, 1 + 2 * self.z_dim)
        logit_pi = params[..., 0]                    # (B, T, K)
        mu = params[..., 1 : 1 + self.z_dim]         # (B, T, K, z_dim)
        log_sigma = params[..., 1 + self.z_dim :]    # (B, T, K, z_dim)
        # Bound log-sigma to avoid NaNs early in training.
        log_sigma = torch.clamp(log_sigma, min=-7.0, max=2.0)
        return (logit_pi, mu, log_sigma), hidden_out


def mdn_loss(
    logit_pi: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of `target` under the predicted GMM.

    `target` has shape (B, T, z_dim). We expand it across the K components
    and combine the per-dimension Gaussians (assuming independent dims, as
    in the original paper) with the mixing weights.
    """
    target = target.unsqueeze(2)  # (B, T, 1, z_dim)
    # log N(target | mu, sigma) summed over the latent dim (independent dims).
    log_norm = (
        -0.5 * math.log(2 * math.pi)
        - log_sigma
        - 0.5 * ((target - mu) / log_sigma.exp()) ** 2
    ).sum(dim=-1)  # (B, T, K)
    log_pi = F.log_softmax(logit_pi, dim=-1)
    log_prob = torch.logsumexp(log_pi + log_norm, dim=-1)  # (B, T)
    return -log_prob.mean()


def sample_mdn(
    logit_pi: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample one latent from the GMM at a single time step.

    Shapes: `logit_pi` (B, K), `mu` (B, K, z_dim), `log_sigma` (B, K, z_dim).
    """
    # Temperature-scaled sampling, like in the paper's "dream" rollouts:
    # higher temperature = more diverse / less confident hallucinations.
    pi = F.softmax(logit_pi / max(temperature, 1e-6), dim=-1)
    k = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (B,)
    idx = k.view(-1, 1, 1).expand(-1, 1, mu.size(-1))
    chosen_mu = mu.gather(1, idx).squeeze(1)
    chosen_log_sigma = log_sigma.gather(1, idx).squeeze(1)
    eps = torch.randn_like(chosen_mu) * math.sqrt(temperature)
    return chosen_mu + eps * chosen_log_sigma.exp()
