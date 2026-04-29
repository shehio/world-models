"""Single source of truth for hyperparameters and paths.

Everything in the project pulls from here so we never hard-code a magic
number twice. This is also why the trained VAE knows its z-dim, the RNN
knows the VAE's z-dim, and the controller knows both — they all import
this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # -- Environment ------------------------------------------------------
    env_id: str = "CarRacing-v3"
    # CarRacing has continuous actions: [steer in [-1,1], gas in [0,1], brake in [0,1]].
    action_dim: int = 3
    image_size: int = 64           # we resize 96x96 -> 64x64 RGB
    frame_channels: int = 3

    # -- VAE --------------------------------------------------------------
    z_dim: int = 32                # latent dim per frame (Ha & Schmidhuber 2018)
    vae_batch_size: int = 128
    vae_lr: float = 1e-4
    vae_epochs: int = 6            # short for our 2-3h budget; bump later
    vae_kl_weight: float = 1.0

    # -- MDN-RNN ----------------------------------------------------------
    rnn_hidden: int = 256
    rnn_layers: int = 1
    mdn_components: int = 5
    rnn_seq_len: int = 64          # truncated BPTT length
    rnn_batch_size: int = 32
    rnn_lr: float = 1e-3
    rnn_epochs: int = 6

    # -- Controller -------------------------------------------------------
    # Linear policy on [z, h] -> action. Tiny because V+M did the heavy lifting.
    controller_hidden: int = 0     # 0 = pure linear (Ha & Schmidhuber baseline)

    # CMA-ES schedule for the controller. Small numbers to fit our budget.
    cma_pop: int = 12
    cma_generations: int = 6
    cma_episodes_per_eval: int = 1
    cma_max_steps: int = 600       # cap per episode for budget control
    cma_init_sigma: float = 0.3

    # -- Data collection --------------------------------------------------
    n_rollouts: int = 12
    rollout_max_steps: int = 800   # CarRacing default truncates at 1000
    seed: int = 42

    # -- Paths ------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    rollouts_dir: Path = ROOT / "data" / "rollouts"
    latents_dir: Path = ROOT / "data" / "latents"
    ckpt_dir: Path = ROOT / "checkpoints"
    video_dir: Path = ROOT / "videos"

    @property
    def vae_path(self) -> Path:
        return self.ckpt_dir / "vae.pt"

    @property
    def rnn_path(self) -> Path:
        return self.ckpt_dir / "rnn.pt"

    @property
    def controller_path(self) -> Path:
        return self.ckpt_dir / "controller.pt"


CFG = Config()
for d in (CFG.data_dir, CFG.rollouts_dir, CFG.latents_dir, CFG.ckpt_dir, CFG.video_dir):
    d.mkdir(parents=True, exist_ok=True)
