"""Dream rollouts: let M imagine the future, then decode with V.

Pick a random frame from a real rollout, encode it to z_0, then let the
RNN predict z_1, z_2, ... while we feed the *RNN's own* sampled latents
back in. Decode each latent and write a side-by-side animation:

    [ ground-truth real frames | dream frames ]

This is the most fun visualization in the whole project — when it works,
the RNN clearly hallucinates plausible road geometry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from world_models.config import CFG
from world_models.mdn_rnn import MDNRNN, sample_mdn
from world_models.utils import pick_device, seed_everything
from world_models.vae import VAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=str, default=None,
                        help="rollout npz to seed from; default = first one")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--out", type=str, default=str(CFG.video_dir / "dream.gif"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()

    if args.rollout is None:
        rfiles = sorted(Path(CFG.rollouts_dir).glob("*.npz"))
        if not rfiles:
            raise FileNotFoundError("no rollouts to seed from")
        args.rollout = str(rfiles[0])
    data = np.load(args.rollout)
    frames = data["frames"]
    actions = data["actions"]
    T = min(args.steps, frames.shape[0])

    vae = VAE().to(device)
    vae.load_state_dict(torch.load(CFG.vae_path, map_location=device)["state_dict"])
    vae.eval()
    rnn = MDNRNN().to(device)
    rnn.load_state_dict(torch.load(CFG.rnn_path, map_location=device)["state_dict"])
    rnn.eval()

    real_frames = frames[:T]                                                    # (T, H, W, C) float
    a_seq = torch.from_numpy(actions[:T]).float().to(device)                    # (T, 3)
    seed_frame = torch.from_numpy(real_frames[0]).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        z = vae.encode_mean(seed_frame)                                         # (1, z_dim)
    h, c = rnn.init_hidden(batch=1, device=device)

    dream_frames = []
    with torch.no_grad():
        for t in range(T):
            decoded = vae.decoder(z).squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            dream_frames.append((decoded * 255).astype(np.uint8))
            (logit_pi, mu, log_sigma), (h, c) = rnn(
                z.unsqueeze(1), a_seq[t : t + 1].unsqueeze(0), (h, c)
            )
            z = sample_mdn(
                logit_pi.squeeze(1), mu.squeeze(1), log_sigma.squeeze(1),
                temperature=args.temperature,
            )

    side_by_side = []
    real_uint8 = (real_frames * 255).astype(np.uint8)
    for r, d in zip(real_uint8, dream_frames):
        side_by_side.append(np.concatenate([r, d], axis=1))

    imageio.mimwrite(args.out, side_by_side, duration=1 / 15, loop=0)
    print(f"wrote dream gif -> {args.out}  (left=real, right=dream)")


if __name__ == "__main__":
    main()
