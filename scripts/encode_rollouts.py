"""Use the trained VAE to encode every rollout frame into a latent.

We save (z, a) per episode. The RNN trains on these — we don't re-encode
on the fly because that would dominate the per-step cost.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from world_models.config import CFG
from world_models.utils import pick_device
from world_models.vae import VAE


def main() -> None:
    device = pick_device()
    model = VAE().to(device)
    state = torch.load(CFG.vae_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    files = sorted(Path(CFG.rollouts_dir).glob("*.npz"))
    print(f"encoding {len(files)} rollouts...")
    t0 = time.time()
    for f in tqdm(files):
        data = np.load(f)
        frames = data["frames"]  # (T, H, W, C) float32 in [0,1]
        actions = data["actions"]
        x = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).float().to(device)
        with torch.no_grad():
            mu = model.encode_mean(x).cpu().numpy()
        out_path = CFG.latents_dir / f.name
        np.savez_compressed(out_path, z=mu.astype(np.float32), a=actions.astype(np.float32))
    print(f"done in {time.time()-t0:.1f}s -> {CFG.latents_dir}")


if __name__ == "__main__":
    main()
