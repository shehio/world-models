"""Play CarRacing with the full V + M + C stack and record a video."""
from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
import torch

from world_models.config import CFG
from world_models.controller import Controller
from world_models.env import make_env, preprocess
from world_models.mdn_rnn import MDNRNN
from world_models.utils import pick_device, seed_everything
from world_models.vae import VAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--out", type=str, default=str(CFG.video_dir / "play.mp4"))
    parser.add_argument("--gif", type=str, default=str(CFG.video_dir / "play.gif"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()

    vae = VAE().to(device)
    vae.load_state_dict(torch.load(CFG.vae_path, map_location=device)["state_dict"])
    vae.eval()
    rnn = MDNRNN().to(device)
    rnn.load_state_dict(torch.load(CFG.rnn_path, map_location=device)["state_dict"])
    rnn.eval()
    ctrl = Controller().to(device)
    ctrl.load_state_dict(torch.load(CFG.controller_path, map_location=device)["state_dict"])
    ctrl.eval()

    env = make_env(render_mode="rgb_array")
    all_frames = []
    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        h, c = rnn.init_hidden(batch=1, device=device)
        ep_return = 0.0
        for step in range(args.max_steps):
            frame = preprocess(obs)
            x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                z = vae.encode_mean(x)
                h_flat = h[-1]
                a = ctrl(z, h_flat)
            a_np = a.squeeze(0).cpu().numpy()
            render = env.render()
            if render is not None:
                all_frames.append(render)
            obs, reward, terminated, truncated, _ = env.step(a_np)
            ep_return += float(reward)
            with torch.no_grad():
                _, (h, c) = rnn(z.unsqueeze(1), a.unsqueeze(1), (h, c))
            if terminated or truncated:
                break
        returns.append(ep_return)
        print(f"episode {ep+1}: return={ep_return:.1f}")
    env.close()

    if all_frames:
        imageio.mimwrite(args.out, all_frames, fps=30)
        # Light GIF for the README — every 3rd frame, smaller resolution.
        gif_frames = [f[::2, ::2] for f in all_frames[::3]]
        imageio.mimwrite(args.gif, gif_frames, duration=1 / 15, loop=0)
        print(f"video -> {args.out}\ngif   -> {args.gif}")
    print(f"mean return: {np.mean(returns):.1f}")


if __name__ == "__main__":
    main()
