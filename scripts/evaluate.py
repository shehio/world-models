"""Evaluate the trained agent over N episodes for a stable mean return.

CarRacing tracks are randomly generated each episode, so a single
episode is high-variance. Use this to get a number you can trust.
"""
from __future__ import annotations

import argparse

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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--random", action="store_true",
                        help="evaluate a random-action policy instead (baseline)")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()

    vae = VAE().to(device)
    vae.load_state_dict(torch.load(CFG.vae_path, map_location=device)["state_dict"])
    vae.eval()
    rnn = MDNRNN().to(device)
    rnn.load_state_dict(torch.load(CFG.rnn_path, map_location=device)["state_dict"])
    rnn.eval()

    if not args.random:
        ctrl = Controller().to(device)
        ctrl.load_state_dict(torch.load(CFG.controller_path, map_location=device)["state_dict"])
        ctrl.eval()

    env = make_env()
    rng = np.random.default_rng(args.seed)
    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        h, c = rnn.init_hidden(1, device)
        ep_return = 0.0
        for _ in range(args.max_steps):
            if args.random:
                a_np = np.array([rng.uniform(-1, 1), rng.uniform(0, 1) ** 0.7,
                                 rng.uniform(0, 1) ** 3.0], dtype=np.float32)
                obs, r, term, trunc, _ = env.step(a_np)
                ep_return += float(r)
                if term or trunc:
                    break
                continue
            frame = preprocess(obs)
            x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                z = vae.encode_mean(x)
                a = ctrl(z, h[-1])
            a_np = a.squeeze(0).cpu().numpy()
            obs, r, term, trunc, _ = env.step(a_np)
            ep_return += float(r)
            with torch.no_grad():
                _, (h, c) = rnn(z.unsqueeze(1), a.unsqueeze(1), (h, c))
            if term or trunc:
                break
        returns.append(ep_return)
        print(f"ep {ep+1:2d}: return={ep_return:+.1f}", flush=True)
    env.close()
    arr = np.array(returns)
    print(f"\n{'random' if args.random else 'agent '}  episodes={len(arr)}  "
          f"mean={arr.mean():+.1f}  std={arr.std():.1f}  min={arr.min():+.1f}  max={arr.max():+.1f}",
          flush=True)


if __name__ == "__main__":
    main()
