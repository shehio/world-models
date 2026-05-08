"""Evolve the controller with CMA-ES.

Each candidate is a flat parameter vector. We load V and M once, freeze
them, and evaluate each candidate by running it in the real environment
for `cma_episodes_per_eval` episodes (capped at `cma_max_steps`).

The fitness is the mean episode return. CMA-ES updates a Gaussian search
distribution toward better candidates — no gradients through the env
required.
"""
from __future__ import annotations

import argparse
import time

import cma
import numpy as np
import torch

from world_models.config import CFG
from world_models.controller import Controller
from world_models.env import make_env, preprocess
from world_models.mdn_rnn import MDNRNN
from world_models.utils import pick_device, seed_everything
from world_models.vae import VAE


def evaluate(
    flat_params: np.ndarray,
    vae: VAE,
    rnn: MDNRNN,
    controller: Controller,
    device: torch.device,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> float:
    controller.set_flat(flat_params)
    controller.to(device).eval()
    env = make_env()
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        h, c = rnn.init_hidden(batch=1, device=device)
        ep_return = 0.0
        for _ in range(max_steps):
            frame = preprocess(obs)
            x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                z = vae.encode_mean(x)                            # (1, z_dim)
                h_flat = h[-1]                                    # (1, hidden) — last layer
                a = controller(z, h_flat)                         # (1, 3)
            a_np = a.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(a_np)
            ep_return += float(reward)
            with torch.no_grad():
                a_t = a.unsqueeze(1)                              # (1, 1, 3)
                z_t = z.unsqueeze(1)                              # (1, 1, z_dim)
                _, (h, c) = rnn(z_t, a_t, (h, c))
            if terminated or truncated:
                break
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=CFG.cma_pop)
    parser.add_argument("--generations", type=int, default=CFG.cma_generations)
    parser.add_argument("--episodes", type=int, default=CFG.cma_episodes_per_eval)
    parser.add_argument("--max-steps", type=int, default=CFG.cma_max_steps)
    parser.add_argument("--sigma", type=float, default=CFG.cma_init_sigma)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--resume", action="store_true",
                        help="seed the search from checkpoints/controller.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    print(f"device={device}")

    vae = VAE().to(device)
    vae.load_state_dict(torch.load(CFG.vae_path, map_location=device)["state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    rnn = MDNRNN().to(device)
    rnn.load_state_dict(torch.load(CFG.rnn_path, map_location=device)["state_dict"])
    rnn.eval()
    for p in rnn.parameters():
        p.requires_grad_(False)

    ctrl = Controller().to(device)
    print(f"controller params: {ctrl.num_params}", flush=True)

    best_score = -float("inf")
    if args.resume and CFG.controller_path.exists():
        ckpt = torch.load(CFG.controller_path, map_location=device)
        ctrl.load_state_dict(ckpt["state_dict"])
        # Anchor best_score so a single noisy gen can't overwrite a better checkpoint.
        best_score = float(ckpt.get("best_score", -float("inf")))
        print(f"resumed CMA-ES init from {CFG.controller_path}  (prior best_score={best_score:.1f})",
              flush=True)

    x0 = ctrl.get_flat()
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {"popsize": args.pop, "seed": args.seed, "verbose": -9})
    best_flat = x0.copy()
    t0 = time.time()
    for gen in range(args.generations):
        candidates = es.ask()
        # CMA-ES minimizes; we want to maximize return, so we negate.
        scores: list[float] = []
        for i, flat in enumerate(candidates):
            r = evaluate(flat, vae, rnn, ctrl, device,
                         n_episodes=args.episodes, max_steps=args.max_steps,
                         seed=args.seed + gen * 1000 + i)
            scores.append(r)
            print(f"  gen {gen+1} cand {i+1}/{len(candidates)}: r={r:+.1f}", flush=True)
        es.tell(candidates, [-s for s in scores])
        gen_best = max(scores)
        gen_mean = float(np.mean(scores))
        if gen_best > best_score:
            best_score = gen_best
            best_flat = candidates[int(np.argmax(scores))]
            ctrl.set_flat(best_flat)
            torch.save({"state_dict": ctrl.state_dict(), "best_score": best_score}, CFG.controller_path)
        elapsed = time.time() - t0
        print(f"gen {gen+1}/{args.generations} | mean={gen_mean:+.1f}  best_gen={gen_best:+.1f}  best_ever={best_score:+.1f}  t={elapsed:.0f}s", flush=True)

    ctrl.set_flat(best_flat)
    torch.save({"state_dict": ctrl.state_dict(), "best_score": best_score}, CFG.controller_path)
    print(f"done. best return = {best_score:.1f}  -> {CFG.controller_path}", flush=True)


if __name__ == "__main__":
    main()
