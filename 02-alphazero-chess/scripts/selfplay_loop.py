"""Main training loop: alternate self-play and learning.

Run from the project root:
    uv run python scripts/selfplay_loop.py --iters 5 --games-per-iter 4 --sims 25
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Make src/ importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from alphazero.arena import network_policy, play_match, random_policy
from alphazero.config import Config
from alphazero.network import AlphaZeroNet, get_device
from alphazero.replay import ReplayBuffer
from alphazero.selfplay import play_game
from alphazero.train import train_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--games-per-iter", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--sims", type=int, default=25, help="MCTS sims per move during self-play")
    p.add_argument("--eval-sims", type=int, default=50, help="MCTS sims during arena eval")
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=200,
                   help="hard cap on game length (random play rarely mates; cap keeps wall-clock predictable)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="MCTS batch size (K parallel descents per network call). 1 = sequential reference.")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--device", default=None, help="cpu | mps | cuda; default auto")
    p.add_argument("--resume", default=None, help="path to a .pt file to resume from")
    args = p.parse_args()

    cfg = replace(Config(), sims_train=args.sims, sims_eval=args.eval_sims, max_plies=args.max_plies)
    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}")

    network = AlphaZeroNet(cfg).to(device)
    if args.resume:
        network.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    replay = ReplayBuffer(cfg.replay_capacity)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    for it in range(args.iters):
        # --- Self-play ---
        t0 = time.time()
        results = []
        for g in range(args.games_per_iter):
            samples, z, ply = play_game(network, cfg, device, sims=args.sims,
                                        batch_size=args.batch_size)
            replay.add_game(samples)
            results.append((z, ply))
            print(f"  iter {it} game {g}: z={z:+.0f} plies={ply}  buffer={len(replay)}")
        sp_dt = time.time() - t0

        # --- Train ---
        loss_acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        n_steps = 0
        if len(replay) >= cfg.batch_size:
            t0 = time.time()
            for _ in range(args.train_steps):
                batch = replay.sample(cfg.batch_size, device)
                losses = train_step(network, optimizer, batch)
                for k, v in losses.items():
                    loss_acc[k] += v
                n_steps += 1
            tr_dt = time.time() - t0
            print(
                f"  iter {it} train ({n_steps} steps, {tr_dt:.1f}s): "
                f"loss={loss_acc['loss']/n_steps:.3f} "
                f"pol={loss_acc['policy_loss']/n_steps:.3f} "
                f"val={loss_acc['value_loss']/n_steps:.3f}"
            )

        # --- Eval ---
        if (it + 1) % args.eval_every == 0:
            t0 = time.time()
            net_pol = network_policy(network, cfg, device, sims=args.eval_sims,
                                     batch_size=args.batch_size)
            stats = play_match(net_pol, random_policy, n_games=args.eval_games)
            ev_dt = time.time() - t0
            print(f"  iter {it} vs random ({ev_dt:.1f}s): {stats}")

            ckpt_path = os.path.join(args.ckpt_dir, f"net_iter{it:03d}.pt")
            torch.save(network.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path}")

        print(f"  iter {it} self-play wall: {sp_dt:.1f}s")


if __name__ == "__main__":
    main()
