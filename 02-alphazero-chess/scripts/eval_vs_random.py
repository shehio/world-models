"""Standalone arena eval: load a checkpoint, play N games vs random."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from alphazero.arena import network_policy, play_match, random_policy
from alphazero.config import Config
from alphazero.network import AlphaZeroNet, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = Config()
    device = torch.device(args.device) if args.device else get_device()
    network = AlphaZeroNet(cfg).to(device)
    network.load_state_dict(torch.load(args.ckpt, map_location=device))
    network.eval()

    net_pol = network_policy(network, cfg, device, sims=args.sims)
    stats = play_match(net_pol, random_policy, n_games=args.games)
    print(stats)


if __name__ == "__main__":
    main()
