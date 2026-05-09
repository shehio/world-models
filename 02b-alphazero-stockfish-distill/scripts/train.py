"""Supervised training on Stockfish self-play data.

Run from project root:
    uv run python scripts/train.py --data data/stockfish_games.npz --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from azdistill.config import Config
from azdistill.network import AlphaZeroNet, get_device
from azdistill.train_supervised import StockfishDataset, train_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/stockfish_games.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-blocks", type=int, default=10)
    p.add_argument("--n-filters", type=int, default=128)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--init-from", default=None,
                   help="optional .pt to warm-start from (else random init)")
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()

    cfg = replace(Config(), n_res_blocks=args.n_blocks, n_filters=args.n_filters)
    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}")

    network = AlphaZeroNet(cfg).to(device)
    if args.init_from:
        network.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"warm-started from {args.init_from}")
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in network.parameters())
    print(f"network: {args.n_blocks} blocks × {args.n_filters} ch = {n_params:,} params")

    print(f"loading {args.data} ...")
    dataset = StockfishDataset(args.data)
    n = len(dataset)
    print(f"  {n:,} positions")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Simple shuffled-index minibatch loop.
    rng = np.random.RandomState(0)

    for epoch in range(args.epochs):
        t0 = time.time()
        idxs = rng.permutation(n)
        n_batches = n // args.batch_size
        loss_acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "top1_acc": 0.0}
        for b in range(n_batches):
            sel = idxs[b * args.batch_size:(b + 1) * args.batch_size]
            states = dataset.states[sel]
            moves = dataset.moves[sel]
            zs = dataset.zs[sel]
            losses = train_step(
                network, optimizer, (states, moves, zs), device,
                value_weight=args.value_weight,
            )
            for k, v in losses.items():
                loss_acc[k] += v
        n_b = max(n_batches, 1)
        dt = time.time() - t0
        print(f"epoch {epoch:02d} ({dt:.1f}s): "
              f"loss={loss_acc['loss']/n_b:.3f} "
              f"pol={loss_acc['policy_loss']/n_b:.3f} "
              f"val={loss_acc['value_loss']/n_b:.4f} "
              f"top1={loss_acc['top1_acc']/n_b:.3f}",
              flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.ckpt_dir, f"distilled_epoch{epoch:03d}.pt")
            torch.save(network.state_dict(), ckpt)
            print(f"  saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
