"""Supervised distillation training for Go.

Local usage (9x9 demo):
    uv run --project experiments/distill-go python experiments/distill-go/scripts/train.py \\
        --data /tmp/go9_merged.npz --board-size 9 --n-input-planes 4 \\
        --epochs 10 --n-blocks 4 --n-filters 64 --ckpt-dir /tmp/go9_ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from distill_go.config import GoConfig
from distill_go.network import AlphaZeroGoNet, get_device
from distill_go.train import GoMultipvDataset, train_step


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--n-input-planes", type=int, default=4, choices=[4, 17])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--n-filters", type=int, default=64)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--max-positions", type=int, default=None)
    p.add_argument("--hard-targets", action="store_true")
    args = p.parse_args()

    cfg = GoConfig(
        board_size=args.board_size,
        n_input_planes=args.n_input_planes,
        n_res_blocks=args.n_blocks,
        n_filters=args.n_filters,
    )
    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}", flush=True)

    network = AlphaZeroGoNet(cfg).to(device)
    n_params = sum(p.numel() for p in network.parameters())
    print(
        f"network: board={cfg.board_size}  planes={cfg.n_input_planes}  "
        f"blocks={cfg.n_res_blocks}  filters={cfg.n_filters}  "
        f"params={n_params:,}",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"loading {args.data} ...", flush=True)
    dataset = GoMultipvDataset(args.data, max_rows=args.max_positions)
    n = len(dataset)
    K = dataset.K
    print(f"  {n:,} positions, multipv K={K}", flush=True)
    # Sanity-check shapes match the network config.
    actual_planes = dataset.states.shape[1]
    actual_size = dataset.states.shape[2]
    if actual_planes != cfg.n_input_planes:
        raise SystemExit(
            f"--n-input-planes={cfg.n_input_planes} but dataset has "
            f"{actual_planes} planes — set --n-input-planes={actual_planes}"
        )
    if actual_size != cfg.board_size:
        raise SystemExit(
            f"--board-size={cfg.board_size} but dataset has "
            f"{actual_size}x{actual_size} boards"
        )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    history: list[dict] = []
    history_path = os.path.join(args.ckpt_dir, "train_history.json")

    overall_t0 = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        idxs = rng.permutation(n)
        n_batches = n // args.batch_size
        if n_batches == 0:
            raise SystemExit(
                f"dataset has {n} positions but batch_size={args.batch_size}"
            )
        loss_acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                    "top1_acc": 0.0, "topk_acc": 0.0, "tgt_entropy": 0.0}

        for b in range(n_batches):
            sel = idxs[b * args.batch_size:(b + 1) * args.batch_size]
            batch = (
                dataset.states[sel],
                dataset.multipv_indices[sel],
                dataset.multipv_logprobs[sel],
                dataset.moves[sel],
                dataset.zs[sel],
            )
            losses = train_step(
                network, optimizer, batch, device,
                n_actions=cfg.policy_size,
                value_weight=args.value_weight,
                hard_targets=args.hard_targets,
            )
            for k, v in losses.items():
                loss_acc[k] += v

        dt = time.time() - t0
        n_b = max(n_batches, 1)
        epoch_summary = {
            "epoch": epoch,
            "epoch_time_s": dt,
            "elapsed_s": time.time() - overall_t0,
            "loss": loss_acc["loss"] / n_b,
            "policy_loss": loss_acc["policy_loss"] / n_b,
            "value_loss": loss_acc["value_loss"] / n_b,
            "top1_acc": loss_acc["top1_acc"] / n_b,
            "topk_acc": loss_acc["topk_acc"] / n_b,
            "tgt_entropy": loss_acc["tgt_entropy"] / n_b,
        }
        history.append(epoch_summary)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(
            f"epoch {epoch:02d} ({dt:.1f}s): "
            f"loss={epoch_summary['loss']:.3f} "
            f"pol={epoch_summary['policy_loss']:.3f} "
            f"val={epoch_summary['value_loss']:.4f} "
            f"top1={epoch_summary['top1_acc']:.3f} "
            f"topK={epoch_summary['topk_acc']:.3f} "
            f"H(tgt)={epoch_summary['tgt_entropy']:.2f}",
            flush=True,
        )
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.ckpt_dir, f"distilled_epoch{epoch:03d}.pt")
            torch.save(network.state_dict(), ckpt)
            print(f"  saved {ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
