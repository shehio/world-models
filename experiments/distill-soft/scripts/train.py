"""Supervised distillation training with multipv-soft-targets.

Defaults to a 20×256 ResNet (paper-sized) — much bigger than 02b's
10×128. Pair this with the bigger soft-target dataset for the experiment.

Run from 02c-distill-scaled root:
    uv run python scripts/train.py \\
        --data data/sf_d15_mpv8_4000g.npz \\
        --epochs 20 --device mps \\
        --n-blocks 20 --n-filters 256 \\
        --ckpt-dir checkpoints/run01
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet, get_device
from distill_soft.train_supervised import MultipvDataset, train_step


def _sync_to_s3(local_path: str, s3_base: str) -> None:
    """Fire-and-forget S3 upload. Logs failure but never raises —
    the goal is to make checkpoints durable, not to crash training
    if S3 has a hiccup."""
    if not s3_base:
        return
    target = s3_base.rstrip("/") + "/" + os.path.basename(local_path)
    try:
        subprocess.run(
            ["aws", "s3", "cp", local_path, target, "--no-progress"],
            check=True, capture_output=True, text=True, timeout=300,
        )
        print(f"  s3 sync: {target}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  s3 sync FAILED for {local_path}: {e.stderr}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  s3 sync TIMEOUT for {local_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-blocks", type=int, default=20,
                   help="number of ResNet blocks (paper: 20)")
    p.add_argument("--n-filters", type=int, default=256,
                   help="channels per block (paper: 256)")
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-dir", default="checkpoints/run01")
    p.add_argument("--init-from", default=None,
                   help="optional .pt to warm-start from")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--hard-targets", action="store_true",
                   help="use standard CE against the actually-played move "
                        "instead of soft-CE over the multipv distribution. "
                        "Ablation: same data, same net, different target shape.")
    p.add_argument("--s3-ckpt-base", default=None,
                   help="S3 URI prefix (e.g. s3://bucket/.../checkpoints/net-NxN/run-id). "
                        "If set, each saved .pt and updated train_history.json is "
                        "synced here immediately. Required for crash-safe overnight runs.")
    p.add_argument("--max-positions", type=int, default=None,
                   help="cap dataset to first N positions (deterministic slice, "
                        "shuffling is per-epoch anyway). Lets large datasets like "
                        "the d10 30M-position one fit in instance RAM.")
    p.add_argument("--in-ram", action="store_true",
                   help="materialize the dataset into RAM (np.array). Default is "
                        "memmap (lazy disk-backed). Use this when the instance has "
                        "enough RAM — batch fetch is ~10x faster than memmap "
                        "random-page-reads.")
    p.add_argument("--amp", action="store_true",
                   help="enable bfloat16 autocast for forward+loss. ~2x faster on "
                        "Ampere/Ada tensor cores (A10G/L4/A100/H100). bf16 shares "
                        "fp32's exponent range so no GradScaler is needed.")
    p.add_argument("--compile", action="store_true",
                   help="wrap the network in torch.compile() — fuses ops + uses "
                        "Inductor. Adds a 1-2 min warmup on the first batch, "
                        "then ~1.3-1.5x faster steady-state.")
    p.add_argument("--start-epoch", type=int, default=0,
                   help="resume the for-loop at this epoch number. Pair with "
                        "--init-from when continuing a previous run so the "
                        "saved ckpt filenames keep monotonically increasing "
                        "(distilled_epochNNN.pt). Without this, a resumed "
                        "run overwrites the source run's early-epoch ckpts.")
    args = p.parse_args()

    cfg = replace(Config(), n_res_blocks=args.n_blocks, n_filters=args.n_filters)
    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}", flush=True)

    network = AlphaZeroNet(cfg).to(device)
    if args.init_from:
        network.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"warm-started from {args.init_from}", flush=True)

    # torch.compile must happen AFTER load_state_dict — compiling first
    # would wrap the model and load_state_dict's key names would be off
    # (compiled models prefix params with `_orig_mod.`).
    if args.compile:
        print("compiling network with torch.compile() ...", flush=True)
        network = torch.compile(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    print(f"amp={args.amp}  compile={args.compile}  batch_size={args.batch_size}",
          flush=True)

    n_params = sum(p.numel() for p in network.parameters())
    print(f"network: {args.n_blocks} blocks × {args.n_filters} ch = {n_params:,} params",
          flush=True)

    print(f"loading {args.data} ...", flush=True)
    # Pass max_positions through so the extractor only writes the rows
    # we'll actually train on — at 30M positions the states array
    # uncompresses to ~150 GB; truncating up front saves disk + time.
    dataset = MultipvDataset(args.data, max_rows=args.max_positions)
    if args.max_positions is not None and dataset.states.shape[0] > args.max_positions:
        n_full = dataset.states.shape[0]
        n_keep = args.max_positions
        # Slice all arrays to first N — the per-epoch rng.permutation already
        # randomizes the within-epoch order, so a deterministic head-slice
        # keeps a consistent subset across runs (good for ablations).
        dataset.states = dataset.states[:n_keep]
        dataset.moves = dataset.moves[:n_keep]
        dataset.multipv_indices = dataset.multipv_indices[:n_keep]
        dataset.multipv_logprobs = dataset.multipv_logprobs[:n_keep]
        dataset.zs = dataset.zs[:n_keep]
        print(f"  subsampled: {n_full:,} -> {n_keep:,} positions", flush=True)
    if args.in_ram:
        print("  materializing dataset into RAM (--in-ram) ...", flush=True)
        t_load = time.time()
        dataset.states = np.array(dataset.states)
        dataset.moves = np.array(dataset.moves)
        dataset.multipv_indices = np.array(dataset.multipv_indices)
        dataset.multipv_logprobs = np.array(dataset.multipv_logprobs)
        dataset.zs = np.array(dataset.zs)
        rss_gb = (dataset.states.nbytes + dataset.moves.nbytes
                  + dataset.multipv_indices.nbytes
                  + dataset.multipv_logprobs.nbytes
                  + dataset.zs.nbytes) / 1e9
        print(f"  in RAM: {rss_gb:.1f} GB  ({time.time()-t_load:.1f}s)", flush=True)
    n = len(dataset)
    K = dataset.K
    print(f"  {n:,} positions, multipv K={K}", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    # If resuming mid-run, advance the shared RNG past the epochs that
    # were already trained — keeps the per-epoch shuffles identical to
    # what an uninterrupted run would have produced.
    for _ in range(args.start_epoch):
        rng.permutation(1)

    # Observability: per-epoch metrics JSON history + live training progress file.
    history_path = os.path.join(args.ckpt_dir, "train_history.json")
    progress_path = os.path.join(args.ckpt_dir, ".train_progress.json")
    history: list[dict] = []

    def _write_train_progress(payload: dict) -> None:
        import json
        tmp = progress_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, progress_path)

    def _gpu_stats() -> dict:
        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_mem_alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "gpu_max_mem_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
        }

    def _grad_norm() -> float:
        # L2 norm of all gradients. Diagnostic for vanishing / exploding gradients.
        total = 0.0
        for p in network.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm(2).item()) ** 2
        return total ** 0.5

    print(f"[obs] history -> {history_path}", flush=True)
    print(f"[obs] live progress -> {progress_path}", flush=True)
    overall_t0 = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        t0 = time.time()
        idxs = rng.permutation(n)
        n_batches = n // args.batch_size
        loss_acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                    "top1_acc": 0.0, "topk_acc": 0.0, "tgt_entropy": 0.0}

        # Per-batch sub-epoch progress: prints every PROGRESS_EVERY batches so
        # we can see learning happen in real time and detect stalls.
        progress_every = max(1, n_batches // 5)  # ~5 prints per epoch
        grad_norm_at_print = 0.0

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
                value_weight=args.value_weight,
                hard_targets=args.hard_targets,
                use_amp=args.amp,
            )
            for k, v in losses.items():
                loss_acc[k] += v

            if (b + 1) % progress_every == 0 or b == n_batches - 1:
                grad_norm_at_print = _grad_norm()
                cur = b + 1
                print(f"  [b{cur:>4}/{n_batches}] "
                      f"loss={losses['loss']:.3f} "
                      f"pol={losses['policy_loss']:.3f} "
                      f"val={losses['value_loss']:.4f} "
                      f"top1={losses['top1_acc']:.3f} "
                      f"topK={losses['topk_acc']:.3f} "
                      f"gnorm={grad_norm_at_print:.2f}",
                      flush=True)

            # Live progress for external watchers (every batch is fine, small write).
            if (b + 1) % 50 == 0 or b == n_batches - 1:
                _write_train_progress({
                    "epoch": epoch, "batch": b + 1, "n_batches": n_batches,
                    "epochs_total": args.epochs,
                    "elapsed_s": time.time() - overall_t0,
                    "epoch_elapsed_s": time.time() - t0,
                    "loss": float(losses["loss"]),
                    "policy_loss": float(losses["policy_loss"]),
                    "value_loss": float(losses["value_loss"]),
                    "top1_acc": float(losses["top1_acc"]),
                    "topk_acc": float(losses["topk_acc"]),
                    "grad_norm": grad_norm_at_print,
                    "ts": time.time(),
                    **_gpu_stats(),
                })

        n_b = max(n_batches, 1)
        dt = time.time() - t0
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
            "final_grad_norm": grad_norm_at_print,
            "lr": optimizer.param_groups[0]["lr"],
            **_gpu_stats(),
        }
        history.append(epoch_summary)
        import json as _json
        with open(history_path, "w") as f:
            _json.dump(history, f, indent=2)
        # Per-epoch S3 sync of history so we can monitor live runs externally.
        _sync_to_s3(history_path, args.s3_ckpt_base)
        print(f"epoch {epoch:02d} ({dt:.1f}s): "
              f"loss={epoch_summary['loss']:.3f} "
              f"pol={epoch_summary['policy_loss']:.3f} "
              f"val={epoch_summary['value_loss']:.4f} "
              f"top1={epoch_summary['top1_acc']:.3f} "
              f"topK={epoch_summary['topk_acc']:.3f} "
              f"H(tgt)={epoch_summary['tgt_entropy']:.2f} "
              f"gnorm={epoch_summary['final_grad_norm']:.2f}",
              flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.ckpt_dir, f"distilled_epoch{epoch:03d}.pt")
            # torch.compile wraps the model and prefixes state-dict keys
            # with `_orig_mod.`. Unwrap so the saved ckpt is loadable by a
            # plain (un-compiled) model — that's what eval.py expects.
            state_to_save = (network._orig_mod.state_dict()
                             if args.compile else network.state_dict())
            torch.save(state_to_save, ckpt)
            print(f"  saved {ckpt}", flush=True)
            # Immediate S3 upload — checkpoint is durable before the next
            # epoch even starts. If the pod dies after epoch 5, we keep
            # the epoch-5 ckpt instead of losing everything.
            _sync_to_s3(ckpt, args.s3_ckpt_base)


if __name__ == "__main__":
    main()
