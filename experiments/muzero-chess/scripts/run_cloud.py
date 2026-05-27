"""Cloud entry point for MuZero chess training.

Paper-ish config (64-channel latent, 5/3/2 residual stacks, sims=400),
eval vs Stockfish UCI every N iters, periodic checkpoint + S3 sync.

Designed to be invoked from infra-eks/entrypoint-muzero-chess.sh inside
the wm-chess-gpu image. Env vars override CLI defaults so the entrypoint
can wire hparams without touching this file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import torch

from muzero_chess.config import MuZeroConfig
from muzero_chess.driver import train_loop


def _s3_sync(local_dir: str, s3_uri: str) -> None:
    """One-shot aws s3 sync. Logs failures but doesn't raise — the next
    sync will pick up missed files, and we don't want a transient S3
    blip to take down a 24h training run."""
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, s3_uri, "--no-progress"],
            check=False, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"[s3 sync] timed out syncing {local_dir} → {s3_uri}", flush=True)


def _env_default(name: str, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


def main() -> None:
    p = argparse.ArgumentParser()
    # Network shape — paper-ish defaults that fit comfortably on a single L4.
    p.add_argument("--latent-channels", type=int,
                   default=_env_default("LATENT_CHANNELS", 64, int))
    p.add_argument("--repr-blocks", type=int,
                   default=_env_default("REPR_BLOCKS", 5, int))
    p.add_argument("--repr-filters", type=int,
                   default=_env_default("REPR_FILTERS", 64, int))
    p.add_argument("--dyn-blocks", type=int,
                   default=_env_default("DYN_BLOCKS", 3, int))
    p.add_argument("--dyn-filters", type=int,
                   default=_env_default("DYN_FILTERS", 64, int))
    p.add_argument("--pred-blocks", type=int,
                   default=_env_default("PRED_BLOCKS", 2, int))
    p.add_argument("--pred-filters", type=int,
                   default=_env_default("PRED_FILTERS", 64, int))
    # MCTS + training.
    p.add_argument("--sims", type=int,
                   default=_env_default("SIMS", 400, int),
                   help="MCTS simulations per move during self-play + eval (per user)")
    p.add_argument("--mcts-batch-size", type=int,
                   default=_env_default("MCTS_BATCH_SIZE", 8, int),
                   help="parallel descents per network call (virtual-loss MCTS)")
    p.add_argument("--mcts-top-k", type=int,
                   default=_env_default("MCTS_TOP_K", 32, int),
                   help="non-root expansion fan-out (0 = paper-faithful 4672)")
    p.add_argument("--unroll", type=int,
                   default=_env_default("UNROLL", 5, int))
    p.add_argument("--batch-size", type=int,
                   default=_env_default("BATCH_SIZE", 128, int))
    p.add_argument("--lr", type=float,
                   default=_env_default("LR", 1e-3, float))
    p.add_argument("--max-plies", type=int,
                   default=_env_default("MAX_PLIES", 200, int))
    p.add_argument("--temp-moves", type=int,
                   default=_env_default("TEMP_MOVES", 30, int))
    # Loop control.
    p.add_argument("--iterations", type=int,
                   default=_env_default("ITERATIONS", 200, int))
    p.add_argument("--train-steps", type=int,
                   default=_env_default("TRAIN_STEPS", 16, int))
    p.add_argument("--warmup-games", type=int,
                   default=_env_default("WARMUP_GAMES", 4, int))
    p.add_argument("--buffer-capacity", type=int,
                   default=_env_default("BUFFER_CAPACITY", 200, int))
    p.add_argument("--time-budget", type=int,
                   default=_env_default("TIME_BUDGET", 12 * 3600, int),
                   help="seconds; the loop exits when this elapses")
    # Eval.
    p.add_argument("--eval-every", type=int,
                   default=_env_default("EVAL_EVERY", 5, int),
                   help="run Stockfish-UCI eval every N iters (user-requested: 5)")
    p.add_argument("--eval-games", type=int,
                   default=_env_default("EVAL_GAMES", 20, int))
    p.add_argument("--eval-stockfish-elo", type=int,
                   default=_env_default("EVAL_STOCKFISH_ELO", 1320, int))
    p.add_argument("--eval-sims", type=int,
                   default=_env_default("EVAL_SIMS", 0, int),
                   help="0 = same as training sims")
    # Checkpointing + S3.
    p.add_argument("--ckpt-dir", default=_env_default("CKPT_DIR", "/work/checkpoints_muzero"))
    p.add_argument("--ckpt-every", type=int,
                   default=_env_default("CKPT_EVERY", 1, int))
    p.add_argument("--s3-base", default=_env_default("S3_BASE", ""))
    p.add_argument("--device", default=_env_default("DEVICE", ""))
    args = p.parse_args()

    cfg = MuZeroConfig(
        latent_channels=args.latent_channels,
        repr_n_res_blocks=args.repr_blocks, repr_n_filters=args.repr_filters,
        dyn_n_res_blocks=args.dyn_blocks, dyn_n_filters=args.dyn_filters,
        pred_n_res_blocks=args.pred_blocks, pred_n_filters=args.pred_filters,
        num_simulations=args.sims,
        mcts_batch_size=args.mcts_batch_size,
        mcts_top_k=args.mcts_top_k,
        num_unroll_steps=args.unroll,
        batch_size=args.batch_size,
        lr=args.lr,
        max_plies=args.max_plies,
        temp_moves=args.temp_moves,
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[muzero-chess cloud] device={device}", flush=True)
    print(f"[config] {cfg}", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    # Pin a metadata JSON so the eventual eval daemons know the arch.
    with open(os.path.join(args.ckpt_dir, "run_metadata.json"), "w") as f:
        json.dump({
            "kind": "muzero-chess",
            "config": cfg.__dict__,
            "args": vars(args),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    eval_sims = args.eval_sims if args.eval_sims > 0 else None

    def on_iter_end(it: int, network, history: dict) -> None:
        # Always write the latest history JSON locally — cheap.
        with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
            json.dump(history, f, default=float)
        # Mirror checkpoint dir to S3 every ckpt_every iters.
        if args.s3_base and args.ckpt_every > 0 and (it + 1) % args.ckpt_every == 0:
            print(f"[s3] sync {args.ckpt_dir} → {args.s3_base}", flush=True)
            _s3_sync(args.ckpt_dir, args.s3_base)

    train_loop(
        cfg,
        n_iterations=args.iterations,
        train_steps_per_iter=args.train_steps,
        buffer_capacity=args.buffer_capacity,
        warmup_games=args.warmup_games,
        device=device,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_stockfish_elo=args.eval_stockfish_elo,
        eval_sims=eval_sims,
        eval_max_plies=args.max_plies,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        on_iter_end=on_iter_end,
        time_budget_s=args.time_budget,
    )

    # Final sync.
    if args.s3_base:
        print(f"[s3] final sync {args.ckpt_dir} → {args.s3_base}", flush=True)
        _s3_sync(args.ckpt_dir, args.s3_base)


if __name__ == "__main__":
    main()
