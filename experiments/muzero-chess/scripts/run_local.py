"""Local CPU smoke run for MuZero chess training.

Goal: prove the loop wires up end-to-end and total loss trends down on a
tiny config. Won't reach competitive Elo — that's the cloud stage.

Usage:
    uv run --project experiments/muzero-chess python scripts/run_local.py
"""
from __future__ import annotations

import argparse

import torch

from muzero_chess.config import MuZeroConfig
from muzero_chess.driver import train_loop


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--sims", type=int, default=10, help="MCTS sims per move (user-set local default)")
    p.add_argument("--train-steps", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=60)
    p.add_argument("--warmup-games", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--unroll", type=int, default=3)
    args = p.parse_args()

    cfg = MuZeroConfig(
        latent_channels=16,
        repr_n_res_blocks=2, repr_n_filters=16,
        dyn_n_res_blocks=2, dyn_n_filters=16,
        pred_n_res_blocks=2, pred_n_filters=16,
        num_simulations=args.sims,
        num_unroll_steps=args.unroll,
        batch_size=args.batch_size,
        max_plies=args.max_plies,
        temp_moves=20,
        lr=2e-3,
    )

    print(
        f"[config] latent={cfg.latent_channels}ch  sims={cfg.num_simulations}  "
        f"K={cfg.num_unroll_steps}  batch={cfg.batch_size}  max_plies={cfg.max_plies}",
        flush=True,
    )
    train_loop(
        cfg,
        n_iterations=args.iterations,
        train_steps_per_iter=args.train_steps,
        warmup_games=args.warmup_games,
        device=torch.device("cpu"),
    )


if __name__ == "__main__":
    main()
