"""Top-level driver: self-play → buffer → train_step → (eval/ckpt) → repeat.

One iteration = (one self-play game pushed to the buffer) + (some gradient
steps on samples from the buffer, once warmup is done).

Optional hooks:
  - eval_every > 0: every N iters, play eval_games games vs Stockfish at
    UCI_Elo=eval_stockfish_elo and append a row to history["evals"].
  - ckpt_dir + ckpt_every > 0: every N iters, save network + history to disk.
  - on_iter_end: callback (iter, network, history) — used by the cloud
    runner to push checkpoints to S3 without coupling the driver to AWS.

Returns a `history` dict with per-iteration game length, outcome, loss
components, and any eval rows — enough to plot a learning curve.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import torch

from .config import MuZeroConfig
from .networks import MuZeroNet
from .replay import ReplayBuffer
from .selfplay import play_game
from .train import train_step


@dataclass
class TrainResult:
    network: MuZeroNet
    buffer: ReplayBuffer
    history: dict


def train_loop(
    cfg: MuZeroConfig,
    *,
    n_iterations: int,
    train_steps_per_iter: int = 4,
    buffer_capacity: int = 200,
    warmup_games: int = 2,
    device: torch.device | None = None,
    log_every: int = 1,
    # Eval hook.
    eval_every: int = 0,
    eval_games: int = 20,
    eval_stockfish_elo: int = 1320,
    eval_sims: int | None = None,
    eval_max_plies: int = 200,
    # Checkpoint hook.
    ckpt_dir: str | None = None,
    ckpt_every: int = 0,
    # Iter-end callback (cloud uses this for S3 sync).
    on_iter_end: Callable[[int, MuZeroNet, dict], None] | None = None,
    # Time budget (seconds). Breaks out of the loop after this elapses;
    # `n_iterations` is the hard upper bound.
    time_budget_s: float | None = None,
) -> TrainResult:
    device = device if device is not None else torch.device("cpu")
    network = MuZeroNet(cfg).to(device)
    optim = torch.optim.Adam(
        network.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    buffer = ReplayBuffer(capacity=buffer_capacity, cfg=cfg)

    history: dict = {"games": [], "losses": [], "evals": []}
    start_t = time.time()

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    for it in range(n_iterations):
        t0 = time.time()
        game = play_game(network, cfg, device=device)
        buffer.push(game)
        game_time = time.time() - t0

        loss_stats: dict | None = None
        if len(buffer) >= warmup_games:
            t1 = time.time()
            agg = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "reward_loss": 0.0}
            for _ in range(train_steps_per_iter):
                batch = buffer.sample(cfg.batch_size, cfg.num_unroll_steps)
                batch = {k: v.to(device) for k, v in batch.items()}
                stats = train_step(network, optim, batch)
                for k in agg:
                    agg[k] += stats[k]
            for k in agg:
                agg[k] /= max(train_steps_per_iter, 1)
            agg["train_time"] = time.time() - t1
            loss_stats = agg

        history["games"].append({
            "iter": it,
            "length": game.length,
            "outcome_for_starter": game.z_per_ply[0] if game.length > 0 else 0.0,
            "game_time": game_time,
        })
        if loss_stats is not None:
            history["losses"].append({"iter": it, **loss_stats})

        if it % log_every == 0:
            outcome_str = {1.0: "W", -1.0: "L", 0.0: "D"}.get(
                game.z_per_ply[0] if game.length > 0 else 0.0, "?",
            )
            line = (
                f"[iter {it:3d}] len={game.length:3d} {outcome_str}  "
                f"game_time={game_time:5.1f}s  "
                f"buf={len(buffer):3d}({buffer.n_positions:5d} pos)"
            )
            if loss_stats is not None:
                line += (
                    f"  L={loss_stats['loss']:.4f}"
                    f"  pi={loss_stats['policy_loss']:.4f}"
                    f"  v={loss_stats['value_loss']:.4f}"
                    f"  r={loss_stats['reward_loss']:.4f}"
                )
            print(line, flush=True)

        # --- Eval ---
        if eval_every > 0 and (it + 1) % eval_every == 0:
            # Import lazily so local smoke runs don't need stockfish on PATH.
            from .eval import evaluate_vs_stockfish
            te = time.time()
            stats = evaluate_vs_stockfish(
                network, cfg, device,
                stockfish_elo=eval_stockfish_elo,
                n_games=eval_games,
                sims=eval_sims,
                max_plies=eval_max_plies,
            )
            eval_time = time.time() - te
            row = {
                "iter": it,
                "stockfish_elo": eval_stockfish_elo,
                "eval_sims": eval_sims if eval_sims is not None else cfg.num_simulations,
                **stats,
                "eval_time": eval_time,
            }
            history["evals"].append(row)
            print(
                f"[iter {it:3d} eval] vs SF UCI={eval_stockfish_elo} "
                f"games={eval_games} score={stats['score']:.3f} "
                f"W={stats['wins']} D={stats['draws']} L={stats['losses']} "
                f"({eval_time:.1f}s)",
                flush=True,
            )

        # --- Checkpoint ---
        if ckpt_dir is not None and ckpt_every > 0 and (it + 1) % ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"net_iter{it:03d}.pt")
            torch.save({
                "iter": it,
                "config": cfg.__dict__,
                "model": network.state_dict(),
                "history": history,
            }, ckpt_path)
            # Also maintain a "current.pt" symlink-equivalent for the auto-eval daemon.
            current_path = os.path.join(ckpt_dir, "current.pt")
            torch.save({
                "iter": it,
                "config": cfg.__dict__,
                "model": network.state_dict(),
                "history": history,
            }, current_path)
            print(f"  saved {ckpt_path}", flush=True)

        if on_iter_end is not None:
            on_iter_end(it, network, history)

        if time_budget_s is not None and (time.time() - start_t) >= time_budget_s:
            print(
                f"[time budget] {time_budget_s:.0f}s elapsed — exiting after iter {it}",
                flush=True,
            )
            break

    return TrainResult(network=network, buffer=buffer, history=history)
