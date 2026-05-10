"""Parallel evaluation of an agent vs random.

Each worker plays a fraction of the games independently with its own
network copy. Results aggregate across workers.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.multiprocessing as mp


def worker(args: tuple) -> dict:
    (worker_id, ckpt_path, n_games, sims, batch_size,
     n_blocks, n_filters, max_plies) = args

    torch.set_num_threads(1)

    from alphazero.arena import network_policy, play_match, random_policy
    from alphazero.config import Config
    from alphazero.network import AlphaZeroNet

    cfg = replace(Config(), n_res_blocks=n_blocks, n_filters=n_filters)
    net = AlphaZeroNet(cfg)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.eval()
    device = torch.device("cpu")
    agent_pol = network_policy(net, cfg, device, sims=sims, batch_size=batch_size)

    stats = play_match(agent_pol, random_policy, n_games=n_games, max_plies=max_plies)
    stats["worker_id"] = worker_id
    return stats


def gap(score):
    if score <= 0: return -float("inf")
    if score >= 1: return float("inf")
    return -400 * math.log10(1/score - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--games-per-worker", type=int, default=40)
    p.add_argument("--sims", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=10)
    p.add_argument("--n-filters", type=int, default=128)
    p.add_argument("--max-plies", type=int, default=200)
    args = p.parse_args()

    mp.set_start_method("spawn", force=True)

    print(f"agent ckpt: {args.ckpt}", flush=True)
    print(f"agent sims: {args.sims}, batch K: {args.batch_size}", flush=True)
    print(f"opponent: random", flush=True)
    print(f"workers: {args.workers} × {args.games_per_worker} games = {args.workers * args.games_per_worker} total", flush=True)

    worker_args = [
        (i, args.ckpt, args.games_per_worker, args.sims, args.batch_size,
         args.n_blocks, args.n_filters, args.max_plies)
        for i in range(args.workers)
    ]

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        all_stats = pool.map(worker, worker_args)
    dt = time.time() - t0

    wins = sum(s["wins"] for s in all_stats)
    draws = sum(s["draws"] for s in all_stats)
    losses = sum(s["losses"] for s in all_stats)
    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n if n else 0
    g = gap(score)

    se = math.sqrt(score * (1 - score) / n) if 0 < score < 1 else 0
    lo = max(0, score - 1.96 * se)
    hi = min(1, score + 1.96 * se)

    print(f"\n=== {n} games in {dt/60:.1f} min ===")
    print(f"W/D/L: {wins} / {draws} / {losses}")
    print(f"score: {score:.3f}   95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Elo gap to random: {g:+.0f}")


if __name__ == "__main__":
    main()
