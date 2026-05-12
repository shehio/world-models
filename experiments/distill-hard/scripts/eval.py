"""Eval distilled network vs Stockfish at high sims (parallel).

Run from project root:
    uv run python scripts/eval.py --ckpt checkpoints/distilled_epoch029.pt \\
        --workers 5 --games-per-worker 20 --sims 800 --stockfish-elo 1320
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
     n_blocks, n_filters, sf_elo, sf_skill, sf_depth, max_plies) = args
    torch.set_num_threads(1)

    from wm_chess.arena import (
        network_policy, play_match, stockfish_engine, stockfish_policy,
    )
    from wm_chess.config import Config
    from wm_chess.network import AlphaZeroNet

    cfg = replace(Config(), n_res_blocks=n_blocks, n_filters=n_filters)
    net = AlphaZeroNet(cfg)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.eval()
    device = torch.device("cpu")
    agent_pol = network_policy(net, cfg, device, sims=sims, batch_size=batch_size)

    sf_kwargs = {"threads": 1}
    if sf_elo is not None:
        sf_kwargs["elo"] = sf_elo
    if sf_skill is not None:
        sf_kwargs["skill"] = sf_skill

    with stockfish_engine(**sf_kwargs) as eng:
        sf_pol = stockfish_policy(eng, depth=sf_depth)
        stats = play_match(agent_pol, sf_pol, n_games=n_games, max_plies=max_plies)
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
    p.add_argument("--games-per-worker", type=int, default=20)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=10)
    p.add_argument("--n-filters", type=int, default=128)
    p.add_argument("--stockfish-elo", type=int, default=1320)
    p.add_argument("--stockfish-skill", type=int, default=None)
    p.add_argument("--stockfish-depth", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=200)
    args = p.parse_args()

    mp.set_start_method("spawn", force=True)

    elo = args.stockfish_elo if args.stockfish_elo >= 0 else None
    print(f"agent: {args.ckpt} (sims={args.sims}, K={args.batch_size})")
    print(f"opponent: Stockfish elo={elo} skill={args.stockfish_skill} depth={args.stockfish_depth}")
    print(f"workers: {args.workers} × {args.games_per_worker} = {args.workers * args.games_per_worker} games", flush=True)

    worker_args = [
        (i, args.ckpt, args.games_per_worker, args.sims, args.batch_size,
         args.n_blocks, args.n_filters,
         elo, args.stockfish_skill, args.stockfish_depth, args.max_plies)
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
    score = (wins + 0.5 * draws) / n
    g = gap(score)

    se = math.sqrt(score * (1 - score) / n) if 0 < score < 1 else 0
    lo = max(0, score - 1.96 * se)
    hi = min(1, score + 1.96 * se)

    print(f"\n=== {n} games in {dt/60:.1f} min ===")
    print(f"W/D/L: {wins} / {draws} / {losses}")
    print(f"score: {score:.3f}   95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Elo gap: {g:+.0f}")
    if elo is not None and 0 < score < 1:
        elo_lo = elo + gap(lo); elo_hi = elo + gap(hi)
        print(f"Agent absolute Elo: {elo + g:.0f} [{elo_lo:.0f}, {elo_hi:.0f}]")


if __name__ == "__main__":
    main()
