"""Eval the 02c distilled network vs Stockfish UCI_Elo=1320.

Default mirrors 02b's eval but with the bigger 20×256 network and
configurable sims (overnight runbook will use 800 for apples-to-apples
with 02b's prior 1185-Elo result).

Run from 02c-distill-scaled root:
    uv run python scripts/eval.py \\
        --ckpt checkpoints/run01/distilled_epoch019.pt \\
        --workers 4 --games-per-worker 25 --sims 800 \\
        --n-blocks 20 --n-filters 256 \\
        --stockfish-elo 1320
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
     n_blocks, n_filters, sf_elo, sf_skill, sf_depth, max_plies,
     agent_device, progress_dir) = args
    torch.set_num_threads(1)

    from wm_chess.arena import (
        network_policy, play_match, stockfish_engine, stockfish_policy,
    )
    from wm_chess.config import Config
    from wm_chess.network import AlphaZeroNet

    import json as _json
    progress_path = (
        os.path.join(progress_dir, f".eval_progress_w{worker_id:02d}.json")
        if progress_dir else None
    )
    t_start = time.time()

    def _write_progress(games_done: int, phase: str) -> None:
        if not progress_path:
            return
        tmp = progress_path + ".tmp"
        with open(tmp, "w") as f:
            _json.dump({
                "worker_id": worker_id,
                "games_done": games_done,
                "games_total": n_games,
                "elapsed_s": time.time() - t_start,
                "phase": phase, "ts": time.time(),
            }, f)
        os.replace(tmp, progress_path)

    _write_progress(0, "start")

    # Pick the device for the agent's NN inference. CUDA wins for bigger nets
    # (20x256 forward is ~50x faster on A10G than on CPU). Multiple workers
    # share the GPU; CUDA serializes calls but small kernels overlap fine.
    if agent_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(agent_device)

    cfg = replace(Config(), n_res_blocks=n_blocks, n_filters=n_filters)
    net = AlphaZeroNet(cfg)
    # Accept two checkpoint shapes:
    #   - bare state_dict (what distill-soft/scripts/train.py writes)
    #   - {"net": state_dict, "opt": ..., "iter": ...} (what the self-play
    #     loop writes — selfplay_loop_mp.py)
    blob = torch.load(ckpt_path, map_location=device)
    if isinstance(blob, dict) and "net" in blob and isinstance(blob["net"], dict):
        state_dict = blob["net"]
    else:
        state_dict = blob
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    agent_pol = network_policy(net, cfg, device, sims=sims, batch_size=batch_size)

    sf_kwargs = {"threads": 1}
    if sf_elo is not None:
        sf_kwargs["elo"] = sf_elo
    if sf_skill is not None:
        sf_kwargs["skill"] = sf_skill

    # Play games one at a time so per-game progress is observable from outside.
    # play_match aggregates W/D/L; we re-aggregate the same way.
    agg = {"wins": 0, "draws": 0, "losses": 0, "games": 0}
    with stockfish_engine(**sf_kwargs) as eng:
        sf_pol = stockfish_policy(eng, depth=sf_depth)
        for g in range(n_games):
            stats = play_match(agent_pol, sf_pol, n_games=1, max_plies=max_plies)
            agg["wins"] += stats["wins"]
            agg["draws"] += stats["draws"]
            agg["losses"] += stats["losses"]
            agg["games"] += stats.get("games", 1)
            _write_progress(g + 1, "running")
    _write_progress(n_games, "done")
    agg["worker_id"] = worker_id
    return agg


def gap(score: float) -> float:
    if score <= 0: return -float("inf")
    if score >= 1: return float("inf")
    return -400 * math.log10(1/score - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games-per-worker", type=int, default=25)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=20)
    p.add_argument("--n-filters", type=int, default=256)
    p.add_argument("--stockfish-elo", type=int, default=1320)
    p.add_argument("--stockfish-skill", type=int, default=None)
    p.add_argument("--stockfish-depth", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--agent-device", default="auto",
                   help="device for the agent's NN inference: auto (cuda if available else cpu), cpu, cuda")
    p.add_argument("--progress-dir", default=None,
                   help="if set, each worker writes .eval_progress_wNN.json here every game (default: dir of --ckpt)")
    args = p.parse_args()

    mp.set_start_method("spawn", force=True)

    elo = args.stockfish_elo if args.stockfish_elo >= 0 else None
    print(f"agent: {args.ckpt} (sims={args.sims}, K={args.batch_size}, "
          f"net={args.n_blocks}x{args.n_filters})", flush=True)
    print(f"opponent: Stockfish elo={elo} skill={args.stockfish_skill} "
          f"depth={args.stockfish_depth}", flush=True)
    print(f"workers: {args.workers} × {args.games_per_worker} = "
          f"{args.workers * args.games_per_worker} games", flush=True)

    progress_dir = args.progress_dir or os.path.dirname(args.ckpt) or "."
    os.makedirs(progress_dir, exist_ok=True)
    # Clear stale per-worker progress from any prior eval.
    for f in os.listdir(progress_dir):
        if f.startswith(".eval_progress_w") and f.endswith(".json"):
            try: os.remove(os.path.join(progress_dir, f))
            except OSError: pass
    print(f"progress files: {progress_dir}/.eval_progress_wNN.json", flush=True)

    worker_args = [
        (i, args.ckpt, args.games_per_worker, args.sims, args.batch_size,
         args.n_blocks, args.n_filters,
         elo, args.stockfish_skill, args.stockfish_depth, args.max_plies,
         args.agent_device, progress_dir)
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
    print(f"Elo gap to opponent: {g:+.0f}")
    if elo is not None and 0 < score < 1:
        elo_lo = elo + gap(lo); elo_hi = elo + gap(hi)
        print(f"Agent absolute Elo (anchor {elo}): {elo + g:.0f} [{elo_lo:.0f}, {elo_hi:.0f}]")


if __name__ == "__main__":
    main()
