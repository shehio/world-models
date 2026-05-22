"""Multi-worker parallel KataGo datagen — the bridge from spike to cluster.

Each worker spawns its own KataGo subprocess (so KataGo concurrency = N_WORKERS)
and writes its own chunks. Mirrors the structure of
`distill_soft.stockfish_data.generate_dataset_parallel` but with KataGo.

This is what the wm-go container's entrypoint would call. Validates the
parallel-orchestration question end-to-end before paying for the cluster
build-out.

Usage:
    KATAGO_BIN=/path/to/katago KATAGO_MODEL=/path/to/model.bin.gz \
    KATAGO_CONFIG=/path/to/analysis.cfg \
    uv run --project experiments/distill-go-spike python \
        experiments/distill-go-spike/scripts/run_spike_parallel.py \
        --workers 4 --games-per-worker 5 --board-size 9 --visits 100 \
        --output-dir /tmp/go_parallel

Writes worker_NN_chunk_NNNN.npz files in `output-dir` matching the chess
chunk schema, so `wm_chess/scripts/merge_chunks.py` will fold them in
without modification.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go_spike import N_INPUT_PLANES, play_one_game


def _save_chunk(out_dir: Path, worker_id: int, chunk_idx: int,
                games_data: list[dict]) -> Path:
    """Concatenate one chunk_size worth of games into a single .npz."""
    states = np.concatenate([g["states"] for g in games_data], axis=0)
    moves = np.concatenate([g["moves"] for g in games_data], axis=0)
    mpv_idx = np.concatenate([g["multipv_indices"] for g in games_data], axis=0)
    mpv_logp = np.concatenate([g["multipv_logprobs"] for g in games_data], axis=0)
    zs = np.concatenate([g["zs"] for g in games_data], axis=0)
    out = out_dir / f"worker_{worker_id:02d}_chunk_{chunk_idx:04d}.npz"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, states=states, moves=moves,
        multipv_indices=mpv_idx, multipv_logprobs=mpv_logp, zs=zs,
    )
    os.replace(tmp, out)
    return out


def _worker(args_tuple: tuple) -> dict:
    """One worker: play `n_games` games, write chunks at every `chunk_size`."""
    (worker_id, n_games, chunk_size, board_size, visits, top_k,
     temperature, komi, rules, output_dir,
     katago_bin, katago_model, katago_config) = args_tuple
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf: list[dict] = []
    chunk_idx = 0
    t0 = time.time()
    n_positions = 0
    for g in range(n_games):
        result = play_one_game(
            katago_bin, katago_model,
            config_path=katago_config,
            board_size=board_size,
            visits=visits, top_k=top_k,
            temperature=temperature, komi=komi, rules=rules,
        )
        buf.append(result)
        n_positions += int(result["states"].shape[0])
        if (g + 1) % chunk_size == 0 or (g + 1) == n_games:
            _save_chunk(out_dir, worker_id, chunk_idx, buf)
            chunk_idx += 1
            buf = []
    return {
        "worker_id": worker_id, "n_games": n_games,
        "n_positions": n_positions, "chunks_written": chunk_idx,
        "elapsed_s": time.time() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games-per-worker", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=5)
    p.add_argument("--board-size", type=int, default=9, choices=[9, 13, 19])
    p.add_argument("--visits", type=int, default=100)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--rules", default="tromp-taylor")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/go_parallel"))
    args = p.parse_args()

    katago_bin = os.environ["KATAGO_BIN"]
    katago_model = os.environ["KATAGO_MODEL"]
    katago_config = os.environ["KATAGO_CONFIG"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[parallel] {args.workers} workers × {args.games_per_worker} games "
          f"on size {args.board_size}, chunk_size={args.chunk_size}", flush=True)
    t0 = time.time()
    worker_args = [
        (wid, args.games_per_worker, args.chunk_size, args.board_size,
         args.visits, args.top_k, args.temperature, args.komi, args.rules,
         str(args.output_dir), katago_bin, katago_model, katago_config)
        for wid in range(args.workers)
    ]
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_worker, worker_args)
    elapsed = time.time() - t0

    total_games = sum(r["n_games"] for r in results)
    total_positions = sum(r["n_positions"] for r in results)
    total_chunks = sum(r["chunks_written"] for r in results)
    print(f"[parallel] done in {elapsed:.1f}s — "
          f"{total_games} games, {total_positions} positions, "
          f"{total_chunks} chunks across {args.workers} workers", flush=True)
    print(f"[parallel] rate: {total_games / elapsed * 3600:.0f} games/hr aggregate",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
