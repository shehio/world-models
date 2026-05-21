"""Multi-worker parallel KataGo datagen with chunked output + resume.

Mirrors `wm_chess.stockfish_data.generate_dataset_parallel`:
  - N workers, each owns its KataGo subprocess
  - Each worker writes worker_NN_chunk_NNNN.npz every `chunk_size` games
  - chunk_idx resumes past existing files in the output dir (matches the
    chess datagen's resume contract — see `feedback_datagen_resume`)
  - Filenames + schema are byte-compatible with `wm_chess/scripts/merge_chunks.py`

The cluster entrypoint (`infra-eks/entrypoint.sh` analogue for `wm-go`)
would invoke this via `scripts/generate_data.py`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import numpy as np

from .katago_data import play_one_game


def _save_chunk(
    out_dir: Path,
    worker_id: int,
    chunk_idx: int,
    games_data: list[dict],
) -> Path:
    states = np.concatenate([g["states"] for g in games_data], axis=0)
    moves = np.concatenate([g["moves"] for g in games_data], axis=0)
    mpv_idx = np.concatenate([g["multipv_indices"] for g in games_data], axis=0)
    mpv_logp = np.concatenate([g["multipv_logprobs"] for g in games_data], axis=0)
    zs = np.concatenate([g["zs"] for g in games_data], axis=0)
    out = out_dir / f"worker_{worker_id:02d}_chunk_{chunk_idx:04d}.npz"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        states=states,
        moves=moves,
        multipv_indices=mpv_idx,
        multipv_logprobs=mpv_logp,
        zs=zs,
    )
    os.replace(tmp, out)
    return out


def _next_chunk_idx_for_worker(out_dir: Path, worker_id: int) -> int:
    """Resume: find the highest chunk_idx already on disk for this worker.

    Returns the next idx to write. Matches chess datagen resume semantics
    so a reclaimed/restarted pod doesn't overwrite existing chunks.
    """
    pattern = re.compile(rf"^worker_{worker_id:02d}_chunk_(\d+)\.npz$")
    max_idx = -1
    if out_dir.exists():
        for f in out_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
    return max_idx + 1


def _worker(args_tuple: tuple) -> dict:
    """One worker process: play n_games games, write chunks every chunk_size."""
    (
        worker_id,
        n_games,
        chunk_size,
        board_size,
        n_input_planes,
        visits,
        top_k,
        temperature,
        komi,
        rules,
        output_dir,
        katago_bin,
        katago_model,
        katago_config,
    ) = args_tuple
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = _next_chunk_idx_for_worker(out_dir, worker_id)
    buf: list[dict] = []
    n_positions = 0
    t0 = time.time()
    for g in range(n_games):
        result = play_one_game(
            katago_bin,
            katago_model,
            config_path=katago_config,
            board_size=board_size,
            n_input_planes=n_input_planes,
            visits=visits,
            top_k=top_k,
            temperature=temperature,
            komi=komi,
            rules=rules,
        )
        buf.append(result)
        n_positions += int(result["states"].shape[0])
        if (g + 1) % chunk_size == 0 or (g + 1) == n_games:
            _save_chunk(out_dir, worker_id, chunk_idx, buf)
            chunk_idx += 1
            buf = []
    return {
        "worker_id": worker_id,
        "n_games": n_games,
        "n_positions": n_positions,
        "elapsed_s": time.time() - t0,
    }


def generate_dataset_parallel(
    *,
    workers: int,
    games_per_worker: int,
    chunk_size: int = 10,
    board_size: int = 9,
    n_input_planes: int = 4,
    visits: int = 400,
    top_k: int = 8,
    temperature: float = 1.0,
    komi: float = 7.5,
    rules: str = "tromp-taylor",
    output_dir: str | Path = "/tmp/go_chunks",
    katago_bin: str | None = None,
    katago_model: str | None = None,
    katago_config: str | None = None,
) -> dict:
    """Orchestrate `workers` parallel KataGo subprocesses, each generating
    `games_per_worker` games. Returns aggregate stats."""
    katago_bin = katago_bin or os.environ.get("KATAGO_BIN")
    katago_model = katago_model or os.environ.get("KATAGO_MODEL")
    katago_config = katago_config or os.environ.get("KATAGO_CONFIG")
    if not katago_bin or not katago_model:
        raise RuntimeError(
            "KATAGO_BIN and KATAGO_MODEL must be set (env or kwargs)"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[gen] {workers} workers × {games_per_worker} games "
        f"on size {board_size}, chunk_size={chunk_size}, visits={visits}",
        flush=True,
    )
    t0 = time.time()
    worker_args = [
        (
            wid,
            games_per_worker,
            chunk_size,
            board_size,
            n_input_planes,
            visits,
            top_k,
            temperature,
            komi,
            rules,
            str(out_dir),
            katago_bin,
            katago_model,
            katago_config,
        )
        for wid in range(workers)
    ]
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_worker, worker_args)
    elapsed = time.time() - t0

    total_games = sum(r["n_games"] for r in results)
    total_positions = sum(r["n_positions"] for r in results)
    print(
        f"[gen] done in {elapsed:.1f}s — "
        f"{total_games} games, {total_positions} positions",
        flush=True,
    )
    return {
        "workers": workers,
        "games": total_games,
        "positions": total_positions,
        "elapsed_s": elapsed,
        "rate_games_per_hr": total_games / elapsed * 3600,
        "per_worker": results,
    }
