"""Concatenate worker chunks into a single training .npz.

The cluster pipeline uses `wm_chess/scripts/merge_chunks.py` for chess;
that script is shape-agnostic and could be used for Go too. This is a
self-contained version for the small-scale demo.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def merge_chunks(chunk_dir: str | Path, out_path: str | Path) -> dict:
    chunk_dir = Path(chunk_dir)
    out_path = Path(out_path)
    pattern = re.compile(r"^worker_\d+_chunk_\d+\.npz$")
    files = sorted(f for f in chunk_dir.iterdir() if pattern.match(f.name))
    if not files:
        raise FileNotFoundError(f"no worker_*.npz under {chunk_dir}")

    states_list = []
    moves_list = []
    mpv_idx_list = []
    mpv_logp_list = []
    zs_list = []
    K = None
    for f in files:
        d = np.load(f)
        states_list.append(d["states"])
        moves_list.append(d["moves"])
        mpv_idx_list.append(d["multipv_indices"])
        mpv_logp_list.append(d["multipv_logprobs"])
        zs_list.append(d["zs"])
        if K is None:
            K = int(d["multipv_indices"].shape[1])

    states = np.concatenate(states_list, axis=0)
    moves = np.concatenate(moves_list, axis=0)
    mpv_idx = np.concatenate(mpv_idx_list, axis=0)
    mpv_logp = np.concatenate(mpv_logp_list, axis=0)
    zs = np.concatenate(zs_list, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        states=states,
        moves=moves,
        multipv_indices=mpv_idx,
        multipv_logprobs=mpv_logp,
        zs=zs,
        K=np.int64(K),
    )
    tmp.replace(out_path)

    return {
        "n_files": len(files),
        "n_positions": int(states.shape[0]),
        "K": K,
        "out_path": str(out_path),
    }
