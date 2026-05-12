"""Concatenate N shard datasets in library/games into one merged dataset.

Each shard is a complete library leaf (data.npz + games.pgn + metadata.json).
We concatenate the tensor arrays end-to-end, append the PGNs, and write a
merged metadata.json that records the source shard paths.

Usage:
    python wm_chess/scripts/merge_shards.py \
        --shards-glob 'library/games/sf-18/d15-mpv8-T1/g12500-seed*' \
        --output      'library/games/sf-18/d15-mpv8-T1/g100000-merged-20260512T1700Z' \
        --multipv 8 --temperature 1.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards-glob", required=True,
                   help="glob for shard directories (each must contain data.npz)")
    p.add_argument("--output", required=True,
                   help="output directory; data.npz + games.pgn + metadata.json written here")
    p.add_argument("--multipv", type=int, required=True)
    p.add_argument("--temperature", type=float, required=True)
    args = p.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    shards = [s for s in shards if os.path.isfile(os.path.join(s, "data.npz"))]
    if not shards:
        raise SystemExit(f"no complete shards matched {args.shards_glob}")
    print(f"merging {len(shards)} shards into {args.output}", flush=True)

    os.makedirs(args.output, exist_ok=True)

    # Concatenate NPZs.
    states_l, moves_l, mpv_idx_l, mpv_logp_l, zs_l = [], [], [], [], []
    total_positions = 0
    source_meta: list[dict] = []
    for s in shards:
        d = np.load(os.path.join(s, "data.npz"))
        states_l.append(d["states"])
        moves_l.append(d["moves"])
        mpv_idx_l.append(d["multipv_indices"])
        mpv_logp_l.append(d["multipv_logprobs"])
        zs_l.append(d["zs"])
        total_positions += int(d["states"].shape[0])
        meta_path = os.path.join(s, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                source_meta.append({"path": os.path.relpath(s), **json.load(f)})

    out_npz = os.path.join(args.output, "data.npz")
    tmp = out_npz + ".tmp.npz"
    np.savez_compressed(
        tmp,
        states=np.concatenate(states_l, axis=0),
        moves=np.concatenate(moves_l, axis=0),
        multipv_indices=np.concatenate(mpv_idx_l, axis=0),
        multipv_logprobs=np.concatenate(mpv_logp_l, axis=0),
        zs=np.concatenate(zs_l, axis=0),
        K=np.array(args.multipv, dtype=np.int32),
        temperature_pawns=np.array(args.temperature, dtype=np.float32),
    )
    os.replace(tmp, out_npz)
    print(f"  data.npz: {total_positions:,} positions", flush=True)

    # Concatenate PGNs.
    out_pgn = os.path.join(args.output, "games.pgn")
    n_games = 0
    with open(out_pgn, "w") as outf:
        for s in shards:
            pgn = os.path.join(s, "games.pgn")
            if not os.path.exists(pgn):
                continue
            with open(pgn) as f:
                text = f.read()
            outf.write(text)
            if not text.endswith("\n\n"):
                outf.write("\n\n")
            n_games += text.count('[Event "')
    print(f"  games.pgn: {n_games:,} games", flush=True)

    # Merged metadata.
    out_meta = {
        "merged_from_shards": [m["path"] for m in source_meta],
        "n_shards": len(shards),
        "n_positions": total_positions,
        "n_games": n_games,
        "multipv": args.multipv,
        "temperature_pawns": args.temperature,
        "merge_ts": time.time(),
        "shards_metadata": source_meta,
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(out_meta, f, indent=2, sort_keys=True)

    print(f"wrote merged dataset to {args.output}", flush=True)


if __name__ == "__main__":
    main()
