"""Concatenate worker chunks from a multi-pod shards_partial/ tree.

The single-dir `distill_go.merge.merge_chunks` can't handle the cloud
layout because worker indices repeat across pods. This walks
`shards_partial/pod-*/worker_*_chunk_*.npz` and concatenates them in
pod-order.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


POD_RE = re.compile(r"/pod-(\d+)/")
NAME_RE = re.compile(r"^worker_(\d+)_chunk_(\d+)\.npz$")


def sort_key(path: Path):
    pod = POD_RE.search(str(path))
    m = NAME_RE.match(path.name)
    return (
        int(pod.group(1)) if pod else -1,
        int(m.group(1)) if m else -1,
        int(m.group(2)) if m else -1,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-root", required=True,
                   help="root containing pod-*/worker_*_chunk_*.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.shards_root)
    files = sorted(
        (f for f in root.glob("pod-*/worker_*_chunk_*.npz")
         if NAME_RE.match(f.name)),
        key=sort_key,
    )
    if not files:
        raise FileNotFoundError(f"no chunks under {root}/pod-*/")
    print(f"merging {len(files)} chunks from {root}")

    parts: dict[str, list] = {
        "states": [], "moves": [], "multipv_indices": [],
        "multipv_logprobs": [], "zs": [],
    }
    for f in files:
        d = np.load(f)
        for k in parts:
            parts[k].append(d[k])

    arrays = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    n = arrays["states"].shape[0]
    print(f"total positions: {n}")
    for k, a in arrays.items():
        print(f"  {k}: shape={a.shape} dtype={a.dtype}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
