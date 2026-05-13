"""Streaming chunks -> merged data.npz for a whole region's shards_partial/.

Bypasses generate_data.py --finalize-only and merge_shards.py both, which
materialize all chunks in RAM. This script:

  - globs every worker_*_chunk_*.npz under shards_partial/pod-*/.../chunks/
  - for each array key (states, moves, multipv_indices, multipv_logprobs, zs):
      pass 1: read each chunk's shape/dtype for that key
      pass 2: open one zip entry, write npy header, then stream each chunk's
              bytes directly into the deflate stream

Peak memory is bounded by chunk size (~30 MB). Tested-equivalent output to
np.savez_compressed(states=..., moves=...,  K=..., temperature_pawns=...).

Also concatenates worker_*.pgn into games.pgn and writes metadata.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
import zipfile

import numpy as np
from numpy.lib import format as npy_format


KEYS = ("states", "moves", "multipv_indices", "multipv_logprobs", "zs")
POD_RE = re.compile(r"/pod-(\d+)/")
WORKER_RE = re.compile(r"worker_(\d+)_chunk_(\d+)\.npz$")


def chunk_sort_key(path: str):
    pod = POD_RE.search(path)
    wm = WORKER_RE.search(path)
    pod_i = int(pod.group(1)) if pod else -1
    worker_i = int(wm.group(1)) if wm else -1
    chunk_i = int(wm.group(2)) if wm else -1
    return (pod_i, worker_i, chunk_i)


def stream_key(
    zf: zipfile.ZipFile,
    key: str,
    chunk_files: list[str],
) -> tuple[int, np.dtype, tuple]:
    """Pass 1 + pass 2 for a single key. Returns (total_rows, dtype, inner_shape)."""
    # Pass 1: collect shapes/dtype without holding arrays
    total_rows = 0
    dtype = None
    inner_shape = None
    for cf in chunk_files:
        with np.load(cf) as d:
            a = d[key]
            total_rows += int(a.shape[0])
            if dtype is None:
                dtype = a.dtype
                inner_shape = tuple(a.shape[1:])
            elif a.dtype != dtype or tuple(a.shape[1:]) != inner_shape:
                raise RuntimeError(
                    f"shape/dtype mismatch in {cf} for {key}: "
                    f"got {a.dtype}{a.shape}, expected {dtype}{(None,)+inner_shape}"
                )

    outer_shape = (total_rows,) + inner_shape
    header = {
        "shape": outer_shape,
        "fortran_order": False,
        "descr": npy_format.dtype_to_descr(dtype),
    }
    print(f"  {key}: dtype={dtype} shape={outer_shape} "
          f"({np.prod(outer_shape) * dtype.itemsize / 1e9:.2f} GB uncompressed)",
          flush=True)

    # Pass 2: open zip entry, write header, stream bytes
    with zf.open(key + ".npy", "w", force_zip64=True) as f:
        npy_format.write_array_header_2_0(f, header)
        written = 0
        for cf in chunk_files:
            with np.load(cf) as d:
                a = np.ascontiguousarray(d[key])
            # ensure dtype/shape match (already validated above)
            f.write(a.tobytes())
            written += a.nbytes
            del a
        print(f"  {key}: wrote {written:,} bytes to zip entry", flush=True)

    return total_rows, dtype, inner_shape


def write_scalar(zf: zipfile.ZipFile, name: str, value: np.ndarray) -> None:
    """Write a scalar array (K, temperature_pawns) as its own .npy entry."""
    with zf.open(name + ".npy", "w") as f:
        npy_format.write_array(f, value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards-root", required=True,
                   help="local dir containing pod-*/.../chunks/ tree (synced from S3)")
    p.add_argument("--output", required=True,
                   help="output dir; data.npz + games.pgn + metadata.json written here")
    p.add_argument("--multipv", type=int, required=True)
    p.add_argument("--temperature", type=float, required=True)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    chunks = sorted(
        glob.glob(os.path.join(args.shards_root, "**/chunks/worker_*_chunk_*.npz"),
                  recursive=True),
        key=chunk_sort_key,
    )
    if not chunks:
        raise SystemExit(f"no chunks under {args.shards_root}")

    pods = sorted({POD_RE.search(c).group(0) for c in chunks if POD_RE.search(c)})
    print(f"found {len(chunks)} chunks across {len(pods)} pods", flush=True)

    out_npz = os.path.join(args.output, "data.npz")
    t0 = time.time()
    total_positions = None
    with zipfile.ZipFile(out_npz, "w", compression=zipfile.ZIP_DEFLATED,
                        compresslevel=6, allowZip64=True) as zf:
        for key in KEYS:
            print(f"--- {key} ---", flush=True)
            total_rows, _, _ = stream_key(zf, key, chunks)
            if total_positions is None:
                total_positions = total_rows
            elif total_rows != total_positions:
                raise RuntimeError(f"row count mismatch: {key}={total_rows}, "
                                   f"expected {total_positions}")
        write_scalar(zf, "K", np.array(args.multipv, dtype=np.int32))
        write_scalar(zf, "temperature_pawns",
                     np.array(args.temperature, dtype=np.float32))
    print(f"wrote {out_npz}  ({os.path.getsize(out_npz)/1e9:.2f} GB compressed) "
          f"in {time.time()-t0:.0f}s", flush=True)

    # PGN concat: streaming, no memory pressure
    pgn_files = sorted(
        glob.glob(os.path.join(args.shards_root, "**/chunks/worker_*.pgn"),
                  recursive=True),
        key=lambda p: (POD_RE.search(p).group(0) if POD_RE.search(p) else "",
                       p),
    )
    out_pgn = os.path.join(args.output, "games.pgn")
    n_games = 0
    with open(out_pgn, "wb") as out:
        for pf in pgn_files:
            with open(pf, "rb") as f:
                while True:
                    buf = f.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
                    n_games += buf.count(b'[Event "')
            out.write(b"\n\n")
    print(f"wrote {out_pgn}  ({n_games:,} games)", flush=True)

    out_meta = {
        "n_pods": len(pods),
        "n_chunks": len(chunks),
        "n_positions": total_positions,
        "n_games": n_games,
        "multipv": args.multipv,
        "temperature_pawns": args.temperature,
        "merge_ts": time.time(),
        "merge_tool": "merge_chunks.py (streaming)",
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(out_meta, f, indent=2, sort_keys=True)
    print(f"wrote {args.output}/metadata.json", flush=True)


if __name__ == "__main__":
    main()
