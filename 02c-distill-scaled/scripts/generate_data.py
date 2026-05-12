"""Generate Stockfish self-play dataset with multipv-soft-policy targets.

Two output modes (pick exactly one or both):

  Legacy flat NPZ (backward compat):
    uv run python scripts/generate_data.py \\
        --n-games 4000 --workers 6 --depth 15 --multipv 8 \\
        --output data/sf_d15_mpv8_4000g.npz

  Library mode (crash-safe, indexed by SF metadata):
    uv run python scripts/generate_data.py \\
        --n-games 4000 --workers 6 --depth 15 --multipv 8 \\
        --library-root data/library

  Library layout:
    <root>/sf-<version>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/
        data.npz            (assembled tensors)
        games.pgn           (assembled PGN)
        metadata.json       (full provenance)
        chunks/             (per-worker incremental saves)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from azdistill_scaled.stockfish_data import (
    generate_dataset_parallel,
    finalize_library_path,
    library_dataset_path,
    detect_stockfish_version,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=4000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--depth", type=int, default=15,
                   help="Stockfish search depth per analyse() call")
    p.add_argument("--stockfish-elo", type=int, default=None,
                   help="if set, restrict Stockfish to this Elo (else full strength)")
    p.add_argument("--stockfish-skill", type=int, default=None,
                   help="if set, Stockfish Skill Level (0-20)")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--random-opening-plies", type=int, default=6)
    p.add_argument("--multipv", type=int, default=8,
                   help="how many top moves to record per position (soft target width)")
    p.add_argument("--temperature-pawns", type=float, default=1.0,
                   help="softmax T in pawns. T=1 → 50cp gap ≈ 62/38 split. Lower = sharper.")
    p.add_argument("--output", default=None,
                   help="flat NPZ output path (legacy). If unset, requires --library-root.")
    p.add_argument("--library-root", default=None,
                   help="library root directory. Dataset is written to "
                        "<root>/sf-<v>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/")
    p.add_argument("--chunk-size", type=int, default=50,
                   help="games per worker chunk (crash-safe checkpoint granularity)")
    p.add_argument("--finalize-only", action="store_true",
                   help="don't run gen — just re-assemble any existing chunks "
                        "in the library path (recovery after spot reclaim).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-dir", default=None,
                   help="if set, each worker writes .progress_wNN.json here every game")
    args = p.parse_args()

    if not args.output and not args.library_root:
        p.error("must specify --output or --library-root (or both)")

    if args.finalize_only:
        if not args.library_root:
            p.error("--finalize-only requires --library-root")
        sf_version = detect_stockfish_version()
        lib_path = library_dataset_path(
            args.library_root, sf_version, args.depth, args.multipv,
            args.temperature_pawns, args.n_games, args.seed,
        )
        print(f"finalizing chunks under {lib_path}", flush=True)
        fin = finalize_library_path(
            lib_path, multipv=args.multipv,
            temperature_pawns=args.temperature_pawns,
        )
        print(f"  used_chunks: {fin['used_chunks']}")
        print(f"  positions:   {fin['n_positions']:,}")
        print(f"  games(pgn):  {fin['n_games_pgn']:,}")
        print(f"  data:        {fin['data_npz']}")
        print(f"  pgn:         {fin['games_pgn']}")
        return

    print(f"generating {args.n_games} games | depth={args.depth} | multipv={args.multipv} "
          f"| T={args.temperature_pawns}pw | workers={args.workers} "
          f"| chunk_size={args.chunk_size}", flush=True)

    import os as _os
    if args.progress_dir:
        progress_dir = args.progress_dir
    elif args.output:
        progress_dir = _os.path.dirname(args.output) or "."
    else:
        progress_dir = args.library_root
    print(f"progress files: {progress_dir}/.progress_wNN.json  "
          f"(poll with `scripts/progress.py {progress_dir}`)", flush=True)

    meta = generate_dataset_parallel(
        output_path=args.output,
        n_games=args.n_games,
        n_workers=args.workers,
        sf_elo=args.stockfish_elo,
        sf_skill=args.stockfish_skill,
        depth=args.depth,
        max_plies=args.max_plies,
        random_opening_plies=args.random_opening_plies,
        multipv=args.multipv,
        temperature_pawns=args.temperature_pawns,
        seed=args.seed,
        progress_dir=progress_dir,
        library_root=args.library_root,
        chunk_size=args.chunk_size,
    )
    print(f"\ndone in {meta['wall_seconds']/60:.1f} min")
    print(f"  sf_version:          {meta['sf_version']}")
    print(f"  positions:           {meta['n_positions']:,}")
    print(f"  positions w/<K PVs:  {meta['n_positions_below_k']:,}  "
          f"(end-game positions with few legal moves; fine)")
    print(f"  saved to:            {meta['output']}")
    if "library_path" in meta:
        print(f"  library path:        {meta['library_path']}")


if __name__ == "__main__":
    main()
