"""Multi-worker KataGo datagen. Writes worker_*.npz chunks then merges.

Local usage (9x9 demo):
    KATAGO_BIN=$(which katago) \\
    KATAGO_MODEL=/opt/homebrew/Cellar/katago/<ver>/share/katago/g170e-b20c256x2-s5303129600-d1228401921.bin.gz \\
    KATAGO_CONFIG=/opt/homebrew/Cellar/katago/<ver>/share/katago/configs/analysis_example.cfg \\
    uv run --project experiments/distill-go python experiments/distill-go/scripts/generate_data.py \\
        --workers 4 --games-per-worker 25 --board-size 9 --visits 100 \\
        --output-dir /tmp/go9_chunks --merged-path /tmp/go9_merged.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go.katago_data_parallel import generate_dataset_parallel
from distill_go.merge import merge_chunks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games-per-worker", type=int, default=25)
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--board-size", type=int, default=9, choices=[9, 13, 19])
    p.add_argument("--n-input-planes", type=int, default=4, choices=[4, 17])
    p.add_argument("--visits", type=int, default=400)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--rules", default="tromp-taylor")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="dir for worker_*.npz chunks (resume-safe)")
    p.add_argument("--merged-path", type=Path, default=None,
                   help="if set, merge chunks into this single .npz at the end")
    args = p.parse_args()

    t0 = time.time()
    stats = generate_dataset_parallel(
        workers=args.workers,
        games_per_worker=args.games_per_worker,
        chunk_size=args.chunk_size,
        board_size=args.board_size,
        n_input_planes=args.n_input_planes,
        visits=args.visits,
        top_k=args.top_k,
        temperature=args.temperature,
        komi=args.komi,
        rules=args.rules,
        output_dir=args.output_dir,
    )
    print(
        f"[datagen] {stats['games']} games, {stats['positions']} positions, "
        f"{stats['elapsed_s']:.1f}s = {stats['rate_games_per_hr']:.0f} games/hr",
        flush=True,
    )
    if args.merged_path is not None:
        merged = merge_chunks(args.output_dir, args.merged_path)
        print(
            f"[merge] {merged['n_files']} chunks → {merged['out_path']} "
            f"({merged['n_positions']:,} positions, K={merged['K']})",
            flush=True,
        )
    print(f"[total] {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
