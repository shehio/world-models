"""Generate Stockfish self-play dataset with multipv-soft-policy targets.

Run from 02c-distill-scaled root:
    uv run python scripts/generate_data.py \\
        --n-games 4000 --workers 6 --depth 15 --multipv 8 \\
        --temperature-pawns 1.0 \\
        --output data/sf_d15_mpv8_4000g.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from azdistill_scaled.stockfish_data import generate_dataset_parallel


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
    p.add_argument("--output", default="data/sf_multipv.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"generating {args.n_games} games | depth={args.depth} | multipv={args.multipv} "
          f"| T={args.temperature_pawns}pw | workers={args.workers}", flush=True)

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
    )
    print(f"\ndone in {meta['wall_seconds']/60:.1f} min")
    print(f"  positions:           {meta['n_positions']:,}")
    print(f"  positions w/<K PVs:  {meta['n_positions_below_k']:,}  "
          f"(these are end-game positions with few legal moves; fine)")
    print(f"  saved to:            {meta['output']}")


if __name__ == "__main__":
    main()
