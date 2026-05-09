"""Generate Stockfish self-play dataset.

Run from project root:
    uv run python scripts/generate_data.py --n-games 1000 --workers 6 --depth 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from azdistill.stockfish_data import generate_dataset_parallel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--depth", type=int, default=8,
                   help="Stockfish search depth per move during data generation")
    p.add_argument("--stockfish-elo", type=int, default=None,
                   help="if set, restrict Stockfish to this Elo for diversity (else full strength)")
    p.add_argument("--stockfish-skill", type=int, default=None,
                   help="if set, Stockfish Skill Level (0-20)")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--random-opening-plies", type=int, default=6)
    p.add_argument("--output", default="data/stockfish_games.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"generating {args.n_games} Stockfish self-play games "
          f"(depth={args.depth}, elo={args.stockfish_elo}, skill={args.stockfish_skill}) "
          f"across {args.workers} workers ...")
    meta = generate_dataset_parallel(
        output_path=args.output,
        n_games=args.n_games,
        n_workers=args.workers,
        sf_elo=args.stockfish_elo,
        sf_skill=args.stockfish_skill,
        depth=args.depth,
        max_plies=args.max_plies,
        random_opening_plies=args.random_opening_plies,
        seed=args.seed,
    )
    print(f"\ndone in {meta['wall_seconds']/60:.1f} min")
    print(f"  positions: {meta['n_positions']:,}")
    print(f"  saved to:  {meta['output']}")
    print(f"  shapes:    states {meta['shape_states']}  "
          f"moves {meta['shape_moves']}  zs {meta['shape_zs']}")


if __name__ == "__main__":
    main()
