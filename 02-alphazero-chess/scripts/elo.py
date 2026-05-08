"""Compute the trained agent's Elo against a battery of opponents.

Anchors:
  - random player: assumed Elo 0 (a convention; absolute scale is arbitrary).
  - Stockfish at UCI_Elo=1320 (lowest the engine permits): anchor 1320.
  - Stockfish skill=0 + depth=1: very weak depth-limited Stockfish, no anchor.

For each opponent we play N games (colors alternated), compute score
S = (wins + 0.5 * draws) / N, derive an Elo gap
    Δ = -400 * log10(1/S - 1)
and a 95% CI from the standard error sqrt(S*(1-S)/N) propagated through
the same transform.

When S is exactly 0 or 1 the gap is unbounded; we report a one-sided
bound at the edge of the 95% CI.

Run from the project root:
    uv run python scripts/elo.py --ckpt checkpoints/net_iter009.pt \\
        --games-vs-random 100 --games-vs-stockfish 30 --sims 50
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from alphazero.arena import (
    network_policy,
    play_match,
    random_policy,
    stockfish_engine,
    stockfish_policy,
)
from alphazero.config import Config
from alphazero.network import AlphaZeroNet, get_device


def score_to_elo_gap(score: float) -> float:
    """Convert match score in (0, 1) to Elo difference (positive = winner)."""
    if score <= 0:
        return -float("inf")
    if score >= 1:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_with_ci(wins: int, draws: int, losses: int, anchor: float = 0.0) -> dict:
    """Return absolute Elo (anchor + gap) and 95% CI bounds."""
    n = wins + draws + losses
    score = (wins + 0.5 * draws) / max(n, 1)
    gap = score_to_elo_gap(score)
    # 95% CI on score via normal approximation.
    if 0.0 < score < 1.0 and n > 0:
        se = math.sqrt(score * (1.0 - score) / n)
        lo, hi = max(0.0, score - 1.96 * se), min(1.0, score + 1.96 * se)
        gap_lo, gap_hi = score_to_elo_gap(lo), score_to_elo_gap(hi)
    else:
        # All-win or all-loss: one-sided bound using +/- 1 effective game.
        if score >= 1.0:
            adj = (n - 0.5) / n
        else:
            adj = 0.5 / n
        gap_lo = gap_hi = score_to_elo_gap(adj)
    return {
        "n": n,
        "wins": wins, "draws": draws, "losses": losses,
        "score": score,
        "elo": anchor + gap if math.isfinite(gap) else anchor + gap_lo,
        "elo_lo": anchor + gap_lo,
        "elo_hi": anchor + gap_hi,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games-vs-random", type=int, default=100)
    p.add_argument("--games-vs-stockfish", type=int, default=30)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8,
                   help="MCTS batch size for the agent. 1 = sequential.")
    p.add_argument("--device", default=None)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--skip-stockfish", action="store_true")
    args = p.parse_args()

    cfg = Config()
    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}")

    network = AlphaZeroNet(cfg).to(device)
    network.load_state_dict(torch.load(args.ckpt, map_location=device))
    network.eval()
    print(f"loaded ckpt: {args.ckpt}")
    net_pol = network_policy(network, cfg, device, sims=args.sims, batch_size=args.batch_size)

    rows = []

    # --- vs random (anchor: 0 Elo) ---
    print(f"\nplaying {args.games_vs_random} games vs random ...")
    t0 = time.time()
    stats = play_match(net_pol, random_policy, n_games=args.games_vs_random,
                       max_plies=args.max_plies)
    print(f"  W{stats['wins']}/D{stats['draws']}/L{stats['losses']}  "
          f"score={stats['score']:.3f}  ({time.time()-t0:.1f}s)")
    rows.append(("random (anchor 0)",
                 elo_with_ci(stats['wins'], stats['draws'], stats['losses'], anchor=0.0)))

    # --- vs Stockfish ---
    if not args.skip_stockfish:
        # Configurations to test, ordered weakest-to-strongest.
        sf_configs = [
            ("Stockfish skill=0 d=1",   {"skill": 0},      {"depth": 1}, 0.0),     # no Elo anchor
            ("Stockfish UCI_Elo=1320",  {"elo": 1320},     {"depth": 8}, 1320.0),  # anchored
        ]
        for label, eng_kwargs, play_kwargs, anchor in sf_configs:
            print(f"\nplaying {args.games_vs_stockfish} games vs {label} ...")
            t0 = time.time()
            try:
                with stockfish_engine(**eng_kwargs) as eng:
                    sf_pol_local = stockfish_policy(eng, **play_kwargs)
                    stats = play_match(net_pol, sf_pol_local,
                                       n_games=args.games_vs_stockfish,
                                       max_plies=args.max_plies)
                print(f"  W{stats['wins']}/D{stats['draws']}/L{stats['losses']}  "
                      f"score={stats['score']:.3f}  ({time.time()-t0:.1f}s)")
                rows.append((label, elo_with_ci(stats['wins'], stats['draws'],
                                                 stats['losses'], anchor=anchor)))
            except Exception as e:
                print(f"  skipped: {e}")

    # --- summary table ---
    print("\n" + "=" * 70)
    print(f"{'Opponent':<28} {'N':>4} {'W/D/L':>10} {'Score':>7} {'Elo (95% CI)':>20}")
    print("-" * 70)
    for label, r in rows:
        wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
        if math.isfinite(r['elo_lo']) and math.isfinite(r['elo_hi']):
            elo_str = f"{r['elo']:.0f} [{r['elo_lo']:.0f}, {r['elo_hi']:.0f}]"
        else:
            elo_str = f"{r['elo']:.0f} (bound)"
        print(f"{label:<28} {r['n']:>4} {wdl:>10} {r['score']:>7.3f} {elo_str:>20}")
    print("=" * 70)


if __name__ == "__main__":
    main()
