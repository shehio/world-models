"""Binary-search the agent's Elo by playing it against increasingly-strong
Stockfish opponents until the score is in [0.4, 0.6] — that's the Elo
where the agent is well-matched, which gives the tightest CI.

Strategy: probe at the midpoint, halve the bracket each round. Each probe
plays N games at a specific UCI_Elo. After at most log2(2800-1350) ≈ 10
rounds we've localised the agent's Elo to ~50 Elo precision; in practice
3-4 rounds is enough because we stop the moment a probe lands in-bracket.

Sits next to eval.py and shells out to it for each probe.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ELO_MIN, ELO_MAX = 1350, 2800
STOP_WHEN_BRACKET_LEQ = 100
IN_BRACKET_SCORE_LO, IN_BRACKET_SCORE_HI = 0.45, 0.55


def implied_elo_gap(score: float) -> float:
    """Standard Elo formula: gap = -400 * log10(1/score - 1).
    Clamps score away from 0/1 to keep math finite when the agent
    sweeps or gets swept."""
    s = max(1e-3, min(1 - 1e-3, score))
    return -400.0 * math.log10(1.0 / s - 1.0)


def update_bracket(score: float, mid: int, lo: int, hi: int) -> tuple[int, int]:
    """Standard binary-search half: if we scored ≥0.5 at `mid`, the agent's
    Elo is above `mid` (raise lo); otherwise below (lower hi)."""
    if score >= 0.5:
        return mid, hi
    return lo, mid


def should_stop(lo: int, hi: int, score: float,
                stop_bracket: int = STOP_WHEN_BRACKET_LEQ) -> bool:
    """Stop when the bracket is tight enough, OR when a probe landed in
    [0.45, 0.55] — the score range where the implied Elo is most accurate."""
    if hi - lo <= stop_bracket:
        return True
    return IN_BRACKET_SCORE_LO <= score <= IN_BRACKET_SCORE_HI


def run_match(ckpt: str, uci_elo: int, n_games: int, workers: int,
              sims: int, n_blocks: int, n_filters: int) -> dict:
    script = (Path(__file__).resolve().parent / "eval.py")
    games_per_worker = max(1, n_games // workers)
    cmd = [
        sys.executable, str(script),
        "--ckpt", ckpt,
        "--workers", str(workers),
        "--games-per-worker", str(games_per_worker),
        "--sims", str(sims),
        "--n-blocks", str(n_blocks),
        "--n-filters", str(n_filters),
        "--stockfish-elo", str(uci_elo),
        "--agent-device", "cuda",
    ]
    print(f"\n=== probe vs UCI={uci_elo} ({n_games} games) ===", flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    score = None
    wdl = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("W/D/L:"):
            wdl = line
        if line.startswith("score:"):
            score = float(line.split()[1])
    print(out.stdout, flush=True)
    if score is None:
        raise RuntimeError(
            f"could not parse score from eval.py output:\n{out.stdout}\n{out.stderr}")
    return {"uci_elo": uci_elo, "n_games": n_games, "score": score, "wdl": wdl}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-probe", type=int, default=40,
                   help="games at each Stockfish level. 40 keeps a probe to "
                        "~10-12 min on g6.4xlarge — CI per probe is ±15%% "
                        "which is fine for bracketing.")
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--n-blocks", type=int, default=20)
    p.add_argument("--n-filters", type=int, default=256)
    p.add_argument("--out", default=None,
                   help="optional path to dump JSON trace")
    args = p.parse_args()

    lo, hi = ELO_MIN, ELO_MAX
    trace: list[dict] = []
    estimated_elo: float | None = None

    while hi - lo > STOP_WHEN_BRACKET_LEQ:
        mid = (lo + hi) // 2
        r = run_match(args.ckpt, mid, args.games_per_probe,
                      args.workers, args.sims, args.n_blocks, args.n_filters)
        trace.append(r)
        estimated_elo = mid + implied_elo_gap(r["score"])
        lo, hi = update_bracket(r["score"], mid, lo, hi)
        print(f"  -> bracket now [{lo}, {hi}], "
              f"estimated Elo ≈ {estimated_elo:.0f}", flush=True)
        if should_stop(lo, hi, r["score"]):
            if IN_BRACKET_SCORE_LO <= r["score"] <= IN_BRACKET_SCORE_HI:
                print("  in-bracket (score≈0.5), stopping early.", flush=True)
            break

    print("\n=== summary ===", flush=True)
    print(f"final bracket: [{lo}, {hi}]", flush=True)
    print(f"estimated Elo: {estimated_elo:.0f}", flush=True)
    print(f"probes: {len(trace)}", flush=True)
    for r in trace:
        print(f"  UCI={r['uci_elo']:>4}  score={r['score']:.3f}  {r['wdl']}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"final_bracket": [lo, hi],
             "estimated_elo": estimated_elo,
             "trace": trace},
            indent=2))


if __name__ == "__main__":
    main()
