"""Calibration match: KataGo@N vs GNU Go on 9x9 → derive KataGo's absolute Elo.

Pins a known-strength external anchor (GNU Go, ~1800 Elo on the AlphaGo paper
scale) and measures KataGo's Elo gap to it via head-to-head play. This lets
us convert all our "student vs KataGo@N" deltas to absolute Elo.

Usage (host):
    KATAGO_BIN=$(which katago) \\
    KATAGO_MODEL=/path/to/kata1-b18c384nbt-...bin.gz \\
    KATAGO_CONFIG=/path/to/katago-analysis.cfg \\
    GNUGO_BIN=$(which gnugo) \\
    uv run --project experiments/distill-go python experiments/distill-go/scripts/calibrate.py \\
        --katago-visits 200 --gnugo-level 10 --games 100 --board-size 9
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go.board import BLACK, WHITE, GoBoard, flat_to_gtp, gtp_to_flat
from distill_go.katago_data import KataGoAnalysisEngine


class GnuGoEngine:
    """Minimal GTP wrapper around `gnugo --mode gtp`."""

    def __init__(self, binary: str, level: int = 10) -> None:
        self.binary = binary
        self.level = level
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "GnuGoEngine":
        self._proc = subprocess.Popen(
            [self.binary, "--mode", "gtp", "--level", str(self.level)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, *_exc) -> None:
        if self._proc is not None:
            try:
                self._send("quit")
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _send(self, cmd: str) -> str:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()
        # GTP reply: '= <body>\n\n' or '? <err>\n\n'
        body_lines: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"gnugo closed stdout while waiting for '{cmd}'")
            if line.startswith("=") or line.startswith("?"):
                body_lines.append(line[1:].strip())
                # consume the blank terminator
                term = self._proc.stdout.readline()
                if term.strip() != "":
                    body_lines.append(term.strip())
                break
            if line.strip():
                body_lines.append(line.strip())
        return " ".join(body_lines).strip()

    def setup(self, board_size: int, komi: float) -> None:
        self._send(f"boardsize {board_size}")
        self._send(f"komi {komi}")
        self._send("clear_board")

    def play(self, color: str, gtp_move: str) -> None:
        self._send(f"play {color} {gtp_move}")

    def genmove(self, color: str) -> str:
        return self._send(f"genmove {color}").strip().lower()


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n)) / denom
    return (center - half, center + half)


def score_to_elo_diff(score: float, eps: float = 1e-9) -> float:
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def play_one_game(
    kata: KataGoAnalysisEngine,
    gnugo: GnuGoEngine,
    board_size: int,
    komi: float,
    katago_visits: int,
    katago_color: int,
    max_moves: int = 200,
) -> dict:
    """One game KataGo vs GNU Go. katago_color ∈ {BLACK, WHITE}."""
    board = GoBoard(size=board_size, komi=komi)
    history_gtp: list[tuple[str, str]] = []
    gnugo.setup(board_size, komi)
    color_to_gtp = {BLACK: "B", WHITE: "W"}

    t0 = time.time()
    passes = 0
    ply = 0
    for ply in range(max_moves):
        if board.is_game_over:
            break
        to_move_color = "B" if board.to_move == BLACK else "W"
        if board.to_move == katago_color:
            gtp_move = kata.best_move(
                history_gtp, board_size=board_size,
                visits=katago_visits, komi=komi, rules="tromp-taylor",
            )
        else:
            gtp_move = gnugo.genmove(to_move_color)
        gtp_move = gtp_move.strip()

        # Normalize KataGo/GNU Go output to local flat index.
        try:
            flat_idx = gtp_to_flat(gtp_move, board_size)
        except Exception:
            flat_idx = board.pass_move

        # Legality guard — illegal moves become passes.
        legal_mask = board.legal_mask()
        if flat_idx >= len(legal_mask) or legal_mask[flat_idx] == 0:
            flat_idx = board.pass_move
        # Canonicalize so 'resign'/illegal both surface as 'pass' to gnugo.
        canon_move = flat_to_gtp(flat_idx, board_size)

        # Mirror to the opponent that didn't generate this move.
        if board.to_move == katago_color:
            # KataGo moved; tell gnugo.
            gnugo.play(to_move_color, canon_move)
        # if gnugo moved, nothing to tell KataGo — its analysis engine is
        # stateless and takes the full history each query.

        history_gtp.append((to_move_color, canon_move))
        board.play(flat_idx)
        passes = passes + 1 if flat_idx == board.pass_move else 0
        if passes >= 2:
            break

    b_pts, w_pts = board.tromp_taylor_score()
    winner = BLACK if b_pts > w_pts else WHITE
    katago_won = (winner == katago_color)
    return {
        "katago_color": "B" if katago_color == BLACK else "W",
        "ply": ply,
        "black_score": b_pts,
        "white_score": w_pts,
        "winner": "B" if winner == BLACK else "W",
        "katago_won": bool(katago_won),
        "elapsed_s": time.time() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--katago-visits", type=int, default=200)
    p.add_argument("--gnugo-level", type=int, default=10)
    p.add_argument("--gnugo-anchor-elo", type=float, default=1800.0,
                   help="Assumed Elo for GNU Go at the given level. Default 1800 "
                        "follows the AlphaGo paper (Silver et al. 2016) calibration scale.")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--max-moves", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    katago_bin = os.environ["KATAGO_BIN"]
    katago_model = os.environ["KATAGO_MODEL"]
    katago_config = os.environ.get("KATAGO_CONFIG")
    gnugo_bin = os.environ.get("GNUGO_BIN", "gnugo")

    random.seed(args.seed)
    print(f"katago: {katago_bin} @ {args.katago_visits}v", flush=True)
    print(f"opponent: gnugo --level {args.gnugo_level} (anchor Elo = {args.gnugo_anchor_elo:.0f})", flush=True)
    print(f"games: {args.games}  board: {args.board_size}x{args.board_size}  komi: {args.komi}", flush=True)
    print(f"max_moves: {args.max_moves}", flush=True)
    print("", flush=True)

    wins = 0
    t0 = time.time()
    with KataGoAnalysisEngine(katago_bin, katago_model, katago_config) as kata, \
         GnuGoEngine(gnugo_bin, level=args.gnugo_level) as gnugo:
        for g in range(args.games):
            katago_color = BLACK if g % 2 == 0 else WHITE
            res = play_one_game(
                kata, gnugo,
                board_size=args.board_size, komi=args.komi,
                katago_visits=args.katago_visits,
                katago_color=katago_color,
                max_moves=args.max_moves,
            )
            wins += int(res["katago_won"])
            print(
                f"game {g+1:>3}/{args.games}  katago={res['katago_color']}  "
                f"ply={res['ply']:>3}  B={res['black_score']:>5.1f} W={res['white_score']:>5.1f}  "
                f"winner={res['winner']}  katago_won={res['katago_won']}  "
                f"({res['elapsed_s']:.1f}s)",
                flush=True,
            )

    n = args.games
    score = wins / n
    lo, hi = wilson_ci(wins, n)
    elo_diff = score_to_elo_diff(score)
    elo_lo = score_to_elo_diff(lo)
    elo_hi = score_to_elo_diff(hi)
    katago_abs = args.gnugo_anchor_elo + elo_diff
    abs_lo = args.gnugo_anchor_elo + elo_lo
    abs_hi = args.gnugo_anchor_elo + elo_hi
    elapsed = time.time() - t0

    print("=" * 64, flush=True)
    print(
        f"KataGo wins {wins}/{n}  score={score:.3f}  "
        f"95% CI=[{lo:.3f}, {hi:.3f}]",
        flush=True,
    )
    print(
        f"Elo diff (KataGo@{args.katago_visits}v − GNUGo L{args.gnugo_level}) "
        f"≈ {elo_diff:+.0f}  95% CI=[{elo_lo:+.0f}, {elo_hi:+.0f}]",
        flush=True,
    )
    print(
        f"GNU Go anchor Elo = {args.gnugo_anchor_elo:.0f}",
        flush=True,
    )
    print(
        f"Implied KataGo@{args.katago_visits}v absolute Elo "
        f"≈ {katago_abs:.0f}  95% CI=[{abs_lo:.0f}, {abs_hi:.0f}]",
        flush=True,
    )
    print(f"total eval time: {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
