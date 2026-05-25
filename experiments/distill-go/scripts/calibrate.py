"""Calibration match: KataGo@N vs an anchor opponent → derive KataGo's absolute Elo.

Two opponent kinds are supported, picked by --opponent:

  gnugo
      External GTP binary (`gnugo --mode gtp`). Anchor Elo = a literature
      value (default 1800, AlphaGo paper scale).

  katago
      A weaker KataGo (same engine, lower --opponent-visits). Useful for
      *chained* calibration when GNU Go is too weak to give an informative
      head-to-head — e.g., KataGo@v200 swept GNU Go 100/0, so we re-anchor
      via the chain  v200 ↔ v1 ↔ GNU Go.

Reports KataGo (test side)'s implied absolute Elo using the anchor.

Usage examples:

    # Match A: KataGo@v1 (pure policy) vs GNU Go on 9x9.
    uv run python experiments/distill-go/scripts/calibrate.py \\
        --katago-visits 1 --opponent gnugo --gnugo-level 10 \\
        --anchor-elo 1800 --games 100

    # Match B: KataGo@v200 vs KataGo@v1 on 9x9. Set --anchor-elo to
    # match A's *measured* KataGo@v1 absolute Elo.
    uv run python experiments/distill-go/scripts/calibrate.py \\
        --katago-visits 200 --opponent katago --opponent-visits 1 \\
        --anchor-elo <from-match-A> --games 100
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


class GTPEngine:
    """Minimal GTP wrapper around any binary that speaks GTP on stdio.

    Subclasses provide the argv list. Used for both GNU Go and Pachi.
    """

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "GTPEngine":
        self._proc = subprocess.Popen(
            self.argv,
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

    @classmethod
    def gnugo(cls, binary: str, level: int = 10) -> "GTPEngine":
        return cls([binary, "--mode", "gtp", "--level", str(level)])

    @classmethod
    def pachi(cls, binary: str, playouts: int = 5000) -> "GTPEngine":
        # `-t =N` = exactly N playouts (no time-based budget). `-d 0` disables
        # Pachi's curses-based debug display so it doesn't open a TTY (in
        # containers ncurses fails with "Error opening terminal: unknown"
        # unless TERM=dumb is also set in the env). 5000 playouts ≈ KGS 1d
        # on 9x9.
        return cls([binary, "-d", "0", "-t", f"={playouts}"])

    def _send(self, cmd: str) -> str:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()
        body_lines: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"GTP engine ({self.argv[0]}) closed stdout while waiting for '{cmd}'")
            if line.startswith("=") or line.startswith("?"):
                body_lines.append(line[1:].strip())
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
    gtp_opp: GTPEngine | None,
    *,
    opponent: str,
    board_size: int,
    komi: float,
    test_visits: int,
    opponent_visits: int,
    test_color: int,
    max_moves: int = 200,
) -> dict:
    """One game test-KataGo vs opponent. test_color ∈ {BLACK, WHITE}."""
    board = GoBoard(size=board_size, komi=komi)
    history_gtp: list[tuple[str, str]] = []
    if opponent in ("gnugo", "pachi"):
        assert gtp_opp is not None
        gtp_opp.setup(board_size, komi)

    t0 = time.time()
    passes = 0
    ply = 0
    for ply in range(max_moves):
        if board.is_game_over:
            break
        to_move_color = "B" if board.to_move == BLACK else "W"
        if board.to_move == test_color:
            gtp_move = kata.best_move(
                history_gtp, board_size=board_size,
                visits=test_visits, komi=komi, rules="tromp-taylor",
            )
        elif opponent == "katago":
            gtp_move = kata.best_move(
                history_gtp, board_size=board_size,
                visits=opponent_visits, komi=komi, rules="tromp-taylor",
            )
        else:  # gnugo or pachi via GTP
            gtp_move = gtp_opp.genmove(to_move_color)
        gtp_move = gtp_move.strip()

        try:
            flat_idx = gtp_to_flat(gtp_move, board_size)
        except Exception:
            flat_idx = board.pass_move

        legal_mask = board.legal_mask()
        if flat_idx >= len(legal_mask) or legal_mask[flat_idx] == 0:
            flat_idx = board.pass_move
        canon_move = flat_to_gtp(flat_idx, board_size)

        # GTP opponents (gnugo/pachi) are stateful — mirror the test side's
        # move to them. KataGo (both sides if opponent=katago) is queried
        # per-position with the full history, so it's stateless.
        if opponent in ("gnugo", "pachi") and board.to_move == test_color:
            gtp_opp.play(to_move_color, canon_move)

        history_gtp.append((to_move_color, canon_move))
        board.play(flat_idx)
        passes = passes + 1 if flat_idx == board.pass_move else 0
        if passes >= 2:
            break

    b_pts, w_pts = board.tromp_taylor_score()
    winner = BLACK if b_pts > w_pts else WHITE
    test_won = (winner == test_color)
    return {
        "test_color": "B" if test_color == BLACK else "W",
        "ply": ply,
        "black_score": b_pts,
        "white_score": w_pts,
        "winner": "B" if winner == BLACK else "W",
        "test_won": bool(test_won),
        "elapsed_s": time.time() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--katago-visits", type=int, default=200,
                   help="visits for the KataGo side under test")
    p.add_argument("--opponent", choices=["gnugo", "pachi", "katago"], default="gnugo")
    p.add_argument("--opponent-visits", type=int, default=1,
                   help="visits for the opponent KataGo (only if --opponent katago)")
    p.add_argument("--gnugo-level", type=int, default=10,
                   help="gnugo --level (only if --opponent gnugo)")
    p.add_argument("--pachi-playouts", type=int, default=5000,
                   help="Pachi `-t =N` playouts per move (only if --opponent pachi)")
    p.add_argument("--anchor-elo", type=float, default=1800.0,
                   help="Assumed absolute Elo of the opponent. Defaults: gnugo L10 ≈ 1800 "
                        "(AlphaGo paper, Silver 2016); Pachi @ 5000 playouts ≈ 2100 (KGS 1d).")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--max-moves", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    katago_bin = os.environ["KATAGO_BIN"]
    katago_model = os.environ["KATAGO_MODEL"]
    katago_config = os.environ.get("KATAGO_CONFIG")
    gnugo_bin = os.environ.get("GNUGO_BIN", "gnugo")
    pachi_bin = os.environ.get("PACHI_BIN", "pachi")

    random.seed(args.seed)
    print(f"test:    KataGo @ {args.katago_visits}v", flush=True)
    if args.opponent == "gnugo":
        print(f"opponent: gnugo --level {args.gnugo_level} (anchor Elo = {args.anchor_elo:.0f})", flush=True)
    elif args.opponent == "pachi":
        print(f"opponent: pachi -t ={args.pachi_playouts} (anchor Elo = {args.anchor_elo:.0f})", flush=True)
    else:
        print(f"opponent: KataGo @ {args.opponent_visits}v (anchor Elo = {args.anchor_elo:.0f})", flush=True)
    print(f"games: {args.games}  board: {args.board_size}x{args.board_size}  komi: {args.komi}", flush=True)
    print("", flush=True)

    wins = 0
    t0 = time.time()
    if args.opponent == "gnugo":
        gtp_ctx = GTPEngine.gnugo(gnugo_bin, level=args.gnugo_level)
    elif args.opponent == "pachi":
        gtp_ctx = GTPEngine.pachi(pachi_bin, playouts=args.pachi_playouts)
    else:
        gtp_ctx = _NullCtx()
    with KataGoAnalysisEngine(katago_bin, katago_model, katago_config) as kata, gtp_ctx as gtp_opp:
        for g in range(args.games):
            test_color = BLACK if g % 2 == 0 else WHITE
            res = play_one_game(
                kata, gtp_opp if args.opponent in ("gnugo", "pachi") else None,
                opponent=args.opponent,
                board_size=args.board_size, komi=args.komi,
                test_visits=args.katago_visits,
                opponent_visits=args.opponent_visits,
                test_color=test_color,
                max_moves=args.max_moves,
            )
            wins += int(res["test_won"])
            print(
                f"game {g+1:>3}/{args.games}  test={res['test_color']}  "
                f"ply={res['ply']:>3}  B={res['black_score']:>5.1f} W={res['white_score']:>5.1f}  "
                f"winner={res['winner']}  test_won={res['test_won']}  "
                f"({res['elapsed_s']:.1f}s)",
                flush=True,
            )

    n = args.games
    score = wins / n
    lo, hi = wilson_ci(wins, n)
    elo_diff = score_to_elo_diff(score)
    elo_lo = score_to_elo_diff(lo)
    elo_hi = score_to_elo_diff(hi)
    test_abs = args.anchor_elo + elo_diff
    abs_lo = args.anchor_elo + elo_lo
    abs_hi = args.anchor_elo + elo_hi
    elapsed = time.time() - t0

    print("=" * 64, flush=True)
    print(
        f"KataGo@{args.katago_visits}v wins {wins}/{n}  score={score:.3f}  "
        f"95% CI=[{lo:.3f}, {hi:.3f}]",
        flush=True,
    )
    if args.opponent == "gnugo":
        opp_label = f"gnugo L{args.gnugo_level}"
    elif args.opponent == "pachi":
        opp_label = f"pachi @ {args.pachi_playouts} playouts"
    else:
        opp_label = f"KataGo@{args.opponent_visits}v"
    print(
        f"Elo diff (KataGo@{args.katago_visits}v − {opp_label}) "
        f"≈ {elo_diff:+.0f}  95% CI=[{elo_lo:+.0f}, {elo_hi:+.0f}]",
        flush=True,
    )
    print(f"Anchor Elo ({opp_label}) = {args.anchor_elo:.0f}", flush=True)
    print(
        f"Implied KataGo@{args.katago_visits}v absolute Elo "
        f"≈ {test_abs:.0f}  95% CI=[{abs_lo:.0f}, {abs_hi:.0f}]",
        flush=True,
    )
    print(f"total eval time: {elapsed:.1f}s", flush=True)
    return 0


class _NullCtx:
    def __enter__(self):
        return None
    def __exit__(self, *_exc):
        return False


if __name__ == "__main__":
    sys.exit(main())
