"""Eval: student-MCTS vs KataGo gauntlet → win rate → Elo difference.

Per game:
  - Shared GoBoard, alternating sides
  - Student plays via our MCTS with the trained checkpoint
  - KataGo plays via its analysis engine at `--katago-visits`
  - Scoring via Tromp-Taylor on our local board (komi 7.5)

Reports:
  - Per-game outcome (B/W/Result)
  - Aggregate score & 95% Wilson CI on win rate
  - Elo difference (student − KataGo) derived from the score

Usage:
    KATAGO_BIN=$(which katago) KATAGO_MODEL=... KATAGO_CONFIG=... \\
    uv run --project experiments/distill-go python experiments/distill-go/scripts/eval.py \\
        --ckpt /tmp/go9_ckpt/distilled_epoch009.pt \\
        --board-size 9 --n-input-planes 4 \\
        --student-sims 100 --katago-visits 50 \\
        --games 30 --output /tmp/go9_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from distill_go.board import BLACK, WHITE, GoBoard, flat_to_gtp, gtp_to_flat
from distill_go.config import GoConfig
from distill_go.katago_data import KataGoAnalysisEngine
from distill_go.mcts import run_mcts, select_move
from distill_go.network import AlphaZeroGoNet, get_device


def _student_move(
    board: GoBoard,
    network,
    device: torch.device,
    cfg: GoConfig,
    sims: int,
    history_boards: list[GoBoard],
    move_num: int,
) -> int:
    visits = run_mcts(
        board, network, num_sims=sims,
        c_puct=cfg.c_puct, add_root_noise=False,
        device=device, n_input_planes=cfg.n_input_planes,
        game_history=history_boards,
    )
    # Greedy after the first few moves; small temperature early to break symmetry.
    temp = 0.5 if move_num < 4 else 0.0
    return select_move(visits, temperature=temp)


def _katago_move(
    eng: KataGoAnalysisEngine,
    history_gtp: list[tuple[str, str]],
    board_size: int,
    visits: int,
    komi: float,
    rules: str,
) -> int:
    gtp = eng.best_move(history_gtp, board_size=board_size, visits=visits,
                        komi=komi, rules=rules)
    return gtp_to_flat(gtp, board_size)


def play_one_eval_game(
    network,
    device: torch.device,
    cfg: GoConfig,
    eng: KataGoAnalysisEngine,
    student_color: int,
    student_sims: int,
    katago_visits: int,
    max_moves: int = 200,
) -> dict:
    board = GoBoard(size=cfg.board_size, komi=cfg.komi)
    history_gtp: list[tuple[str, str]] = []
    history_boards: list[GoBoard] = [board.copy()]
    color_to_gtp = {BLACK: "B", WHITE: "W"}

    t0 = time.time()
    ply = 0
    for ply in range(max_moves):
        if board.is_game_over:
            break
        if board.to_move == student_color:
            move_idx = _student_move(board, network, device, cfg,
                                     student_sims, history_boards, ply)
        else:
            move_idx = _katago_move(eng, history_gtp, cfg.board_size,
                                    katago_visits, cfg.komi, cfg.rules)

        # Defensive: if either side hands us an illegal move, treat it as a pass.
        legal_mask = board.legal_mask()
        if move_idx >= len(legal_mask) or legal_mask[move_idx] == 0:
            move_idx = cfg.pass_move

        gtp = flat_to_gtp(move_idx, cfg.board_size)
        played_color = color_to_gtp[board.to_move]
        history_gtp.append((played_color, gtp))
        board.play(move_idx)
        history_boards.append(board.copy())
        if len(history_boards) > 9:
            history_boards = history_boards[-9:]

    b_pts, w_pts = board.tromp_taylor_score()
    winner = BLACK if b_pts > w_pts else WHITE
    student_won = (winner == student_color)
    return {
        "student_color": "B" if student_color == BLACK else "W",
        "ply": ply,
        "black_score": b_pts,
        "white_score": w_pts,
        "winner": "B" if winner == BLACK else "W",
        "student_won": bool(student_won),
        "elapsed_s": time.time() - t0,
    }


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n)) / denom
    return (center - half, center + half)


def score_to_elo_diff(score: float, eps: float = 1e-9) -> float:
    """Convert score (= wins / games, draws as 0.5) to Elo diff via logistic."""
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help=".pt checkpoint to load into student net")
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--n-input-planes", type=int, default=4, choices=[4, 17])
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--n-filters", type=int, default=64)
    p.add_argument("--student-sims", type=int, default=100)
    p.add_argument("--katago-visits", type=int, default=50)
    p.add_argument("--games", type=int, default=20,
                   help="total games (split half black, half white)")
    p.add_argument("--max-moves", type=int, default=200)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=None,
                   help="optional .json to write per-game results to")
    args = p.parse_args()

    cfg = GoConfig(
        board_size=args.board_size,
        n_input_planes=args.n_input_planes,
        n_res_blocks=args.n_blocks,
        n_filters=args.n_filters,
        komi=args.komi,
    )
    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}", flush=True)

    network = AlphaZeroGoNet(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    network.load_state_dict(state)
    network.eval()

    katago_bin = os.environ.get("KATAGO_BIN")
    katago_model = os.environ.get("KATAGO_MODEL")
    katago_config = os.environ.get("KATAGO_CONFIG")
    if not katago_bin or not katago_model:
        raise SystemExit("KATAGO_BIN and KATAGO_MODEL env vars must be set")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results: list[dict] = []
    wins = 0
    t0 = time.time()
    with KataGoAnalysisEngine(katago_bin, katago_model, katago_config) as eng:
        for g in range(args.games):
            student_color = BLACK if g % 2 == 0 else WHITE  # alternate
            res = play_one_eval_game(
                network, device, cfg, eng,
                student_color=student_color,
                student_sims=args.student_sims,
                katago_visits=args.katago_visits,
                max_moves=args.max_moves,
            )
            results.append(res)
            wins += int(res["student_won"])
            print(
                f"game {g+1:>2}/{args.games}  student={res['student_color']}  "
                f"ply={res['ply']:>3}  B={res['black_score']:>5.1f}  "
                f"W={res['white_score']:>5.1f}  winner={res['winner']}  "
                f"student_won={res['student_won']}  ({res['elapsed_s']:.1f}s)",
                flush=True,
            )

    score = wins / args.games
    lo, hi = wilson_ci(wins, args.games)
    elo = score_to_elo_diff(score)
    elo_lo = score_to_elo_diff(lo)
    elo_hi = score_to_elo_diff(hi)
    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print(
        f"student wins {wins}/{args.games}  "
        f"score={score:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]",
        flush=True,
    )
    print(
        f"Elo diff (student − KataGo@{args.katago_visits}v) ≈ "
        f"{elo:+.0f}  95% CI=[{elo_lo:+.0f}, {elo_hi:+.0f}]",
        flush=True,
    )
    print(f"total eval time: {elapsed:.1f}s", flush=True)

    if args.output is not None:
        out = {
            "config": {
                "ckpt": args.ckpt,
                "board_size": cfg.board_size,
                "n_input_planes": cfg.n_input_planes,
                "n_blocks": cfg.n_res_blocks,
                "n_filters": cfg.n_filters,
                "student_sims": args.student_sims,
                "katago_visits": args.katago_visits,
                "games": args.games,
            },
            "summary": {
                "wins": wins,
                "losses": args.games - wins,
                "score": score,
                "score_ci_lo": lo,
                "score_ci_hi": hi,
                "elo_diff": elo,
                "elo_ci_lo": elo_lo,
                "elo_ci_hi": elo_hi,
                "elapsed_s": elapsed,
            },
            "games": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
