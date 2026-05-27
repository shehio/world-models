"""Head-to-head match between two Go checkpoints.

Both sides play with MCTS at the same `--sims`. Colors alternate game by
game so each side plays equal numbers as black and white. Score reported
from net A's perspective.

This is the canonical "did self-play actually improve over the prior?"
eval — when both sides use identical MCTS, any non-50% score is purely
the network quality difference.

Usage:
    uv run python scripts/h2h.py \\
        --ckpt-a /tmp/selfplay/net_iter042.pt \\
        --ckpt-b /tmp/prior/distilled_epoch015.pt \\
        --n-blocks 8 --n-filters 128 --n-input-planes 4 \\
        --workers 4 --games-per-worker 8 \\
        --sims 100 --board-size 9 --komi 7.5 \\
        --output /tmp/h2h_result.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go.board import BLACK, WHITE, GoBoard
from distill_go.config import GoConfig
from distill_go.mcts import run_mcts, select_move
from distill_go.network import AlphaZeroGoNet


def _net_move(net, board: GoBoard, history: list[GoBoard], cfg: GoConfig,
              device: torch.device, sims: int, temp: float) -> int:
    """One MCTS move from `net` at this position. Returns flat move idx."""
    visits = run_mcts(
        board, net, num_sims=sims,
        c_puct=cfg.c_puct, add_root_noise=False,
        device=device, n_input_planes=cfg.n_input_planes,
        game_history=history,
    )
    if not visits:
        return cfg.pass_move
    return select_move(visits, temperature=temp)


def play_one_game(net_a, net_b, cfg: GoConfig, device: torch.device,
                  sims: int, a_color: int, max_moves: int) -> dict:
    """One game. Returns winner (BLACK/WHITE/None) and a_won flag."""
    board = GoBoard(size=cfg.board_size, komi=cfg.komi)
    history: list[GoBoard] = []
    ply = 0
    while not board.is_game_over and ply < max_moves:
        # Tiny temperature on the opening for game diversity; greedy after.
        temp = 0.5 if ply < 4 else 0.0
        active = net_a if board.to_move == a_color else net_b
        move = _net_move(active, board, history, cfg, device, sims, temp)
        history.append(board.copy())
        if cfg.n_input_planes == 17 and len(history) > 8:
            history = history[-8:]
        try:
            board.play(move)
        except Exception:
            # If MCTS proposed an illegal move (defensive), forfeit by pass.
            board.play(cfg.pass_move)
        ply += 1
    w = board.winner()
    return {
        "winner": w,                 # BLACK / WHITE / None
        "a_won": (w == a_color),
        "draw": (w is None),
        "a_color": a_color,
        "ply": ply,
    }


def worker(args, ckpt_a: str, ckpt_b: str, out_queue, worker_id: int) -> None:
    """Play `games_per_worker` games on this process."""
    seed = (int(time.time() * 1000) ^ (worker_id * 2654435761)) & 0x7FFFFFFF
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        cfg = GoConfig(
            board_size=args.board_size,
            komi=args.komi,
            n_input_planes=args.n_input_planes,
            n_res_blocks=args.n_blocks,
            n_filters=args.n_filters,
            c_puct=args.c_puct,
        )
        device = torch.device("cpu")    # CPU is plenty for 9x9 evals
        net_a = AlphaZeroGoNet(cfg).to(device)
        net_b = AlphaZeroGoNet(cfg).to(device)
        net_a.load_state_dict(_load_state(ckpt_a, device))
        net_b.load_state_dict(_load_state(ckpt_b, device))
        net_a.eval(); net_b.eval()

        for g in range(args.games_per_worker):
            # Alternate colors: even worker+game index → A=BLACK, odd → A=WHITE.
            a_color = BLACK if (worker_id + g) % 2 == 0 else WHITE
            res = play_one_game(net_a, net_b, cfg, device,
                                 sims=args.sims, a_color=a_color,
                                 max_moves=args.max_moves)
            out_queue.put({"worker": worker_id, "game": g, **res})
    except Exception as e:
        import traceback
        out_queue.put({"worker": worker_id, "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})
    finally:
        out_queue.put({"worker": worker_id, "done": True})


def _load_state(path: str, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def score_to_elo(score: float) -> float:
    """Convert score s ∈ [0, 1] to Elo gap (A − B). Clamp at extremes."""
    if score <= 0: return -float("inf")
    if score >= 1: return float("inf")
    return -400 * math.log10(1 / score - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True,
                   help="Net A (score is from A's POV). Local path or s3:// uri.")
    p.add_argument("--ckpt-b", required=True,
                   help="Net B (opponent).")
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--n-filters", type=int, default=128)
    p.add_argument("--n-input-planes", type=int, default=4)
    p.add_argument("--c-puct", type=float, default=1.5)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games-per-worker", type=int, default=8)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--max-moves", type=int, default=200)

    p.add_argument("--output", required=True,
                   help="Where to write the JSON summary (local path).")
    args = p.parse_args()

    # Pull s3:// ckpts if needed.
    import subprocess, tempfile
    local_dir = tempfile.mkdtemp(prefix="h2h_")
    def _localize(uri: str) -> str:
        if not uri.startswith("s3://"):
            return uri
        local = f"{local_dir}/{Path(uri).name}"
        subprocess.run(["aws", "s3", "cp", uri, local, "--no-progress"], check=True)
        return local
    ckpt_a_local = _localize(args.ckpt_a)
    ckpt_b_local = _localize(args.ckpt_b)
    print(f"A: {args.ckpt_a}\n   → {ckpt_a_local}", flush=True)
    print(f"B: {args.ckpt_b}\n   → {ckpt_b_local}", flush=True)
    print(f"sims={args.sims} workers={args.workers}×{args.games_per_worker}games "
          f"board={args.board_size}x{args.board_size} komi={args.komi}", flush=True)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=args.workers * args.games_per_worker + args.workers)
    procs = [ctx.Process(target=worker,
                         args=(args, ckpt_a_local, ckpt_b_local, queue, w),
                         daemon=False)
             for w in range(args.workers)]

    t0 = time.time()
    for p_ in procs:
        p_.start()

    n_games = a_wins = draws = b_wins = 0
    n_errors = 0
    done_workers = 0
    while done_workers < args.workers:
        try:
            msg = queue.get(timeout=600)
        except Exception:
            print(f"  queue.get timed out after {done_workers}/{args.workers} workers done; aborting",
                  flush=True)
            break
        if msg.get("done"):
            done_workers += 1
            continue
        if "error" in msg:
            n_errors += 1
            print(f"  worker {msg['worker']} error: {msg['error']}", flush=True)
            continue
        n_games += 1
        if msg["draw"]:
            draws += 1
        elif msg["a_won"]:
            a_wins += 1
        else:
            b_wins += 1
        # One-line progress per game.
        a_color_str = "B" if msg["a_color"] == BLACK else "W"
        outcome = ("A-win" if msg["a_won"] else
                   "draw" if msg["draw"] else "B-win")
        print(f"  game {n_games:3d}  A={a_color_str}  ply={msg['ply']:3d}  → {outcome}",
              flush=True)

    for p_ in procs:
        p_.join(timeout=30)
        if p_.is_alive():
            p_.terminate()
            p_.join(timeout=5)

    score = (a_wins + 0.5 * draws) / max(n_games, 1)
    lo, hi = wilson_ci(a_wins + 0.5 * draws, n_games) if n_games else (0.0, 1.0)
    elo_gap = score_to_elo(score)
    elapsed = time.time() - t0

    summary = {
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "n_games": n_games,
        "a_wins": a_wins,
        "draws": draws,
        "b_wins": b_wins,
        "score_a": score,
        "score_ci_95": [lo, hi],
        "elo_gap_a_minus_b": elo_gap,
        "elo_gap_ci_95": [score_to_elo(lo), score_to_elo(hi)],
        "sims": args.sims,
        "board_size": args.board_size,
        "komi": args.komi,
        "errors": n_errors,
        "wallclock_s": elapsed,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60, flush=True)
    print(f"H2H: A wins {a_wins} / {draws} / {b_wins} (W/D/L from A's POV)", flush=True)
    print(f"     score = {score:.3f}  95% CI [{lo:.3f}, {hi:.3f}]", flush=True)
    print(f"     Elo gap (A − B) = {elo_gap:+.0f}  CI [{score_to_elo(lo):+.0f}, {score_to_elo(hi):+.0f}]",
          flush=True)
    print(f"     wallclock: {elapsed/60:.1f} min · errors: {n_errors}", flush=True)
    print(f"     summary written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
