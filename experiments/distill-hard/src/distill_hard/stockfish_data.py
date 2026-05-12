"""Generate Stockfish self-play games and save as (state, move, z) tuples.

Each game:
  1. Play `random_opening_plies` random legal moves from the initial
     position. This diversifies openings (Stockfish without randomization
     is otherwise deterministic from move 1).
  2. Hand the position to Stockfish; Stockfish plays both sides at the
     given depth/strength until terminal or max_plies.
  3. For every position visited, record:
        state_planes:  (19, 8, 8) float32 — same encoder as 02
        move_idx:      int in [0, 4672) — Stockfish's chosen move
        z_pov:         {-1, 0, +1} from the side-to-move's POV at that position

Output: a single NPZ with stacked arrays.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import chess
import chess.engine
import numpy as np

from wm_chess.board import N_INPUT_PLANES, N_POLICY, encode_board, encode_move


@dataclass
class GameStats:
    plies: int
    outcome: float  # +1 white / -1 black / 0 draw
    n_recorded: int  # number of positions saved (== plies after random opening)


def play_one_game(
    engine: chess.engine.SimpleEngine,
    depth: int,
    max_plies: int,
    random_opening_plies: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[int], list[float], GameStats]:
    """Play one Stockfish self-play game. Returns (states, moves, zs, stats).

    `engine` must already be configured for the desired strength.
    Each `state` is encoded BEFORE the move is played (so the move is the
    target action FROM that state).
    """
    board = chess.Board()

    # Random opening: just play random legal moves for `random_opening_plies` plies.
    # Don't record these (Stockfish didn't choose them).
    for _ in range(random_opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))

    states: list[np.ndarray] = []
    move_idxs: list[int] = []
    side_to_move_at_record: list[bool] = []  # True if white-to-move

    limit = chess.engine.Limit(depth=depth)
    while not board.is_game_over(claim_draw=True) and len(states) < max_plies:
        result = engine.play(board, limit)
        move = result.move
        if move is None:
            break  # paranoia
        states.append(encode_board(board))
        move_idxs.append(encode_move(move, board))
        side_to_move_at_record.append(board.turn == chess.WHITE)
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        z_white = 0.0
    else:
        z_white = 1.0 if outcome.winner == chess.WHITE else -1.0

    zs = [z_white if w else -z_white for w in side_to_move_at_record]

    return states, move_idxs, zs, GameStats(
        plies=len(states),
        outcome=z_white,
        n_recorded=len(states),
    )


def _worker_generate(args) -> dict:
    (worker_id, n_games, stockfish_path, sf_elo, sf_skill,
     depth, max_plies, random_opening_plies, seed) = args

    eng_kwargs = {}
    if sf_elo is not None:
        eng_kwargs["UCI_LimitStrength"] = True
        eng_kwargs["UCI_Elo"] = int(sf_elo)
    if sf_skill is not None:
        eng_kwargs["Skill Level"] = int(sf_skill)
    eng_kwargs["Threads"] = 1

    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure(eng_kwargs)
        all_states, all_moves, all_zs = [], [], []
        for g in range(n_games):
            states, moves, zs, stats = play_one_game(
                engine, depth=depth,
                max_plies=max_plies,
                random_opening_plies=random_opening_plies,
                rng=rng,
            )
            all_states.extend(states)
            all_moves.extend(moves)
            all_zs.extend(zs)
        return {
            "worker_id": worker_id,
            "states": np.stack(all_states) if all_states else np.zeros((0, N_INPUT_PLANES, 8, 8), dtype=np.float32),
            "moves": np.array(all_moves, dtype=np.int64),
            "zs": np.array(all_zs, dtype=np.float32),
            "n_games": n_games,
            "n_positions": len(all_states),
        }
    finally:
        engine.quit()


def generate_dataset_parallel(
    output_path: str,
    n_games: int,
    n_workers: int = 6,
    stockfish_path: str | None = None,
    sf_elo: int | None = None,
    sf_skill: int | None = None,
    depth: int = 8,
    max_plies: int = 200,
    random_opening_plies: int = 6,
    seed: int = 0,
) -> dict:
    """Run n_games of Stockfish self-play across n_workers processes.

    Saves a single NPZ at `output_path` with arrays:
        states (N, 19, 8, 8) float32
        moves  (N,) int64       — target action index in [0, 4672)
        zs     (N,) float32     — z from side-to-move's POV at that position

    Returns metadata dict.
    """
    import shutil
    if stockfish_path is None:
        stockfish_path = shutil.which("stockfish")
    if stockfish_path is None:
        raise FileNotFoundError("stockfish binary not found")

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # Distribute games across workers.
    games_per_worker = [n_games // n_workers] * n_workers
    for i in range(n_games % n_workers):
        games_per_worker[i] += 1

    worker_args = [
        (i, games_per_worker[i], stockfish_path, sf_elo, sf_skill,
         depth, max_plies, random_opening_plies, seed + i * 10007)
        for i in range(n_workers)
    ]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_generate, worker_args)
    dt = time.time() - t0

    states = np.concatenate([r["states"] for r in results], axis=0)
    moves = np.concatenate([r["moves"] for r in results], axis=0)
    zs = np.concatenate([r["zs"] for r in results], axis=0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, states=states, moves=moves, zs=zs)

    return {
        "n_games": n_games,
        "n_positions": int(states.shape[0]),
        "wall_seconds": dt,
        "output": output_path,
        "shape_states": states.shape,
        "shape_moves": moves.shape,
        "shape_zs": zs.shape,
    }
