"""Generate Stockfish self-play games with MULTIPV soft policy targets.

Difference from 02b
-------------------
02b saved (state, move_idx, z): one hard one-hot target per position.

02c saves (state, multipv_indices[K], multipv_logprobs[K], move_idx, z):
the same position, but with the top-K moves Stockfish considered, scored
via a softmax over their centipawn evaluations. Hard one-hot is also kept
for compatibility / sanity checks.

Why this matters
----------------
With hard targets the student learns "Stockfish played e4 here." Period.
With multipv soft targets it learns "e4 was best (+0.30), Nf3 was close
(+0.25), c4 also fine (+0.20), ...other moves much worse." Roughly 10×
more bits-per-position about chess from the same Stockfish search.

How the target is built per position
------------------------------------
For each visited position we ask Stockfish for an `analyse(MultiPV=K)`.
This is ONE search (Stockfish keeps K principal variations alongside its
normal search), so it's NOT K× slower — typically 1.3–1.7× slower than
single-PV at the same depth.

Each of the K results has a `Score.relative` in centipawns from the side
to move's POV. We softmax those scores at temperature T (in pawns):

    p_i ∝ exp(score_i_cp / (100 * T))

with mate scores clipped to ±MATE_CP (default 1000 cp = 10 pawns) so a
detected mate doesn't collapse the distribution to one-hot.

The output policy distribution lives ONLY on those K moves; everything
else is implicitly 0. We store (indices, logprobs) — both length K — so
the training loss can scatter into the full 4672-action vector.

Output NPZ schema
-----------------
    states              (N, 19, 8, 8) float32
    move_idx            (N,)          int64   - top-1 (Stockfish's actual play)
    z                   (N,)          float32 - game outcome, side-to-move POV
    multipv_indices     (N, K)        int64   - top-K action indices
    multipv_logprobs    (N, K)        float32 - log-prob over those K moves
    K                   ()            int     - the K value used
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import chess
import chess.engine
import chess.pgn
import io
import numpy as np

from .board import N_INPUT_PLANES, N_POLICY, encode_board, encode_move


# Clipping for mate scores so they don't collapse the softmax to one-hot.
# 1000 cp = 10 pawns. With T=1 pawn this means p_mate / p_next ≈ exp(10) — still
# very concentrated but not infinite, so the student gets *some* gradient on
# the runner-up move too.
MATE_CP = 1000


@dataclass
class GameStats:
    plies: int
    outcome: float       # +1 white / -1 black / 0 draw
    n_recorded: int      # positions saved
    n_below_k: int       # positions where SF returned fewer than K PVs (legal_moves < K)


def _score_to_cp(score: chess.engine.PovScore, mate_cp: int = MATE_CP) -> float:
    """Convert a python-chess PovScore (relative to side-to-move) into a
    centipawn value, clipping mate scores so the softmax stays well-behaved.

    score.relative is a Cp(...) or a Mate(...) wrapper. .score(mate_score=X)
    returns an int: positive cp value, or X for a mate-for-us, -X for
    mate-against-us. We use that to fold mates into the same scale.
    """
    return float(score.relative.score(mate_score=mate_cp))


def _multipv_to_distribution(
    info_list: list[dict],
    board: chess.Board,
    temperature_pawns: float,
    K: int,
) -> tuple[list[int], np.ndarray, int]:
    """Turn an `engine.analyse(MultiPV=K)` result into (indices, logprobs).

    info_list is a list of length up to K. Each entry has at least:
        info["pv"][0]  : chess.Move        — the head of this PV (the candidate move)
        info["score"]  : chess.engine.PovScore (relative to current STM)

    Returns
    -------
    indices : list[int] of length K (padded with -1 if SF returned fewer PVs)
    logprobs: np.ndarray of length K (padded with -inf for missing entries,
              then re-normalized softmax over the valid entries only)
    n_valid : how many real PVs we got (≤ K). Padding entries have logprob = -inf.
    """
    valid = []  # list of (move_idx, score_cp)
    for info in info_list[:K]:
        pv = info.get("pv")
        if not pv:
            continue
        move = pv[0]
        # Skip illegal/malformed entries defensively (shouldn't happen w/ SF).
        if move not in board.legal_moves:
            continue
        idx = encode_move(move, board)
        cp = _score_to_cp(info["score"])
        valid.append((idx, cp))

    n_valid = len(valid)

    # Softmax over scores in pawns / T.
    # Numerical stability: subtract max BEFORE exponentiating.
    indices = [v[0] for v in valid]
    if n_valid == 0:
        # Shouldn't happen (the game wasn't over so SF had moves), but be safe.
        indices = []
        logprobs = np.full((K,), -np.inf, dtype=np.float32)
        return [-1] * K, logprobs, 0

    scores_cp = np.array([v[1] for v in valid], dtype=np.float64)
    scaled = scores_cp / (100.0 * temperature_pawns)
    scaled -= scaled.max()  # stability
    weights = np.exp(scaled)
    probs = weights / weights.sum()
    valid_logprobs = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)

    # Pad to length K so all rows in the NPZ have the same shape.
    out_indices = indices + [-1] * (K - n_valid)
    out_logprobs = np.full((K,), -np.inf, dtype=np.float32)
    out_logprobs[:n_valid] = valid_logprobs
    return out_indices, out_logprobs, n_valid


def _board_to_pgn(board_initial: chess.Board, moves: list[chess.Move],
                  outcome_white_pov: float, headers: dict) -> str:
    """Replay a sequence of moves from the initial position and emit a PGN
    string. Used to build the game-library archive alongside the NPZ.
    """
    game = chess.pgn.Game()
    for k, v in headers.items():
        game.headers[k] = str(v)
    node = game
    b = board_initial.copy()
    for mv in moves:
        if mv not in b.legal_moves:
            break  # paranoia
        node = node.add_variation(mv)
        b.push(mv)
    result_str = "1-0" if outcome_white_pov > 0 else "0-1" if outcome_white_pov < 0 else "1/2-1/2"
    game.headers["Result"] = result_str
    return str(game)


def play_one_game(
    engine: chess.engine.SimpleEngine,
    depth: int,
    max_plies: int,
    random_opening_plies: int,
    multipv: int,
    temperature_pawns: float,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[int], list[list[int]], list[np.ndarray], list[float], GameStats, str]:
    """Play one Stockfish self-play game with multipv-soft-target recording.

    Returns
    -------
    states           : list of (19,8,8) arrays
    played_moves     : list of int (Stockfish's actually-played move idx)
    multipv_indices  : list of length-K lists of int (top-K candidate move idxs)
    multipv_logprobs : list of length-K np.float32 arrays (softmax-logprobs)
    zs               : list of float (game outcome from side-to-move POV)
    stats            : GameStats
    """
    board = chess.Board()

    # Track every move played (random opening + Stockfish moves) so we can
    # emit a complete PGN for the game library.
    initial_board = board.copy()
    all_moves_played: list[chess.Move] = []

    # Random opening plies for diversity. We don't *record* these — Stockfish
    # didn't choose them — but it doesn't matter that they're random because
    # all targets come from Stockfish's subsequent search.
    for _ in range(random_opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        moves = list(board.legal_moves)
        mv = rng.choice(moves)
        all_moves_played.append(mv)
        board.push(mv)

    states: list[np.ndarray] = []
    played_moves: list[int] = []
    multipv_indices: list[list[int]] = []
    multipv_logprobs: list[np.ndarray] = []
    side_to_move_at_record: list[bool] = []

    limit = chess.engine.Limit(depth=depth)
    n_below_k = 0

    while not board.is_game_over(claim_draw=True) and len(states) < max_plies:
        # engine.analyse with MultiPV=K returns a list of dicts, one per PV.
        # This is a SINGLE search — internally SF maintains K PVs alongside
        # its alpha-beta, so it's typically only 1.3–1.7× slower than MultiPV=1
        # at the same depth.
        info_list = engine.analyse(board, limit, multipv=multipv)
        if not isinstance(info_list, list):
            info_list = [info_list]  # python-chess returns dict if multipv=1

        mpv_idx, mpv_logp, n_valid = _multipv_to_distribution(
            info_list, board, temperature_pawns=temperature_pawns, K=multipv,
        )
        if n_valid < multipv:
            n_below_k += 1

        # Top-1 from the multipv result IS Stockfish's chosen move at this depth.
        # We always have at least 1 PV here (game wasn't over and SF found moves).
        if not info_list or not info_list[0].get("pv"):
            break  # paranoia — should never trigger
        played_move = info_list[0]["pv"][0]

        states.append(encode_board(board))
        played_moves.append(encode_move(played_move, board))
        multipv_indices.append(mpv_idx)
        multipv_logprobs.append(mpv_logp)
        side_to_move_at_record.append(board.turn == chess.WHITE)
        all_moves_played.append(played_move)
        board.push(played_move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        z_white = 0.0
    else:
        z_white = 1.0 if outcome.winner == chess.WHITE else -1.0
    zs = [z_white if w else -z_white for w in side_to_move_at_record]

    pgn_text = _board_to_pgn(
        initial_board, all_moves_played, z_white,
        headers={
            "Event": "Stockfish self-play (02c distillation data gen)",
            "White": f"Stockfish d={depth} mpv={multipv}",
            "Black": f"Stockfish d={depth} mpv={multipv}",
            "Plies": len(all_moves_played),
            "RandomOpeningPlies": random_opening_plies,
        },
    )

    return states, played_moves, multipv_indices, multipv_logprobs, zs, GameStats(
        plies=len(states),
        outcome=z_white,
        n_recorded=len(states),
        n_below_k=n_below_k,
    ), pgn_text


def _write_progress(progress_path: str, payload: dict) -> None:
    """Atomic progress write. Workers in mp.Pool don't pipe stdout back to the
    parent (with `spawn` start method), so we expose per-worker progress via
    on-disk JSON instead. Polled externally by scripts/progress.py.
    """
    import json
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, progress_path)


def _worker_generate(args) -> dict:
    (worker_id, n_games, stockfish_path, sf_elo, sf_skill,
     depth, max_plies, random_opening_plies, multipv,
     temperature_pawns, seed, progress_dir) = args

    eng_kwargs: dict = {"Threads": 1}
    if sf_elo is not None:
        eng_kwargs["UCI_LimitStrength"] = True
        eng_kwargs["UCI_Elo"] = int(sf_elo)
    if sf_skill is not None:
        eng_kwargs["Skill Level"] = int(sf_skill)

    progress_path = (
        os.path.join(progress_dir, f".progress_w{worker_id:02d}.json")
        if progress_dir else None
    )

    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    t_start = time.time()
    try:
        engine.configure(eng_kwargs)
        all_states, all_played, all_mpv_idx, all_mpv_logp, all_zs = [], [], [], [], []
        all_pgns: list[str] = []
        total_below_k = 0
        total_plies = 0

        if progress_path:
            _write_progress(progress_path, {
                "worker_id": worker_id, "games_done": 0, "games_total": n_games,
                "positions": 0, "plies": 0, "elapsed_s": 0.0, "ts": time.time(),
                "phase": "start",
            })

        for g in range(n_games):
            states, played, mpv_idx, mpv_logp, zs, stats, pgn_text = play_one_game(
                engine,
                depth=depth,
                max_plies=max_plies,
                random_opening_plies=random_opening_plies,
                multipv=multipv,
                temperature_pawns=temperature_pawns,
                rng=rng,
            )
            all_states.extend(states)
            all_played.extend(played)
            all_mpv_idx.extend(mpv_idx)
            all_mpv_logp.extend(mpv_logp)
            all_zs.extend(zs)
            all_pgns.append(pgn_text)
            total_below_k += stats.n_below_k
            total_plies += stats.plies

            if progress_path:
                _write_progress(progress_path, {
                    "worker_id": worker_id,
                    "games_done": g + 1,
                    "games_total": n_games,
                    "positions": len(all_states),
                    "plies": total_plies,
                    "elapsed_s": time.time() - t_start,
                    "ts": time.time(),
                    "phase": "running",
                })

        if progress_path:
            _write_progress(progress_path, {
                "worker_id": worker_id, "games_done": n_games, "games_total": n_games,
                "positions": len(all_states), "plies": total_plies,
                "elapsed_s": time.time() - t_start, "ts": time.time(),
                "phase": "done",
            })

        return {
            "worker_id": worker_id,
            "states": (np.stack(all_states)
                       if all_states else np.zeros((0, N_INPUT_PLANES, 8, 8), dtype=np.float32)),
            "moves": np.array(all_played, dtype=np.int64),
            "multipv_indices": (np.array(all_mpv_idx, dtype=np.int64)
                                if all_mpv_idx else np.zeros((0, multipv), dtype=np.int64)),
            "multipv_logprobs": (np.stack(all_mpv_logp)
                                 if all_mpv_logp else np.zeros((0, multipv), dtype=np.float32)),
            "zs": np.array(all_zs, dtype=np.float32),
            "pgns": all_pgns,
            "n_games": n_games,
            "n_positions": len(all_states),
            "n_below_k": total_below_k,
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
    depth: int = 15,
    max_plies: int = 200,
    random_opening_plies: int = 6,
    multipv: int = 8,
    temperature_pawns: float = 1.0,
    seed: int = 0,
    progress_dir: str | None = None,
) -> dict:
    """Run n_games of Stockfish self-play across n_workers processes,
    capturing multipv-soft-target NPZ at output_path.

    Schema described at the top of this file.
    """
    import shutil
    if stockfish_path is None:
        stockfish_path = shutil.which("stockfish")
    if stockfish_path is None:
        raise FileNotFoundError("stockfish binary not found")

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    games_per_worker = [n_games // n_workers] * n_workers
    for i in range(n_games % n_workers):
        games_per_worker[i] += 1

    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)
        # Clear stale progress files from any prior run.
        for f in os.listdir(progress_dir):
            if f.startswith(".progress_w") and f.endswith(".json"):
                try: os.remove(os.path.join(progress_dir, f))
                except OSError: pass

    worker_args = [
        (i, games_per_worker[i], stockfish_path, sf_elo, sf_skill,
         depth, max_plies, random_opening_plies, multipv, temperature_pawns,
         seed + i * 10007, progress_dir)
        for i in range(n_workers)
    ]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_generate, worker_args)
    dt = time.time() - t0

    states = np.concatenate([r["states"] for r in results], axis=0)
    moves = np.concatenate([r["moves"] for r in results], axis=0)
    multipv_indices = np.concatenate([r["multipv_indices"] for r in results], axis=0)
    multipv_logprobs = np.concatenate([r["multipv_logprobs"] for r in results], axis=0)
    zs = np.concatenate([r["zs"] for r in results], axis=0)
    all_pgns = [pgn for r in results for pgn in r.get("pgns", [])]
    n_below_k = sum(r["n_below_k"] for r in results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states, moves=moves, zs=zs,
        multipv_indices=multipv_indices, multipv_logprobs=multipv_logprobs,
        K=np.array(multipv, dtype=np.int32),
        temperature_pawns=np.array(temperature_pawns, dtype=np.float32),
    )

    # Sibling .pgn file: full game library, one PGN per game, separated by
    # blank lines (standard concat-PGN format). Lets us replay any game
    # without re-running Stockfish.
    pgn_path = os.path.splitext(output_path)[0] + ".pgn"
    with open(pgn_path, "w") as f:
        f.write("\n\n".join(all_pgns))

    return {
        "n_games": n_games,
        "n_positions": int(states.shape[0]),
        "wall_seconds": dt,
        "output": output_path,
        "pgn_output": pgn_path,
        "n_pgns": len(all_pgns),
        "depth": depth,
        "multipv": multipv,
        "temperature_pawns": temperature_pawns,
        "n_positions_below_k": int(n_below_k),
    }
