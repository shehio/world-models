"""Generate Stockfish self-play games with MULTIPV soft policy targets.

Public API (importable):
    play_one_game(engine, ...)                  -- play one game, return tensors + PGN
    generate_dataset_parallel(...)              -- run N games on M workers, save
    detect_stockfish_version(binary_path)       -- parse "Stockfish 14.1" header
    library_dataset_path(root, sf_version, ...) -- build the structured library path
    finalize_library_path(library_path)         -- assemble chunks → data.npz + games.pgn

Two storage modes:

    1. Legacy flat NPZ (backward compat with in-flight runs)
       - `--output path.npz` writes one big NPZ at the end
       - Sibling `path.pgn` written alongside

    2. Library mode (crash-safe, indexed by SF metadata)
       - `--library-root data/library` builds:
         <root>/sf-<version>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/
                 data.npz            <- assembled at end of run
                 games.pgn           <- assembled at end of run
                 metadata.json       <- generation metadata (SF version, args, host, timestamps)
                 chunks/             <- per-worker checkpoint files
                   worker_NN_chunk_MMM.npz   (every chunk_size games)
                   worker_NN.pgn             (appended per game)
       - On worker termination (spot reclaim, SIGKILL, anything): chunks
         survive. Call finalize_library_path() to recover whatever was
         saved up to the last chunk_size boundary per worker.

NPZ schema (same in both modes):

    states            (N, 19, 8, 8)  float32   - board positions
    moves             (N,)           int64     - SF's actually-played move idx
    zs                (N,)           float32   - game outcome from STM POV
    multipv_indices   (N, K)         int64     - top-K candidate move indices
    multipv_logprobs  (N, K)         float32   - softmax-logprobs over those K
    K                 ()             int32     - the K value used
    temperature_pawns ()             float32   - the softmax T used
"""
from __future__ import annotations

import glob
import json
import os
import platform
import random
import re
import socket
import subprocess
import time
from dataclasses import dataclass

import chess
import chess.engine
import chess.pgn
import numpy as np

from wm_chess.board import N_INPUT_PLANES, N_POLICY, encode_board, encode_move


# Clip mate scores so the softmax doesn't collapse to one-hot on detected mates.
# 1000 cp = 10 pawns. With T=1 pawn this means p_mate / p_next ≈ exp(10).
MATE_CP = 1000


@dataclass
class GameStats:
    plies: int
    outcome: float       # +1 white / -1 black / 0 draw
    n_recorded: int      # positions saved
    n_below_k: int       # positions where SF returned fewer than K PVs


# ---------------------------------------------------------------------------
# Stockfish version detection + library path construction
# ---------------------------------------------------------------------------


_SF_HEADER_RE = re.compile(r"Stockfish\s+([\w\.\-]+)\b", re.IGNORECASE)


def detect_stockfish_version(stockfish_path: str | None = None) -> str:
    """Return a normalized Stockfish version string for path/metadata use.

    Parses the engine's UCI banner (first stdout line after launch). Falls
    back to "unknown" if anything goes wrong — we never want this to be a
    hard failure mid-run.
    """
    import shutil
    if stockfish_path is None:
        stockfish_path = shutil.which("stockfish")
    if stockfish_path is None:
        return "unknown"
    try:
        # Probe via stdin "uci" → engine prints "id name Stockfish XX..."
        proc = subprocess.run(
            [stockfish_path],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in (proc.stdout or "").splitlines():
            m = _SF_HEADER_RE.search(line)
            if m:
                # Normalize "14.1" → "14.1", "18" → "18", "dev-20250101" → "dev-20250101"
                return m.group(1).strip().replace(" ", "")
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def library_dataset_path(
    library_root: str,
    sf_version: str,
    depth: int,
    multipv: int,
    temperature_pawns: float,
    n_games: int,
    seed: int,
) -> str:
    """Build the structured path for a dataset in the library.

    Schema:
        <root>/sf-<version>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/

    Stable across re-runs (deterministic from args), so a previous run's
    chunks can be picked up by a re-run with the same args.
    """
    sf_part = f"sf-{sf_version}"
    cfg_part = f"d{depth}-mpv{multipv}-T{temperature_pawns:g}"
    run_part = f"g{n_games}-seed{seed}"
    return os.path.join(library_root, sf_part, cfg_part, run_part)


# ---------------------------------------------------------------------------
# Chunk I/O — crash-safe incremental saves
# ---------------------------------------------------------------------------


def _atomic_npz_save(path: str, **arrays) -> None:
    """Atomic NPZ write: write to a sibling tmp file then os.replace.

    np.savez_compressed auto-appends ".npz" unless the path already ends in
    it, so we keep the tmp suffix BEFORE the extension to avoid the implicit
    rename.
    """
    base, ext = os.path.splitext(path)
    tmp = base + ".tmp" + ext  # e.g. foo.tmp.npz
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _save_worker_chunk(chunks_dir: str, worker_id: int, chunk_idx: int,
                       states, moves, mpv_idx, mpv_logp, zs, multipv: int) -> str:
    """Write one chunk of accumulated games for a worker."""
    os.makedirs(chunks_dir, exist_ok=True)
    path = os.path.join(chunks_dir,
                        f"worker_{worker_id:02d}_chunk_{chunk_idx:04d}.npz")
    _atomic_npz_save(
        path,
        states=(np.stack(states) if states
                else np.zeros((0, N_INPUT_PLANES, 8, 8), dtype=np.float32)),
        moves=np.array(moves, dtype=np.int64),
        multipv_indices=(np.array(mpv_idx, dtype=np.int64)
                         if mpv_idx else np.zeros((0, multipv), dtype=np.int64)),
        multipv_logprobs=(np.stack(mpv_logp) if mpv_logp
                          else np.zeros((0, multipv), dtype=np.float32)),
        zs=np.array(zs, dtype=np.float32),
    )
    return path


def _append_worker_pgn(chunks_dir: str, worker_id: int, pgn_text: str) -> None:
    """Append one PGN to this worker's accumulating .pgn file."""
    os.makedirs(chunks_dir, exist_ok=True)
    path = os.path.join(chunks_dir, f"worker_{worker_id:02d}.pgn")
    with open(path, "a") as f:
        f.write(pgn_text)
        f.write("\n\n")


def finalize_library_path(
    library_path: str,
    multipv: int,
    temperature_pawns: float,
) -> dict:
    """Read all chunks under <library_path>/chunks/ and assemble:
        <library_path>/data.npz   (concatenated tensors)
        <library_path>/games.pgn  (concatenated PGNs in worker order, then game order)

    Safe to call at any time — even if data gen was killed. Returns
    {n_positions, n_games_pgn, used_chunks}. If chunks_dir doesn't exist
    or is empty, raises FileNotFoundError.

    This is the recovery path. After spot reclamation: re-run with the
    same args (idempotent — workers detect existing chunks) or just call
    this on the existing chunks to assemble what we have.
    """
    chunks_dir = os.path.join(library_path, "chunks")
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "worker_*_chunk_*.npz")))
    if not chunk_files:
        raise FileNotFoundError(f"no chunk files under {chunks_dir}")

    arrays: dict[str, list] = {k: [] for k in
                               ("states", "moves", "multipv_indices",
                                "multipv_logprobs", "zs")}
    for cf in chunk_files:
        d = np.load(cf)
        for k in arrays:
            arrays[k].append(d[k])

    final_path = os.path.join(library_path, "data.npz")
    _atomic_npz_save(
        final_path,
        states=np.concatenate(arrays["states"], axis=0),
        moves=np.concatenate(arrays["moves"], axis=0),
        multipv_indices=np.concatenate(arrays["multipv_indices"], axis=0),
        multipv_logprobs=np.concatenate(arrays["multipv_logprobs"], axis=0),
        zs=np.concatenate(arrays["zs"], axis=0),
        K=np.array(multipv, dtype=np.int32),
        temperature_pawns=np.array(temperature_pawns, dtype=np.float32),
    )
    n_positions = int(np.concatenate(arrays["states"], axis=0).shape[0])

    # PGN concatenation
    pgn_files = sorted(glob.glob(os.path.join(chunks_dir, "worker_*.pgn")))
    final_pgn = os.path.join(library_path, "games.pgn")
    n_games_pgn = 0
    with open(final_pgn, "w") as out:
        for pf in pgn_files:
            with open(pf) as f:
                text = f.read()
            out.write(text)
            if not text.endswith("\n\n"):
                out.write("\n\n")
            n_games_pgn += text.count('[Event "')

    return {
        "n_positions": n_positions,
        "n_games_pgn": n_games_pgn,
        "used_chunks": len(chunk_files),
        "data_npz": final_path,
        "games_pgn": final_pgn,
    }


# ---------------------------------------------------------------------------
# Soft-target distribution helpers
# ---------------------------------------------------------------------------


def _score_to_cp(score: chess.engine.PovScore, mate_cp: int = MATE_CP) -> float:
    """PovScore → centipawns from STM's POV. Mate scores clip to ±mate_cp."""
    return float(score.relative.score(mate_score=mate_cp))


def _multipv_to_distribution(
    info_list: list[dict],
    board: chess.Board,
    temperature_pawns: float,
    K: int,
) -> tuple[list[int], np.ndarray, int]:
    """engine.analyse(MultiPV=K) result → (indices, logprobs, n_valid)."""
    valid = []
    for info in info_list[:K]:
        pv = info.get("pv")
        if not pv:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        idx = encode_move(move, board)
        cp = _score_to_cp(info["score"])
        valid.append((idx, cp))

    n_valid = len(valid)
    indices = [v[0] for v in valid]

    if n_valid == 0:
        return [-1] * K, np.full((K,), -np.inf, dtype=np.float32), 0

    scores_cp = np.array([v[1] for v in valid], dtype=np.float64)
    scaled = scores_cp / (100.0 * temperature_pawns)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    probs = weights / weights.sum()
    valid_logprobs = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)

    out_indices = indices + [-1] * (K - n_valid)
    out_logprobs = np.full((K,), -np.inf, dtype=np.float32)
    out_logprobs[:n_valid] = valid_logprobs
    return out_indices, out_logprobs, n_valid


# ---------------------------------------------------------------------------
# Game playing + PGN rendering
# ---------------------------------------------------------------------------


def _board_to_pgn(board_initial: chess.Board, moves: list[chess.Move],
                  outcome_white_pov: float, headers: dict) -> str:
    """Replay a move list and emit a PGN string for the game library."""
    game = chess.pgn.Game()
    for k, v in headers.items():
        game.headers[k] = str(v)
    node = game
    b = board_initial.copy()
    for mv in moves:
        if mv not in b.legal_moves:
            break
        node = node.add_variation(mv)
        b.push(mv)
    result_str = ("1-0" if outcome_white_pov > 0
                  else "0-1" if outcome_white_pov < 0
                  else "1/2-1/2")
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
) -> tuple[list[np.ndarray], list[int], list[list[int]],
           list[np.ndarray], list[float], GameStats, str]:
    """Play one Stockfish self-play game with multipv-soft-target recording.

    Returns (states, played_moves, multipv_indices, multipv_logprobs,
             zs, GameStats, pgn_text).
    """
    board = chess.Board()
    initial_board = board.copy()
    all_moves_played: list[chess.Move] = []

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
    n_below_k = 0
    limit = chess.engine.Limit(depth=depth)

    while not board.is_game_over(claim_draw=True) and len(states) < max_plies:
        info_list = engine.analyse(board, limit, multipv=multipv)
        if not isinstance(info_list, list):
            info_list = [info_list]

        mpv_idx, mpv_logp, n_valid = _multipv_to_distribution(
            info_list, board, temperature_pawns=temperature_pawns, K=multipv,
        )
        if n_valid < multipv:
            n_below_k += 1

        if not info_list or not info_list[0].get("pv"):
            break
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
    return (states, played_moves, multipv_indices, multipv_logprobs,
            zs, GameStats(plies=len(states), outcome=z_white,
                          n_recorded=len(states), n_below_k=n_below_k),
            pgn_text)


# ---------------------------------------------------------------------------
# Parallel data generation (chunked, crash-safe)
# ---------------------------------------------------------------------------


def _worker_generate(args) -> dict:
    (worker_id, n_games, stockfish_path, sf_elo, sf_skill,
     depth, max_plies, random_opening_plies, multipv,
     temperature_pawns, seed, progress_dir, chunks_dir, chunk_size) = args

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

    def _write_progress(payload: dict) -> None:
        if not progress_path:
            return
        tmp = progress_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, progress_path)

    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    t_start = time.time()

    # Per-chunk accumulation buffers. Flushed to disk every chunk_size games.
    buf_states, buf_moves, buf_mpv_idx, buf_mpv_logp, buf_zs = [], [], [], [], []
    chunk_idx = 0
    cumulative_done = 0
    cumulative_positions = 0
    total_below_k = 0
    total_plies = 0

    try:
        engine.configure(eng_kwargs)
        _write_progress({
            "worker_id": worker_id, "games_done": 0, "games_total": n_games,
            "positions": 0, "plies": 0, "elapsed_s": 0.0, "ts": time.time(),
            "phase": "start",
        })

        for g in range(n_games):
            states, played, mpv_idx, mpv_logp, zs, stats, pgn_text = play_one_game(
                engine,
                depth=depth, max_plies=max_plies,
                random_opening_plies=random_opening_plies,
                multipv=multipv, temperature_pawns=temperature_pawns, rng=rng,
            )
            buf_states.extend(states)
            buf_moves.extend(played)
            buf_mpv_idx.extend(mpv_idx)
            buf_mpv_logp.extend(mpv_logp)
            buf_zs.extend(zs)
            cumulative_positions += len(states)
            cumulative_done += 1
            total_below_k += stats.n_below_k
            total_plies += stats.plies

            # PGN appended immediately so even a per-game crash doesn't lose it.
            if chunks_dir:
                _append_worker_pgn(chunks_dir, worker_id, pgn_text)

            # Flush every chunk_size games OR at the end.
            flush = ((g + 1) % chunk_size == 0) or (g == n_games - 1)
            if flush and chunks_dir:
                _save_worker_chunk(
                    chunks_dir, worker_id, chunk_idx,
                    buf_states, buf_moves, buf_mpv_idx, buf_mpv_logp, buf_zs,
                    multipv=multipv,
                )
                chunk_idx += 1
                buf_states, buf_moves, buf_mpv_idx, buf_mpv_logp, buf_zs = (
                    [], [], [], [], []
                )

            _write_progress({
                "worker_id": worker_id, "games_done": cumulative_done,
                "games_total": n_games,
                "positions": cumulative_positions, "plies": total_plies,
                "elapsed_s": time.time() - t_start, "ts": time.time(),
                "phase": "running",
            })

        _write_progress({
            "worker_id": worker_id, "games_done": n_games, "games_total": n_games,
            "positions": cumulative_positions, "plies": total_plies,
            "elapsed_s": time.time() - t_start, "ts": time.time(),
            "phase": "done",
        })

        return {
            "worker_id": worker_id,
            "n_games": n_games,
            "n_positions": cumulative_positions,
            "n_below_k": total_below_k,
            "chunks_written": chunk_idx,
        }
    finally:
        engine.quit()


def _stockfish_path_or_die(stockfish_path: str | None) -> str:
    import shutil
    if stockfish_path is None:
        stockfish_path = shutil.which("stockfish")
    if stockfish_path is None:
        raise FileNotFoundError("stockfish binary not found")
    return stockfish_path


def generate_dataset_parallel(
    output_path: str | None,
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
    *,
    library_root: str | None = None,
    chunk_size: int = 50,
) -> dict:
    """Run n_games of Stockfish self-play, writing chunked output for
    crash safety.

    Two output modes:
      - output_path != None: legacy flat NPZ (assembled at end). Chunks
        written to <dirname(output_path)>/.<basename>_chunks/.
      - library_root != None: structured library path. Chunks written to
        <lib_path>/chunks/, finalized to <lib_path>/data.npz + games.pgn
        + metadata.json.

    Either (or both) may be set. If both are set, chunks live in the
    library path and the flat NPZ is also written as a convenience copy.
    """
    sf_bin = _stockfish_path_or_die(stockfish_path)
    sf_version = detect_stockfish_version(sf_bin)

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # Decide where chunks live.
    if library_root:
        lib_path = library_dataset_path(
            library_root, sf_version, depth, multipv, temperature_pawns,
            n_games, seed,
        )
        chunks_dir = os.path.join(lib_path, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
    elif output_path:
        lib_path = None
        chunks_dir = os.path.join(
            os.path.dirname(output_path) or ".",
            "." + os.path.splitext(os.path.basename(output_path))[0] + "_chunks",
        )
        os.makedirs(chunks_dir, exist_ok=True)
    else:
        raise ValueError("must specify either output_path or library_root")

    # Distribute games across workers.
    games_per_worker = [n_games // n_workers] * n_workers
    for i in range(n_games % n_workers):
        games_per_worker[i] += 1

    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)
        for f in os.listdir(progress_dir):
            if f.startswith(".progress_w") and f.endswith(".json"):
                try: os.remove(os.path.join(progress_dir, f))
                except OSError: pass

    worker_args = [
        (i, games_per_worker[i], sf_bin, sf_elo, sf_skill,
         depth, max_plies, random_opening_plies, multipv, temperature_pawns,
         seed + i * 10007, progress_dir, chunks_dir, chunk_size)
        for i in range(n_workers)
    ]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_generate, worker_args)
    dt = time.time() - t0

    n_below_k = sum(r["n_below_k"] for r in results)

    # Always assemble from chunks. Even on partial run, this is the
    # canonical way to get a finalized data.npz.
    if lib_path is None:
        # Flat-mode finalize: read chunks, write to output_path + sibling pgn.
        chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "worker_*_chunk_*.npz")))
        arrays = {k: [] for k in ("states", "moves", "multipv_indices",
                                  "multipv_logprobs", "zs")}
        for cf in chunk_files:
            d = np.load(cf)
            for k in arrays:
                arrays[k].append(d[k])
        states = np.concatenate(arrays["states"], axis=0)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _atomic_npz_save(
            output_path,
            states=states,
            moves=np.concatenate(arrays["moves"], axis=0),
            multipv_indices=np.concatenate(arrays["multipv_indices"], axis=0),
            multipv_logprobs=np.concatenate(arrays["multipv_logprobs"], axis=0),
            zs=np.concatenate(arrays["zs"], axis=0),
            K=np.array(multipv, dtype=np.int32),
            temperature_pawns=np.array(temperature_pawns, dtype=np.float32),
        )
        pgn_files = sorted(glob.glob(os.path.join(chunks_dir, "worker_*.pgn")))
        pgn_path = os.path.splitext(output_path)[0] + ".pgn"
        with open(pgn_path, "w") as out:
            for pf in pgn_files:
                with open(pf) as f:
                    out.write(f.read())
                out.write("\n\n")
        return {
            "n_games": n_games, "n_positions": int(states.shape[0]),
            "wall_seconds": dt, "output": output_path, "pgn_output": pgn_path,
            "sf_version": sf_version, "n_positions_below_k": int(n_below_k),
            "depth": depth, "multipv": multipv,
            "temperature_pawns": temperature_pawns,
            "chunks_dir": chunks_dir,
        }

    # Library mode: full assemble + metadata.json
    meta = {
        "sf_version": sf_version,
        "depth": depth,
        "multipv": multipv,
        "temperature_pawns": temperature_pawns,
        "n_games_requested": n_games,
        "n_workers": n_workers,
        "chunk_size": chunk_size,
        "max_plies": max_plies,
        "random_opening_plies": random_opening_plies,
        "seed": seed,
        "wall_seconds": dt,
        "n_positions_below_k": int(n_below_k),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "started_ts": t0,
        "finished_ts": time.time(),
    }
    fin = finalize_library_path(lib_path, multipv=multipv,
                                temperature_pawns=temperature_pawns)
    meta.update({
        "n_positions": fin["n_positions"],
        "n_games_in_pgn": fin["n_games_pgn"],
        "used_chunks": fin["used_chunks"],
    })
    with open(os.path.join(lib_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return {
        "n_games": n_games,
        "n_positions": fin["n_positions"],
        "wall_seconds": dt,
        "output": fin["data_npz"],
        "pgn_output": fin["games_pgn"],
        "sf_version": sf_version,
        "library_path": lib_path,
        "chunks_dir": chunks_dir,
        "n_positions_below_k": int(n_below_k),
        "depth": depth,
        "multipv": multipv,
        "temperature_pawns": temperature_pawns,
    }
