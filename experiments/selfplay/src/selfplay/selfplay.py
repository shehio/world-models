"""Play one self-play game with MCTS-from-network. Yield training samples.

Two variants here:
  play_game        — every move uses the same `sims` count. Standard AZ.
  play_game_pcr    — Playout Cap Randomization (KataGo trick): on a small
                     fraction of moves use FULL sims with Dirichlet noise
                     and record the policy target; on the rest use REDUCED
                     sims without noise and DON'T record a policy target.
                     Game still completes; value target z covers all
                     positions whose target we keep.

Why PCR works:
  - At low sims, MCTS visit distributions are ~the network's prior — not
    a meaningful improvement target.
  - At high sims, MCTS produces a sharply better policy than the network.
  - So spending compute on high-sim moves *whose targets we don't use*
    is waste. PCR shifts compute to: cheap fast moves for play, expensive
    sharp moves only when we're going to record the result.
  - KataGo paper Table 4: +80–150 Elo at fixed compute on Go.
"""
from __future__ import annotations

import random

import chess
import numpy as np
import torch

from wm_chess.board import encode_board, encode_history
from wm_chess.config import Config
from wm_chess.mcts import run_mcts, run_mcts_batched, select_move, visits_to_distribution


def _encode_state_for_storage(boards: list[chess.Board], n_history: int) -> np.ndarray:
    """Encoder used when stashing samples in the replay buffer."""
    if n_history > 1:
        return encode_history(boards, n_history=n_history)
    return encode_board(boards[-1])


def _run_mcts(
    board, network, sims, cfg, device,
    add_root_noise: bool, batch_size: int,
    game_history: list[chess.Board] | None = None,
    n_history: int = 1,
):
    """Internal: dispatches to batched or sequential MCTS based on batch_size."""
    if batch_size > 1:
        return run_mcts_batched(
            board, network, num_sims=sims, c_puct=cfg.c_puct,
            add_root_noise=add_root_noise, device=device, batch_size=batch_size,
            dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps,
            game_history=game_history, n_history=n_history,
        )
    return run_mcts(
        board, network, num_sims=sims, c_puct=cfg.c_puct,
        add_root_noise=add_root_noise, device=device,
        dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps,
        game_history=game_history, n_history=n_history,
    )


def play_game(
    network,
    cfg: Config,
    device: torch.device,
    sims: int | None = None,
    batch_size: int = 1,
    n_history: int = 1,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int]:
    """Standard self-play: every move uses `sims` MCTS sims with root noise.

    Returns (samples, z_white_pov, ply_count).
    samples: list of (state_planes, pi_4672, z_for_side_to_move) per ply.

    n_history: 1 → legacy 19-plane encoding (just current board).
               >1 → encode_history with that many stacked positions.
    """
    sims = sims if sims is not None else cfg.sims_train
    board = chess.Board()
    history: list[tuple[np.ndarray, np.ndarray, bool]] = []
    ply = 0
    # Boards before the current one (most recent last). Grows as game progresses.
    game_history_boards: list[chess.Board] = []

    while not board.is_game_over(claim_draw=True) and ply < cfg.max_plies:
        visits = _run_mcts(board, network, sims, cfg, device,
                           add_root_noise=True, batch_size=batch_size,
                           game_history=game_history_boards, n_history=n_history)
        pi = visits_to_distribution(visits, board)
        # The state we record is what the network would see at this position.
        full_history = game_history_boards + [board.copy(stack=False)]
        state = _encode_state_for_storage(full_history, n_history)
        history.append((state, pi, board.turn == chess.WHITE))
        temp = 1.0 if ply < cfg.temp_moves else 0.0
        move = select_move(visits, temperature=temp)
        # Snapshot before push so this becomes part of the prior history.
        game_history_boards.append(board.copy(stack=False))
        board.push(move)
        # Cap history at last n_history-1 boards to keep memory bounded.
        # encode_history only uses last n_history entries (game_history + current);
        # we keep one extra so the caller can always append `board` to it.
        if n_history > 1 and len(game_history_boards) > n_history:
            game_history_boards = game_history_boards[-n_history:]
        ply += 1

    z_white = _outcome(board)
    samples = [(s, p, z_white if w else -z_white) for s, p, w in history]
    return samples, z_white, ply


def play_game_pcr(
    network,
    cfg: Config,
    device: torch.device,
    sims_full: int = 200,
    sims_reduced: int = 50,
    p_full: float = 0.25,
    batch_size: int = 8,
    n_history: int = 1,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int, dict]:
    """Self-play with Playout Cap Randomization.

    For each move:
      - With probability p_full:  use sims_full sims + Dirichlet noise + record (state, π, z) target
      - With probability 1-p_full: use sims_reduced sims + NO noise + DO NOT record this position's target

    z is the game outcome (+1 white wins / -1 black wins / 0 draw); written into
    samples for every recorded position from that position's side-to-move POV.

    Returns (samples, z_white_pov, ply_count, stats) where stats includes counts of
    full vs reduced moves so we can verify the ratio.
    """
    board = chess.Board()
    # history: (state, pi, was_white, recorded_flag)
    history: list[tuple[np.ndarray, np.ndarray, bool, bool]] = []
    ply = 0
    full_count = reduced_count = 0
    game_history_boards: list[chess.Board] = []

    while not board.is_game_over(claim_draw=True) and ply < cfg.max_plies:
        is_full = random.random() < p_full
        if is_full:
            sims = sims_full
            add_root_noise = True
            full_count += 1
        else:
            sims = sims_reduced
            add_root_noise = False
            reduced_count += 1

        visits = _run_mcts(board, network, sims, cfg, device,
                           add_root_noise=add_root_noise, batch_size=batch_size,
                           game_history=game_history_boards, n_history=n_history)

        # Only the FULL-sim positions become training samples.
        if is_full:
            pi = visits_to_distribution(visits, board)
            full_history = game_history_boards + [board.copy(stack=False)]
            state = _encode_state_for_storage(full_history, n_history)
            history.append((state, pi, board.turn == chess.WHITE, True))

        temp = 1.0 if ply < cfg.temp_moves else 0.0
        move = select_move(visits, temperature=temp)
        game_history_boards.append(board.copy(stack=False))
        board.push(move)
        if n_history > 1 and len(game_history_boards) > n_history:
            game_history_boards = game_history_boards[-n_history:]
        ply += 1

    z_white = _outcome(board)

    samples = [(s, p, z_white if w else -z_white) for s, p, w, _ in history]
    stats = {
        "full_moves": full_count,
        "reduced_moves": reduced_count,
        "recorded_targets": len(samples),
    }
    return samples, z_white, ply, stats


def _outcome(board: chess.Board) -> float:
    """Game result from white's POV: +1 / -1 / 0."""
    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        return 0.0
    return 1.0 if o.winner == chess.WHITE else -1.0
