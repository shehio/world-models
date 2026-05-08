"""Play one self-play game with MCTS-from-network. Yield training samples."""
from __future__ import annotations

import chess
import numpy as np
import torch

from .board import encode_board
from .config import Config
from .mcts import run_mcts, select_move, visits_to_distribution


def play_game(
    network,
    cfg: Config,
    device: torch.device,
    sims: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int]:
    """Returns (samples, z_white_pov, ply_count).

    samples: list of (state_planes, pi_4672, z_for_side_to_move) per ply.
    z_white_pov: +1 white won / -1 black won / 0 draw.
    """
    sims = sims if sims is not None else cfg.sims_train
    board = chess.Board()
    history: list[tuple[np.ndarray, np.ndarray, bool]] = []  # (state, pi, was_white_to_move)
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < cfg.max_plies:
        visits = run_mcts(
            board, network,
            num_sims=sims,
            c_puct=cfg.c_puct,
            add_root_noise=True,
            device=device,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_eps=cfg.dirichlet_eps,
        )
        pi = visits_to_distribution(visits, board)
        history.append((encode_board(board), pi, board.turn == chess.WHITE))
        temp = 1.0 if ply < cfg.temp_moves else 0.0
        move = select_move(visits, temperature=temp)
        board.push(move)
        ply += 1

    # Outcome from white's POV.
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        z_white = 0.0
    else:
        z_white = 1.0 if outcome.winner == chess.WHITE else -1.0

    samples = []
    for state, pi, was_white in history:
        z_pov = z_white if was_white else -z_white
        samples.append((state, pi, float(z_pov)))
    return samples, z_white, ply
