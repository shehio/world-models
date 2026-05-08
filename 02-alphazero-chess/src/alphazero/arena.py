"""Match two policies head-to-head, with color alternation across games.

A policy is any callable: policy(board: chess.Board) -> chess.Move.
"""
from __future__ import annotations

import random
from typing import Callable

import chess
import torch

from .config import Config
from .mcts import run_mcts, select_move


Policy = Callable[[chess.Board], chess.Move]


def random_policy(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))


def network_policy(network, cfg: Config, device: torch.device, sims: int | None = None) -> Policy:
    sims = sims if sims is not None else cfg.sims_eval

    def policy(board: chess.Board) -> chess.Move:
        visits = run_mcts(
            board, network,
            num_sims=sims,
            c_puct=cfg.c_puct,
            add_root_noise=False,
            device=device,
        )
        return select_move(visits, temperature=0.0)

    return policy


def play_match(
    policy_a: Policy,
    policy_b: Policy,
    n_games: int = 20,
    max_plies: int = 400,
) -> dict:
    """Play n_games with colors alternating. Returns stats from policy_a's POV."""
    a_wins = a_losses = draws = 0

    for g in range(n_games):
        a_is_white = (g % 2 == 0)
        white = policy_a if a_is_white else policy_b
        black = policy_b if a_is_white else policy_a

        board = chess.Board()
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            mover = white if board.turn == chess.WHITE else black
            board.push(mover(board))
            ply += 1

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
        else:
            white_won = outcome.winner == chess.WHITE
            a_won = (white_won and a_is_white) or (not white_won and not a_is_white)
            if a_won:
                a_wins += 1
            else:
                a_losses += 1

    return {
        "games": n_games,
        "wins": a_wins,
        "draws": draws,
        "losses": a_losses,
        "score": (a_wins + 0.5 * draws) / n_games,
    }
