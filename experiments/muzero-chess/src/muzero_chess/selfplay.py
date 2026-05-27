"""Self-play game generation using MCTS over the learned dynamics.

One call to `play_game` plays a single chess game:
  - At every ply, run MCTS rooted at the current position (legal-action masked).
  - Convert root visit counts into a policy distribution π.
  - Sample an action under a temperature schedule (explore early, exploit late).
  - Apply the action to the real board (this is the only place we touch the
    chess rules engine below the MCTS root).
  - Loop until the board reports a terminal state, or cfg.max_plies is hit.

Outcome handling:
  - Terminal reward (mover's POV): +1 if the last action delivered checkmate,
    0 if it ended the game any other way (stalemate, repetition, etc.).
  - Value target z_t for ply t is then derived in GameRecord.from_trajectory by
    flipping sign with parity.
"""
from __future__ import annotations

import chess
import numpy as np
import torch

from wm_chess.board import decode_move, encode_board, encode_move

from .config import MuZeroConfig
from .mcts import root_visit_distribution, run_mcts_batched, select_action
from .replay import GameRecord


def play_game(
    network,
    cfg: MuZeroConfig,
    board: chess.Board | None = None,
    *,
    device: torch.device | None = None,
    add_root_noise: bool = True,
) -> GameRecord:
    """Play one full self-play game; return the trajectory as a GameRecord."""
    board = board if board is not None else chess.Board()
    starter_color = board.turn

    observations: list[np.ndarray] = []
    actions: list[int] = []
    pis: list[np.ndarray] = []

    while not board.is_game_over(claim_draw=True) and len(actions) < cfg.max_plies:
        obs = encode_board(board)
        legal_idxs = [encode_move(m, board) for m in board.legal_moves]
        if not legal_idxs:
            break

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        root = run_mcts_batched(
            network, obs_t, cfg,
            add_root_noise=add_root_noise,
            legal_actions=legal_idxs,
            device=device,
        )
        pi = root_visit_distribution(root, cfg.action_dim)

        temp = 1.0 if len(actions) < cfg.temp_moves else 0.0
        action = select_action(root, temperature=temp)

        observations.append(obs)
        actions.append(action)
        pis.append(pi)

        move = decode_move(action, board)
        assert move in board.legal_moves, (
            f"MCTS chose action {action} decoding to illegal move {move} "
            f"at FEN {board.fen()}"
        )
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        outcome_for_starter = 0.0
        terminal_reward_for_mover = 0.0
    else:
        outcome_for_starter = 1.0 if outcome.winner == starter_color else -1.0
        # Only checkmate counts as a "won by this move" terminal reward; everything
        # else (stalemate, repetition, etc.) is a 0-reward terminal.
        terminal_reward_for_mover = (
            1.0 if outcome.termination == chess.Termination.CHECKMATE else 0.0
        )

    return GameRecord.from_trajectory(
        observations=observations,
        actions=actions,
        pis=pis,
        outcome_for_starter=outcome_for_starter,
        terminal_reward_for_mover=terminal_reward_for_mover,
    )
