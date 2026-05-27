"""Elo eval — MuZero net vs Stockfish UCI at a fixed limit_strength.

Wraps the MuZero (network + MCTS over learned dynamics) as a callable
wm_chess.arena.Policy. The network is asked for one move at a time:
encode the position → run MCTS → pick argmax over root visit counts →
decode the action to a chess.Move. Legal-move masking happens at the
MCTS root only, matching self-play.

Reuses wm_chess.arena.{stockfish_engine, stockfish_policy, play_match}
so MuZero numbers slot into the same shape the rest of the project
uses (score = (wins + 0.5 * draws) / games against a UCI anchor).
"""
from __future__ import annotations

from dataclasses import replace

import chess
import torch

from wm_chess.arena import play_match, stockfish_engine, stockfish_policy
from wm_chess.board import decode_move, encode_board, encode_move

from .config import MuZeroConfig
from .mcts import run_mcts, select_action


def muzero_policy(
    network,
    cfg: MuZeroConfig,
    device: torch.device | None = None,
    *,
    sims: int | None = None,
):
    """Wrap MuZeroNet + MCTS as a wm_chess.arena.Policy (eval-mode, no Dirichlet)."""
    cfg_used = cfg if sims is None else replace(cfg, num_simulations=sims)

    def policy(board: chess.Board) -> chess.Move:
        obs = torch.from_numpy(encode_board(board)).unsqueeze(0)
        legal = [encode_move(m, board) for m in board.legal_moves]
        root = run_mcts(
            network, obs, cfg_used,
            add_root_noise=False,
            legal_actions=legal,
            device=device,
        )
        action = select_action(root, temperature=0.0)
        return decode_move(action, board)

    return policy


def evaluate_vs_stockfish(
    network,
    cfg: MuZeroConfig,
    device: torch.device | None = None,
    *,
    stockfish_elo: int = 1320,
    n_games: int = 20,
    stockfish_depth: int = 1,
    sims: int | None = None,
    max_plies: int = 200,
) -> dict:
    """Play n_games vs Stockfish at UCI_Elo=stockfish_elo, depth=stockfish_depth.

    Returns play_match's stats dict from MuZero's POV:
      {games, wins, draws, losses, score}
    where score = (wins + 0.5*draws) / games.
    """
    network.eval()
    mz_pol = muzero_policy(network, cfg, device=device, sims=sims)
    with stockfish_engine(elo=stockfish_elo) as eng:
        sf_pol = stockfish_policy(eng, depth=stockfish_depth)
        return play_match(mz_pol, sf_pol, n_games=n_games, max_plies=max_plies)
