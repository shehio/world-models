"""Match two policies head-to-head, with color alternation across games.

A policy is any callable: policy(board: chess.Board) -> chess.Move.
"""
from __future__ import annotations

import random
import shutil
from contextlib import contextmanager
from typing import Callable, Iterator

import chess
import chess.engine
import torch

from .config import Config
from .mcts import run_mcts, run_mcts_batched, select_move


Policy = Callable[[chess.Board], chess.Move]


def random_policy(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))


@contextmanager
def stockfish_engine(
    path: str | None = None,
    *,
    elo: int | None = None,
    skill: int | None = None,
    threads: int = 1,
) -> Iterator[chess.engine.SimpleEngine]:
    """Context manager that opens a Stockfish UCI engine and configures strength.

    elo:    if set, enables UCI_LimitStrength and pins UCI_Elo (min 1320).
    skill:  if set (0..20), passes Skill Level instead.

    Use only one of elo/skill. depth/time limits are passed at .play() call.
    """
    binary = path or shutil.which("stockfish")
    if binary is None:
        raise FileNotFoundError("stockfish binary not on PATH")
    eng = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        opts: dict = {"Threads": threads}
        if elo is not None:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = int(elo)
        if skill is not None:
            opts["Skill Level"] = int(skill)
        eng.configure(opts)
        yield eng
    finally:
        eng.quit()


def stockfish_policy(engine: chess.engine.SimpleEngine, depth: int = 1) -> Policy:
    """Wrap an opened Stockfish engine as a Policy."""
    limit = chess.engine.Limit(depth=depth)

    def policy(board: chess.Board) -> chess.Move:
        result = engine.play(board, limit)
        return result.move

    return policy


def network_policy(
    network,
    cfg: Config,
    device: torch.device,
    sims: int | None = None,
    batch_size: int = 1,
) -> Policy:
    """A policy callable wrapping (network + MCTS). batch_size>1 uses batched MCTS."""
    sims = sims if sims is not None else cfg.sims_eval

    def policy(board: chess.Board) -> chess.Move:
        if batch_size > 1:
            visits = run_mcts_batched(
                board, network,
                num_sims=sims,
                c_puct=cfg.c_puct,
                add_root_noise=False,
                device=device,
                batch_size=batch_size,
            )
        else:
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
