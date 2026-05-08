"""Self-play one game. Yields (state_planes, mcts_pi, z) tuples.

Procedure for one game:
    board = chess.Board()
    history = []
    while not terminal:
        visit_counts = run_mcts(board, network, sims, add_root_noise=True)
        pi = visit_counts_to_distribution(visit_counts, temperature_schedule(ply))
        history.append((encode_board(board), pi))
        move = select_action(visit_counts, temp)
        board.push(move)
    z = game_outcome(board)  # +1 / -1 / 0 from white's POV
    for (state, pi) in history:
        # z from the POV of side-to-move at THIS state
        yield (state, pi, z_for_side_to_move(state, z))

NOT YET IMPLEMENTED.
"""
from __future__ import annotations


def play_game(network, cfg) -> list[tuple]:
    """Returns list of (state_planes, pi_4672, z) — one per ply."""
    raise NotImplementedError
