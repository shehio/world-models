"""PUCT Monte Carlo Tree Search.

Conventions
-----------
Each node stores `value_sum` from the POV of the player to move at THAT
node. Backup walks up the path, negating the value at every step (since
adjacent plies have opposite players-to-move).

Selection at parent: we want Q(child) from PARENT's POV, so we negate
child's stored Q:
    Q_parent_pov = -value_sum(child) / max(1, visit_count(child))
    U(child)     = c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
    score(child) = Q_parent_pov + U(child)

Legal-move masking happens at every expansion (we have a real chess.Board
threaded through MCTS, unlike MuZero).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np
import torch

from .board import encode_board, encode_move, legal_move_mask


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    is_expanded: bool = False


def _terminal_value(board: chess.Board) -> float:
    """Game outcome from POV of side-to-move at `board`."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.0
    if outcome.winner is None:
        return 0.0
    # Side-to-move at terminal is the side that just got mated (or has no moves).
    return 1.0 if outcome.winner == board.turn else -1.0


def _expand(
    node: Node,
    board: chess.Board,
    network,
    device: torch.device,
) -> float:
    """Run network on board; create child Nodes with masked priors. Returns value v from board's POV."""
    state = encode_board(board)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
    network.eval()
    with torch.no_grad():
        logits, value = network(state_t)
    logits = logits[0].cpu().numpy()
    v = float(value.item())

    mask = legal_move_mask(board)
    masked = np.where(mask, logits, -1e9)
    masked = masked - masked.max()
    probs = np.exp(masked) * mask
    s = probs.sum()
    if s > 0:
        probs /= s
    else:
        # No legal moves should mean game over, but guard anyway.
        probs = mask.astype(np.float32) / max(mask.sum(), 1)

    for move in board.legal_moves:
        idx = encode_move(move, board)
        node.children[move] = Node(prior=float(probs[idx]))
    node.is_expanded = True
    return v


def _add_dirichlet_noise(node: Node, alpha: float, eps: float) -> None:
    if not node.children:
        return
    noise = np.random.dirichlet([alpha] * len(node.children))
    for (_move, child), n in zip(node.children.items(), noise):
        child.prior = (1 - eps) * child.prior + eps * float(n)


def _select_child(parent: Node, c_puct: float) -> tuple[chess.Move, Node]:
    sqrt_total = math.sqrt(max(parent.visit_count, 1))
    best_score = -float("inf")
    best: Optional[tuple[chess.Move, Node]] = None
    for move, child in parent.children.items():
        if child.visit_count > 0:
            q = -child.value_sum / child.visit_count  # parent-POV = -child-POV
        else:
            q = 0.0
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best = (move, child)
    assert best is not None, "select_child called on node with no children"
    return best


def run_mcts(
    board: chess.Board,
    network,
    num_sims: int,
    c_puct: float,
    add_root_noise: bool,
    device: torch.device,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> dict[chess.Move, int]:
    """Run num_sims simulations from `board`. Returns visit counts at root."""
    root = Node()
    _expand(root, board, network, device)
    if add_root_noise:
        _add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

    for _ in range(num_sims):
        path = [root]
        sim_board = board.copy(stack=False)
        node = root
        # Selection: descend until a leaf or terminal.
        while node.is_expanded and not sim_board.is_game_over(claim_draw=True):
            move, child = _select_child(node, c_puct)
            sim_board.push(move)
            path.append(child)
            node = child

        # Evaluation at leaf.
        if sim_board.is_game_over(claim_draw=True):
            value = _terminal_value(sim_board)
        else:
            value = _expand(node, sim_board, network, device)

        # Backup: each node stores from its own to-move POV; flip every ply.
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    return {move: child.visit_count for move, child in root.children.items()}


def select_move(visit_counts: dict[chess.Move, int], temperature: float) -> chess.Move:
    """Sample a move from visit counts. τ→0 is argmax (with tie-break)."""
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    if temperature <= 1e-6:
        # Greedy with random tie-break.
        max_count = counts.max()
        candidates = [m for m, c in zip(moves, counts) if c == max_count]
        return candidates[np.random.randint(len(candidates))]
    # Stochastic via N^(1/τ).
    counts = counts ** (1.0 / temperature)
    probs = counts / counts.sum()
    return moves[np.random.choice(len(moves), p=probs)]


def visits_to_distribution(visit_counts: dict[chess.Move, int], board: chess.Board) -> np.ndarray:
    """Convert MCTS visits to a 4672-dim distribution (training pi target)."""
    from .board import N_POLICY  # local import to avoid cycles in case of repacking

    pi = np.zeros(N_POLICY, dtype=np.float32)
    total = sum(visit_counts.values())
    if total == 0:
        return pi
    for move, count in visit_counts.items():
        pi[encode_move(move, board)] = count / total
    return pi
