"""PUCT Monte Carlo Tree Search over the Go rules engine.

Algorithm identical to `wm_chess.mcts.run_mcts`. Only the per-game state
type differs: chess.Board → distill_go.GoBoard. Move encoding is flat int
(0..S²-1 = board points, S² = pass) so the dense network policy logits
can be indexed directly without a sparse mapping step.

Sequential only. Batched MCTS (virtual-loss style) could be added later;
9x9 demo doesn't need it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .board import BLACK, GoBoard, board_to_history_planes, board_to_planes
from .config import GoConfig


def _encode_state(boards: list[GoBoard], n_input_planes: int) -> np.ndarray:
    if n_input_planes == 17:
        return board_to_history_planes(boards, n_history=8)
    return board_to_planes(boards[-1])


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    is_expanded: bool = False


def _terminal_value(board: GoBoard) -> float:
    """Game outcome from the POV of board.to_move at terminal."""
    winner = board.winner()
    return 1.0 if winner == board.to_move else -1.0


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """logits + mask → prob distribution over legal moves only."""
    masked = np.where(mask > 0, logits, -1e9)
    masked = masked - masked.max()
    probs = np.exp(masked) * (mask > 0)
    s = probs.sum()
    if s > 0:
        return probs / s
    legal_count = (mask > 0).sum()
    return mask.astype(np.float32) / max(legal_count, 1)


def _expand(
    node: Node,
    board: GoBoard,
    network,
    device: torch.device,
    history: list[GoBoard],
    n_input_planes: int,
) -> float:
    """Run network on board, create children for each legal move, return STM value."""
    state = _encode_state(history, n_input_planes)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
    network.eval()
    with torch.no_grad():
        logits, value = network(state_t)
    logits_np = logits[0].cpu().numpy()
    v = float(value.item())

    mask = board.legal_mask().astype(np.float32)
    probs = _masked_softmax(logits_np, mask)

    for idx in range(len(mask)):
        if mask[idx] > 0:
            node.children[idx] = Node(prior=float(probs[idx]))
    node.is_expanded = True
    return v


def _add_dirichlet_noise(node: Node, alpha: float, eps: float) -> None:
    if not node.children:
        return
    noise = np.random.dirichlet([alpha] * len(node.children))
    for (_idx, child), n in zip(node.children.items(), noise):
        child.prior = (1 - eps) * child.prior + eps * float(n)


def _select_child(parent: Node, c_puct: float) -> tuple[int, Node]:
    sqrt_total = math.sqrt(max(parent.visit_count, 1))
    best_score = -float("inf")
    best: Optional[tuple[int, Node]] = None
    for idx, child in parent.children.items():
        if child.visit_count > 0:
            q = -child.value_sum / child.visit_count
        else:
            q = 0.0
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best = (idx, child)
    assert best is not None
    return best


def run_mcts(
    board: GoBoard,
    network,
    *,
    num_sims: int,
    c_puct: float = 1.5,
    add_root_noise: bool = False,
    dirichlet_alpha: float = 0.15,
    dirichlet_eps: float = 0.25,
    device: torch.device | None = None,
    n_input_planes: int = 4,
    game_history: list[GoBoard] | None = None,
) -> dict[int, int]:
    """Run num_sims PUCT simulations from board. Returns visit count per move."""
    device = device or torch.device("cpu")
    history = list(game_history or [])
    root_history = history + [board.copy()]
    root = Node()
    _expand(root, board, network, device, root_history, n_input_planes)
    if add_root_noise:
        _add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

    for _ in range(num_sims):
        path = [root]
        sim_board = board.copy()
        descent_hist = list(root_history)
        node = root
        while node.is_expanded and not sim_board.is_game_over:
            move_idx, child = _select_child(node, c_puct)
            try:
                sim_board.play(move_idx)
            except Exception:
                # Defensive: if the child somehow encodes an illegal move
                # (shouldn't happen — mask is at expansion time), treat as terminal loss.
                value = -1.0
                for n_ in reversed(path):
                    n_.visit_count += 1
                    n_.value_sum += value
                    value = -value
                break
            descent_hist.append(sim_board.copy())
            path.append(child)
            node = child
        else:
            if sim_board.is_game_over:
                value = _terminal_value(sim_board)
            else:
                value = _expand(node, sim_board, network, device,
                                descent_hist, n_input_planes)
            for n_ in reversed(path):
                n_.visit_count += 1
                n_.value_sum += value
                value = -value
            continue
        # If we broke out due to defensive branch above, fall through.

    return {idx: child.visit_count for idx, child in root.children.items()}


def select_move(visit_counts: dict[int, int], temperature: float) -> int:
    """Sample a move idx from MCTS visits. τ→0 is argmax with random tie-break."""
    if not visit_counts:
        raise ValueError("no moves to select from")
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    if temperature <= 1e-6:
        max_c = counts.max()
        candidates = [m for m, c in zip(moves, counts) if c == max_c]
        return int(candidates[np.random.randint(len(candidates))])
    counts = counts ** (1.0 / temperature)
    if counts.sum() == 0:
        return int(np.random.choice(moves))
    probs = counts / counts.sum()
    return int(moves[np.random.choice(len(moves), p=probs)])


def visits_to_distribution(
    visit_counts: dict[int, int],
    cfg: GoConfig,
) -> np.ndarray:
    """Convert visit counts to a (S² + 1,) probability vector — π training target."""
    pi = np.zeros(cfg.policy_size, dtype=np.float32)
    total = sum(visit_counts.values())
    if total == 0:
        return pi
    for idx, count in visit_counts.items():
        pi[idx] = count / total
    return pi
