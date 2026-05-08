"""PUCT Monte Carlo Tree Search.

Two implementations live in this file:
  * `run_mcts`         — sequential, one network call per simulation. Easy
                         to read, slow because each call is batch-of-1.
  * `run_mcts_batched` — same algorithm, but K parallel descents collect
                         K leaves, evaluated in ONE batched network call.
                         Uses "virtual loss" so the K descents diversify
                         instead of all picking the same path.

Read the sequential version first — it's the textbook PUCT MCTS. The
batched version is exactly the same algorithm in a different control
flow shape.

Conventions (apply to both)
---------------------------
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
    # virtual_loss: how many in-flight (not yet backed up) descents are
    # currently passing through this node. Always 0 outside batched MCTS;
    # always ends at 0 even during batched MCTS once the batch is fully
    # backed up. See `run_mcts_batched` for how it's used.
    virtual_loss: int = 0
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


# =============================================================================
# BATCHED MCTS
# =============================================================================
#
# The plain `run_mcts` above does:
#   for _ in range(50):
#       descend → call network on 1 board → backup
#
# That's 50 forward passes, each with batch-size 1. Network calls have
# fixed per-call overhead (memory allocation, kernel launch, BatchNorm
# running-stats path), so 50 batch-1 calls are wasteful — a single
# batch-8 call takes only slightly longer than a batch-1 call.
#
# Batched MCTS does:
#   for _ in range(50 / K):                       # K parallel descents
#       descend K times → call network on K boards → backup K times
#
# That's 50/K forward passes, each with batch-size K. On our 390k-param
# net, K=8 gives ~3-5x wall-clock speedup at the same num_sims.
#
# THE PROBLEM: if we just descend K times naively, every descent picks
# the same path (same priors, same Q values), so we hit one leaf K times
# instead of K different leaves. No benefit.
#
# THE FIX: "virtual loss". While a descent is in-flight passing through
# a child node, we *pretend* that descent already returned a +1 reward
# from the child's POV. This makes the child look LESS attractive to
# its parent (parent's POV is the negation), so the next descent in
# the batch picks a different child. When the batch is backed up, we
# undo the virtual loss and apply the real reward.
#
# Concretely, while child.virtual_loss = VL:
#     Q(child, child's POV) ≈ (W + VL) / (N + VL)         ← higher
#     Q(child, parent's POV) = -Q(child, child's POV)      ← LOWER
# So the parent picks something else.


def _select_child_with_virtual_loss(
    parent: Node, c_puct: float
) -> tuple[chess.Move, Node]:
    """PUCT selection that accounts for in-flight (virtual_loss) descents.

    Equivalent to `_select_child` when virtual_loss is 0 everywhere.
    """
    # Total visits THROUGH parent = real visits + in-flight visits.
    # We use both for the U-term's parent total.
    parent_total = parent.visit_count + parent.virtual_loss
    sqrt_total = math.sqrt(max(parent_total, 1))

    best_score = -float("inf")
    best: Optional[tuple[chess.Move, Node]] = None
    for move, child in parent.children.items():
        n_eff = child.visit_count + child.virtual_loss
        if n_eff > 0:
            # Pretend in-flight descents will all return +1 from child's POV
            # (best possible for child = worst for parent). So:
            #     w_eff = real wins + virtual_loss * (+1)
            w_eff = child.value_sum + child.virtual_loss
            q_child_pov = w_eff / n_eff
        else:
            q_child_pov = 0.0
        q_parent_pov = -q_child_pov
        u = c_puct * child.prior * sqrt_total / (1 + n_eff)
        score = q_parent_pov + u
        if score > best_score:
            best_score = score
            best = (move, child)
    assert best is not None, "select_child called on node with no children"
    return best


def _expand_with_logits(node: Node, board: chess.Board, logits: np.ndarray) -> None:
    """Same as `_expand` but uses already-computed logits (no network call).

    Used in batched MCTS where we batch K boards into one network call,
    then call this once per board with that board's slice of the output.
    """
    mask = legal_move_mask(board)
    masked = np.where(mask, logits, -1e9)
    masked = masked - masked.max()
    probs = np.exp(masked) * mask
    s = probs.sum()
    if s > 0:
        probs /= s
    else:
        probs = mask.astype(np.float32) / max(mask.sum(), 1)

    for move in board.legal_moves:
        idx = encode_move(move, board)
        node.children[move] = Node(prior=float(probs[idx]))
    node.is_expanded = True


def _backup_with_virtual_loss(path: list[Node], value: float) -> None:
    """Walk path leaf → root: apply real value AND undo virtual loss.

    Virtual loss was applied to every CHILD during descent (path[1:],
    not path[0] which is root). So on backup we undo it for those nodes.
    The root never had virtual_loss applied (selection happens FROM root,
    not TO root), so we don't decrement it.
    """
    v = value
    n_path = len(path)
    for i, n in enumerate(reversed(path)):
        # Apply real visit + value (with sign-flip on each ply, like sequential).
        n.visit_count += 1
        n.value_sum += v
        # Undo virtual loss for nodes that received it (everyone except root).
        # In reversed order, root is the last (i == n_path - 1).
        if i < n_path - 1:
            n.virtual_loss -= 1
        v = -v


def run_mcts_batched(
    board: chess.Board,
    network,
    num_sims: int,
    c_puct: float,
    add_root_noise: bool,
    device: torch.device,
    batch_size: int = 8,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> dict[chess.Move, int]:
    """Batched PUCT MCTS. K parallel descents per batch, one batched network call.

    At batch_size=1 this behaves identically to `run_mcts` (no virtual loss
    ever activates because descent → backup happens within the same iter).

    Args
    ----
    batch_size : int
        How many descents to collect before each network forward pass.
        K=8 is a reasonable default on Mac CPU; bigger batches help less
        for tiny networks but more for bigger ones. Memory scales with K.
    """
    # Step 1 — Expand the root.
    # This is one (non-batched) network call. It's amortized over the
    # whole MCTS for this move, so batching it isn't worth the complexity.
    root = Node()
    _expand(root, board, network, device)
    if add_root_noise:
        _add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

    sims_done = 0
    while sims_done < num_sims:
        # K = how many descents in this batch. The last batch may be
        # smaller than `batch_size` if num_sims doesn't divide evenly.
        K = min(batch_size, num_sims - sims_done)

        # Step 2 — descend K times in parallel.
        # Each descent applies virtual_loss to the children it passes
        # through, so subsequent descents diversify into other parts of
        # the tree.
        leaves: list[tuple[list[Node], chess.Board]] = []
        for _ in range(K):
            path = [root]
            sim_board = board.copy(stack=False)
            node = root
            while node.is_expanded and not sim_board.is_game_over(claim_draw=True):
                move, child = _select_child_with_virtual_loss(node, c_puct)
                sim_board.push(move)
                child.virtual_loss += 1  # mark this child as "in flight"
                path.append(child)
                node = child
            leaves.append((path, sim_board))

        # Step 3 — split leaves into terminal (no network needed) and
        # non-terminal (needs network forward pass).
        nonterm: list[tuple[list[Node], chess.Board]] = []
        terminal: list[tuple[list[Node], chess.Board]] = []
        for path, b in leaves:
            (terminal if b.is_game_over(claim_draw=True) else nonterm).append((path, b))

        # Step 4 — ONE batched network call for all non-terminal leaves.
        # This is the whole point of batching.
        if nonterm:
            states = np.stack([encode_board(b) for _, b in nonterm])  # (K', 19, 8, 8)
            states_t = torch.from_numpy(states).to(device)
            network.eval()
            with torch.no_grad():
                logits_batch, values_batch = network(states_t)
            logits_np = logits_batch.cpu().numpy()         # (K', 4672)
            values_np = values_batch.cpu().numpy()         # (K',)

            for i, (path, b) in enumerate(nonterm):
                _expand_with_logits(path[-1], b, logits_np[i])
                _backup_with_virtual_loss(path, float(values_np[i]))

        # Step 5 — terminal leaves get their value from the rules of chess
        # (no network call needed).
        for path, b in terminal:
            _backup_with_virtual_loss(path, _terminal_value(b))

        sims_done += K

    # By construction, every node's virtual_loss was decremented exactly
    # once for each time it was incremented. So they're all 0 again.
    # (We don't assert this here for performance; the test suite checks it.)
    return {move: child.visit_count for move, child in root.children.items()}
