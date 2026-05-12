"""Correctness tests for run_mcts_batched.

Run with: uv run python tests/test_mcts_batched.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import numpy as np
import torch

from wm_chess.config import Config
from wm_chess.mcts import (
    Node,
    run_mcts,
    run_mcts_batched,
    select_move,
    visits_to_distribution,
)
from wm_chess.network import AlphaZeroNet


def _make_net(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Config()
    net = AlphaZeroNet(cfg).cpu()
    net.eval()
    return cfg, net, torch.device("cpu")


# ---------------------------------------------------------------------------
# Basic invariants — the same things test_mcts.py checks for sequential MCTS.
# ---------------------------------------------------------------------------

def test_visit_counts_sum_to_num_sims():
    """Total visits at root must equal num_sims, independent of batch_size."""
    cfg, net, dev = _make_net()
    board = chess.Board()
    for K in (1, 2, 4, 8, 16):
        for n in (8, 16, 50):
            visits = run_mcts_batched(
                board, net, num_sims=n, c_puct=cfg.c_puct,
                add_root_noise=False, device=dev, batch_size=K,
            )
            total = sum(visits.values())
            assert total == n, f"K={K} num_sims={n}: got {total} visits"


def test_all_visited_children_are_legal():
    cfg, net, dev = _make_net()
    board = chess.Board()
    visits = run_mcts_batched(board, net, num_sims=32, c_puct=cfg.c_puct,
                              add_root_noise=False, device=dev, batch_size=8)
    legal = set(board.legal_moves)
    for move in visits:
        assert move in legal, f"{move.uci()} not legal"


def test_terminal_root_handled():
    """If root is checkmate (no legal moves), batched MCTS returns empty dict."""
    cfg, net, dev = _make_net()
    board = chess.Board()
    # Fool's mate
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate()
    visits = run_mcts_batched(board, net, num_sims=8, c_puct=cfg.c_puct,
                              add_root_noise=False, device=dev, batch_size=4)
    assert visits == {}


def test_distribution_is_valid():
    cfg, net, dev = _make_net()
    board = chess.Board()
    visits = run_mcts_batched(board, net, num_sims=40, c_puct=cfg.c_puct,
                              add_root_noise=True, device=dev, batch_size=8)
    pi = visits_to_distribution(visits, board)
    assert pi.shape == (4672,)
    assert abs(pi.sum() - 1.0) < 1e-5
    assert (pi >= 0).all()


# ---------------------------------------------------------------------------
# Virtual-loss-specific invariants.
# ---------------------------------------------------------------------------

def _walk_all_nodes(root: Node):
    """DFS over the in-memory tree."""
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children.values())


def _build_tree_via_batched(num_sims: int, batch_size: int) -> Node:
    """Run batched MCTS once and capture the search tree by reconstructing
    the root from internal helpers. We patch in our own root creation here."""
    # Easier: directly call the sequence of helpers. Keep this simple by
    # actually running run_mcts_batched and then traversing the children
    # graph from a known root. To do this we need access to the root.
    # Instead, mirror the public interface: we run, then recurse via a
    # ground-truth structural check on visit_counts.
    raise NotImplementedError


def test_virtual_loss_zero_after_run():
    """Every node in the search tree must have virtual_loss == 0 after MCTS finishes.

    We can't easily get the root back from run_mcts_batched (it returns
    visit_counts, not the tree). Instead, monkey-patch the module to
    capture the root.
    """
    from wm_chess import mcts as mcts_mod

    cfg, net, dev = _make_net(seed=7)
    board = chess.Board()

    captured = {}
    orig_expand = mcts_mod._expand

    def capturing_expand(node, *args, **kwargs):
        # First call expands the root.
        if "root" not in captured:
            captured["root"] = node
        return orig_expand(node, *args, **kwargs)

    mcts_mod._expand = capturing_expand
    try:
        run_mcts_batched(board, net, num_sims=40, c_puct=cfg.c_puct,
                        add_root_noise=False, device=dev, batch_size=8)
    finally:
        mcts_mod._expand = orig_expand

    root = captured["root"]
    nonzero = [n for n in _walk_all_nodes(root) if n.virtual_loss != 0]
    assert not nonzero, f"{len(nonzero)} nodes have nonzero virtual_loss after MCTS"


def test_visit_count_matches_subtree_visits():
    """Standard MCTS invariant: a parent's visit_count equals 1 (for the
    expansion visit) + sum of children's visit_counts.

    This catches off-by-ones in the backup. Holds for both sequential and
    batched MCTS.
    """
    from wm_chess import mcts as mcts_mod

    cfg, net, dev = _make_net(seed=9)
    board = chess.Board()

    captured = {}
    orig_expand = mcts_mod._expand

    def capturing_expand(node, *args, **kwargs):
        if "root" not in captured:
            captured["root"] = node
        return orig_expand(node, *args, **kwargs)

    mcts_mod._expand = capturing_expand
    try:
        run_mcts_batched(board, net, num_sims=64, c_puct=cfg.c_puct,
                        add_root_noise=False, device=dev, batch_size=8)
    finally:
        mcts_mod._expand = orig_expand

    root = captured["root"]
    # Root: visit_count = sum of children visit_counts (root doesn't count its own
    # expansion as a visit in our backup loop — actually it does, because backup
    # walks every node in the path INCLUDING root). So the invariant for root is:
    #   root.visit_count == sum(c.visit_count for c in root.children).
    children_sum = sum(c.visit_count for c in root.children.values())
    assert root.visit_count == children_sum, (root.visit_count, children_sum)

    # For internal nodes: node.visit_count == 1 (its own expansion-visit) +
    # sum of children's visit_counts (visits that descended through it).
    # When the node has never been expanded (visit_count = 0 from root's
    # perspective; some children may not have been visited), its children
    # dict is empty.
    bad = []
    for n in _walk_all_nodes(root):
        if n is root or not n.is_expanded:
            continue
        s = sum(c.visit_count for c in n.children.values())
        if n.visit_count != 1 + s:
            bad.append((n.visit_count, 1 + s))
    assert not bad, f"first 3 mismatches: {bad[:3]}"


# ---------------------------------------------------------------------------
# Equivalence: at batch_size=1, batched ≡ sequential modulo random seeds.
# ---------------------------------------------------------------------------

def test_batched_at_k1_matches_sequential():
    """At batch_size=1 there's never an in-flight peer, so virtual_loss is
    always 0 at the time of selection. Batched and sequential should
    produce IDENTICAL visit counts when given identical random state.
    """
    cfg, net, dev = _make_net(seed=42)
    board = chess.Board()

    np.random.seed(0)
    visits_seq = run_mcts(board, net, num_sims=30, c_puct=cfg.c_puct,
                          add_root_noise=False, device=dev)

    np.random.seed(0)
    visits_bat = run_mcts_batched(board, net, num_sims=30, c_puct=cfg.c_puct,
                                  add_root_noise=False, device=dev, batch_size=1)

    # Compare visit counts directly.
    assert set(visits_seq.keys()) == set(visits_bat.keys())
    for m in visits_seq:
        assert visits_seq[m] == visits_bat[m], (m, visits_seq[m], visits_bat[m])


# ---------------------------------------------------------------------------
# Speed: batched should be measurably faster than sequential on a real net.
# ---------------------------------------------------------------------------

def test_batched_is_faster():
    """K=8 should beat K=1 (sequential equivalent) at num_sims=64."""
    cfg, net, dev = _make_net(seed=1)
    board = chess.Board()
    N = 64

    # Warm both paths — first call hits cold caches, BatchNorm running
    # stats path.
    run_mcts_batched(board, net, num_sims=8, c_puct=cfg.c_puct,
                    add_root_noise=False, device=dev, batch_size=8)
    run_mcts(board, net, num_sims=8, c_puct=cfg.c_puct,
             add_root_noise=False, device=dev)

    t0 = time.time()
    run_mcts(board, net, num_sims=N, c_puct=cfg.c_puct,
             add_root_noise=False, device=dev)
    t_seq = time.time() - t0

    t0 = time.time()
    run_mcts_batched(board, net, num_sims=N, c_puct=cfg.c_puct,
                    add_root_noise=False, device=dev, batch_size=8)
    t_bat = time.time() - t0

    speedup = t_seq / max(t_bat, 1e-6)
    print(f"  N={N}: batched {t_bat*1000:.0f}ms vs sequential {t_seq*1000:.0f}ms  ({speedup:.2f}x)")
    # Generous bar — batched should give at least 1.3x. Slack for slow CIs.
    assert speedup > 1.3, f"batched only {speedup:.2f}x faster than sequential"


def main():
    tests = [
        test_visit_counts_sum_to_num_sims,
        test_all_visited_children_are_legal,
        test_terminal_root_handled,
        test_distribution_is_valid,
        test_virtual_loss_zero_after_run,
        test_visit_count_matches_subtree_visits,
        test_batched_at_k1_matches_sequential,
        test_batched_is_faster,
    ]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
