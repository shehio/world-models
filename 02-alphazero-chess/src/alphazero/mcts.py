"""PUCT Monte Carlo Tree Search.

Each Node holds:
    P(s, a)  : prior from network
    N(s, a)  : visit count
    W(s, a)  : total action value
    Q(s, a)  : W / N
    children : dict[move -> Node]

Selection step:
    a* = argmax_a [ Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a)) ]

Expansion: when a leaf is reached, run the network on the position to
get (π, v). Mask π by legal moves, normalize, store as priors. Initialize
N=W=0 for each child.

Backup: walk back up, alternating sign of v at every ply (zero-sum).

Root noise: at the search root, mix Dirichlet(α) noise into priors with
weight ε. This is the only source of training-time exploration.

Action selection: at the root, pick argmax(N) for greedy / sample N^(1/τ)
for stochastic.

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import chess


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def run_mcts(
    board: chess.Board,
    network,  # AlphaZeroNet
    num_sims: int,
    c_puct: float,
    add_root_noise: bool,
) -> dict[chess.Move, int]:
    """Run num_sims simulations from board. Returns visit counts at root."""
    raise NotImplementedError


def select_action(visit_counts: dict[chess.Move, int], temperature: float) -> chess.Move:
    """Sample/argmax based on temperature. τ→0 is argmax."""
    raise NotImplementedError
