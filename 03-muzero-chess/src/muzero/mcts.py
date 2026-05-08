"""PUCT MCTS in latent space.

Differences from AlphaZero MCTS:
  1. Root expansion calls model.initial_inference(obs) once.
  2. Every other expansion calls model.recurrent_inference(parent.state, action).
  3. Each edge stores its predicted reward; backups discount by γ and
     accumulate reward.
  4. Q values are normalized via running min/max so PUCT's c term has
     consistent scale even when value/reward magnitudes drift.

Pseudocode for one simulation:
    path = [root]
    while path[-1].is_expanded:
        action = puct_select(path[-1])
        path.append(path[-1].children[action])
    leaf = path[-1]
    s', r, π, v = model.recurrent_inference(parent.state, leaf.action)
    leaf.state = s'; leaf.reward = r
    leaf.expand(π)  # masked by legal moves only at root
    backup(path, v)

Note: we only know legal moves at the *root* (we have a real chess.Board
there). At depth > 0 we're in latent space — the model doesn't know
chess rules. So priors at depth>0 are NOT masked. The network has to
learn to put low probability on illegal moves. (This is a known MuZero
quirk for boardgames; the AlphaZero paper assumes oracle legality.)

NOT YET IMPLEMENTED.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    state: object = None  # latent tensor; None until expanded
    children: dict = field(default_factory=dict)
    is_expanded: bool = False


def run_mcts(root_obs, model, num_sims: int, cfg) -> dict:
    """Returns visit counts at root over all 4672 actions."""
    raise NotImplementedError
