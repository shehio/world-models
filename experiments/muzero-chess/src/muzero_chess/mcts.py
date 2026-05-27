"""MCTS over the learned dynamics function.

The defining difference from AlphaZero's MCTS:
  - Root expansion calls h_θ(observation) once.
  - All non-root expansions call g_θ(latent, action) — never board.push().
  - The tree only knows latent states, not chess positions.

This means we lose the rules engine's ability to enforce legality,
detect checkmate, etc. inside the search. MuZero's full implementation
recovers some of this by attaching a "legal mask" to the root only
(since we have the real board at the root). Below the root, all moves
are treated as if legal — the dynamics network has to learn the
"this isn't really a legal move" behavior by predicting bad rewards
for illegal-move expansions.

PUCT score (MuZero paper Appendix B):
    score(s, a) = Q(s, a)
                + P(s, a) · sqrt(ΣN(s, b)) / (1 + N(s, a))
                  · (log((ΣN + c_base + 1) / c_base) + c_init)

Value backup uses a discount (1.0 for chess) and the predicted reward
at each step:
    G_t = sum_{k=0}^{∞} γ^k · r_{t+k}
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import MuZeroConfig


@dataclass
class Node:
    """One node in the MCTS tree.

    Each node holds:
      - latent: the hidden state at this position (None at root before expansion)
      - prior: P(s, a) from the parent's policy head — what f_θ recommended
      - reward: r predicted by g_θ for the action that LED HERE (0 at root)
      - children: dict[action_idx → Node]
      - visit_count: N(s) — REAL backups (updated by _backup only)
      - value_sum:  ΣQ from REAL backups
      - virtual_count + virtual_loss: pending visits inside a batched search,
        applied during descent and reverted at backup. The selection function
        (_ucb_score) reads `effective_*` so that the next descent in the same
        batch sees this path as already-traversed and picks a different child.
    """
    prior: float = 0.0
    reward: float = 0.0
    latent: torch.Tensor | None = None
    children: dict[int, "Node"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_count: int = 0
    virtual_loss: float = 0.0

    @property
    def value(self) -> float:
        """Mean value over REAL backups (does not include virtual loss)."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    @property
    def effective_visit_count(self) -> int:
        return self.visit_count + self.virtual_count

    @property
    def effective_value(self) -> float:
        """Mean value INCLUDING virtual loss — used by selection only."""
        n = self.effective_visit_count
        if n <= 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / n

    @property
    def is_expanded(self) -> bool:
        return self.latent is not None


class MinMaxStats:
    """Per-search normalization for Q values.

    MuZero observed that absolute Q values can drift across rollouts;
    normalizing each child's Q into [0, 1] using min/max seen so far
    stabilizes selection. Paper Appendix B.
    """

    def __init__(self):
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, value: float) -> None:
        self._min = min(self._min, value)
        self._max = max(self._max, value)

    def normalize(self, value: float) -> float:
        if self._max > self._min:
            return (value - self._min) / (self._max - self._min)
        return value


def _ucb_score(parent: Node, child: Node, cfg: MuZeroConfig, mm: MinMaxStats) -> float:
    """MuZero's PUCT-with-min-max-normalization.

    Uses effective_* counts/values so that within a batched search, pending
    descents bias subsequent selections away from already-traversed paths.
    In sequential mode (virtual_count = 0 everywhere) this matches the
    paper formulas exactly.
    """
    p_n = parent.effective_visit_count
    c_n = child.effective_visit_count
    pb_c = (math.log((p_n + cfg.c_puct_base + 1) / cfg.c_puct_base)
            + cfg.c_puct_init)
    pb_c *= math.sqrt(p_n) / (1 + c_n)
    prior_score = pb_c * child.prior
    if c_n > 0:
        # Q from child's reward + discount * (effective) child value.
        # The reward is paid TO the parent when transitioning to the child;
        # since selection flips POV at each ply, parent sees +reward + γ * value.
        value_score = mm.normalize(child.reward + cfg.discount * child.effective_value)
    else:
        value_score = 0.0
    return prior_score + value_score


def _select_child(parent: Node, cfg: MuZeroConfig, mm: MinMaxStats) -> tuple[int, Node]:
    """Pick the action with highest PUCT among parent's children."""
    best_score = -float("inf")
    best: tuple[int, Node] | None = None
    for a, child in parent.children.items():
        s = _ucb_score(parent, child, cfg, mm)
        if s > best_score:
            best_score = s
            best = (a, child)
    assert best is not None
    return best


def _expand_with_policy(node: Node, latent: torch.Tensor, policy_logits: torch.Tensor,
                        legal_actions: list[int] | None = None,
                        top_k: int | None = None) -> None:
    """Attach latent + create child stubs with prior probabilities.

    Three modes:
      legal_actions != None (root)          → mask to those actions, keep all
      legal_actions == None, top_k != None  → keep top_k highest-prior actions
      legal_actions == None, top_k == None  → keep all 4672 (paper-faithful but slow)

    Top-K below the root is a wall-clock optimization: the full 4672-child
    fan-out makes _select_child a 4672-iteration Python loop and is the
    Python-bound bottleneck of the search. Keeping the top 32-ish actions
    by prior loses negligible search quality (most actions get vanishingly
    small prior anyway).
    """
    node.latent = latent
    if legal_actions is not None:
        # Root: mask + softmax over legal moves only.
        masked = torch.full_like(policy_logits, fill_value=-1e9)
        for a in legal_actions:
            masked[a] = policy_logits[a]
        probs = torch.softmax(masked, dim=-1).cpu().numpy()
        for a in legal_actions:
            node.children[a] = Node(prior=float(probs[a]))
        return

    # Non-root expansion.
    probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
    if top_k is not None and 0 < top_k < probs.shape[0]:
        # np.argpartition gives the indices of the top-K largest (unordered).
        idx = np.argpartition(-probs, top_k - 1)[:top_k]
        top_probs = probs[idx]
        # Renormalize so the kept priors sum to 1 — selection assumes this.
        s = top_probs.sum()
        if s > 0:
            top_probs = top_probs / s
        for a, p in zip(idx, top_probs):
            node.children[int(a)] = Node(prior=float(p))
    else:
        for a, p in enumerate(probs):
            node.children[int(a)] = Node(prior=float(p))


def _add_dirichlet_noise(node: Node, cfg: MuZeroConfig) -> None:
    if not node.children:
        return
    noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(node.children))
    eps = cfg.dirichlet_eps
    for (a, child), n in zip(node.children.items(), noise):
        child.prior = (1 - eps) * child.prior + eps * float(n)


def _backup(path: list[Node], leaf_value: float, cfg: MuZeroConfig,
            mm: MinMaxStats) -> None:
    """Walk path leaf → root, accumulating value with reward + discount.

    For chess (two-player), the value alternates sign at each ply because
    the player to move alternates.
    """
    value = leaf_value
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        mm.update(node.reward + cfg.discount * node.value)
        # Two-player negamax: flip sign + add reward at this node (the reward
        # was the reward of GETTING to this node, paid by the prior mover).
        value = -(node.reward + cfg.discount * value)


def _apply_virtual_loss(path: list[Node], cfg: MuZeroConfig) -> None:
    """Bump virtual_count + virtual_loss on every node in `path`.

    Applied to a path BEFORE the leaf has been expanded, so subsequent
    descents in the same batch see this path as already-pending and pick
    a different child via _ucb_score's effective_* lookup.
    """
    for node in path:
        node.virtual_count += 1
        node.virtual_loss += cfg.virtual_loss


def _revert_virtual_loss(path: list[Node], cfg: MuZeroConfig) -> None:
    """Reverse of _apply_virtual_loss — call ONCE per path just before backup."""
    for node in path:
        node.virtual_count -= 1
        node.virtual_loss -= cfg.virtual_loss


def run_mcts(
    network,
    obs: torch.Tensor,
    cfg: MuZeroConfig,
    *,
    add_root_noise: bool = True,
    legal_actions: list[int] | None = None,
    device: torch.device | None = None,
) -> Node:
    """Run cfg.num_simulations of PUCT-MCTS over the learned dynamics.

    Returns the root node. Caller derives the move via visit counts:
      action = argmax over root.children[a].visit_count

    `obs` should be (1, 19, 8, 8) — a single-position batch.
    """
    device = device or next(network.parameters()).device
    network.eval()
    mm = MinMaxStats()

    # Root: encode + expand. Reward at the root is 0 (no action led here).
    with torch.no_grad():
        s0, p0, v0 = network.initial_inference(obs.to(device))
    root = Node(prior=0.0, reward=0.0)
    _expand_with_policy(root, s0[0], p0[0], legal_actions=legal_actions)
    if add_root_noise:
        _add_dirichlet_noise(root, cfg)
    root.visit_count = 1
    root.value_sum = float(v0.item())

    for _ in range(cfg.num_simulations):
        node = root
        path = [root]
        # Descend until we hit an unexpanded child.
        while node.is_expanded and node.children:
            action, next_node = _select_child(node, cfg, mm)
            path.append(next_node)
            if not next_node.is_expanded:
                # Expand this child by calling g_θ on the parent's latent + action.
                action_t = torch.tensor([action], device=device, dtype=torch.long)
                with torch.no_grad():
                    s_next, r, p, v = network.recurrent_inference(
                        node.latent.unsqueeze(0), action_t,
                    )
                next_node.reward = float(r.item())
                _expand_with_policy(next_node, s_next[0], p[0], top_k=cfg.mcts_top_k)
                leaf_value = float(v.item())
                _backup(path, leaf_value, cfg, mm)
                break
            node = next_node
        else:
            # node has no children — terminal at the search level. Treat value as 0.
            _backup(path, 0.0, cfg, mm)

    return root


def run_mcts_batched(
    network,
    obs: torch.Tensor,
    cfg: MuZeroConfig,
    *,
    add_root_noise: bool = True,
    legal_actions: list[int] | None = None,
    device: torch.device | None = None,
    batch_size: int | None = None,
) -> Node:
    """Batched-descent variant of run_mcts using virtual loss.

    Collects up to `batch_size` parallel descents before calling g_θ + f_θ
    once on the resulting (latent, action) batch. The virtual loss makes
    each descent's path look slightly less attractive to the next descent,
    so different parallel descents tend to expand different leaves.

    Wall-clock win is roughly batch_size× on GPU once Python descent
    overhead dominates the per-call CUDA launch — which it does on a
    small net like ours.

    batch_size defaults to cfg.mcts_batch_size; batch_size=1 falls back to
    the sequential path (no virtual loss).
    """
    if batch_size is None:
        batch_size = cfg.mcts_batch_size
    if batch_size <= 1:
        return run_mcts(
            network, obs, cfg,
            add_root_noise=add_root_noise,
            legal_actions=legal_actions,
            device=device,
        )

    device = device or next(network.parameters()).device
    network.eval()
    mm = MinMaxStats()

    with torch.no_grad():
        s0, p0, v0 = network.initial_inference(obs.to(device))
    root = Node(prior=0.0, reward=0.0)
    _expand_with_policy(root, s0[0], p0[0], legal_actions=legal_actions)
    if add_root_noise:
        _add_dirichlet_noise(root, cfg)
    root.visit_count = 1
    root.value_sum = float(v0.item())

    sims_remaining = cfg.num_simulations
    while sims_remaining > 0:
        B = min(batch_size, sims_remaining)

        # Phase 1: collect B descent paths. Virtual loss diversifies.
        pending_paths: list[list[Node]] = []
        pending_parent_latents: list[torch.Tensor] = []
        pending_actions: list[int] = []

        for _ in range(B):
            node = root
            path = [root]
            terminal_descent = False
            while node.is_expanded and node.children:
                action, next_node = _select_child(node, cfg, mm)
                path.append(next_node)
                if not next_node.is_expanded:
                    # Found a leaf to expand later.
                    _apply_virtual_loss(path, cfg)
                    pending_paths.append(path)
                    pending_parent_latents.append(node.latent)
                    pending_actions.append(action)
                    terminal_descent = True
                    break
                node = next_node
            if not terminal_descent:
                # Empty-children case — treat as zero-value terminal and
                # backup immediately (no batched expansion needed).
                _backup(path, 0.0, cfg, mm)

        # Phase 2: one batched g+f call for all pending leaves.
        if pending_paths:
            latent_batch = torch.stack(pending_parent_latents, dim=0).to(device)
            action_batch = torch.tensor(pending_actions, device=device, dtype=torch.long)
            with torch.no_grad():
                s_next, r, p, v = network.recurrent_inference(latent_batch, action_batch)

            # Phase 3: expand + revert virtual loss + backup per leaf, in order.
            for i, path in enumerate(pending_paths):
                leaf = path[-1]
                leaf.reward = float(r[i].item())
                _expand_with_policy(leaf, s_next[i], p[i], top_k=cfg.mcts_top_k)
                _revert_virtual_loss(path, cfg)
                _backup(path, float(v[i].item()), cfg, mm)

        sims_remaining -= B

    return root


def select_action(root: Node, temperature: float = 1.0) -> int:
    """Sample / argmax over root visit counts."""
    actions = list(root.children.keys())
    counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
    if temperature <= 1e-6:
        max_c = counts.max()
        candidates = [a for a, c in zip(actions, counts) if c == max_c]
        return int(np.random.choice(candidates))
    counts = counts ** (1.0 / temperature)
    if counts.sum() == 0:
        return int(np.random.choice(actions))
    probs = counts / counts.sum()
    return int(np.random.choice(actions, p=probs))


def root_visit_distribution(root: Node, action_dim: int) -> np.ndarray:
    """Convert root visit counts to a (action_dim,) probability vector — the π target."""
    pi = np.zeros(action_dim, dtype=np.float32)
    total = sum(c.visit_count for c in root.children.values())
    if total == 0:
        return pi
    for a, child in root.children.items():
        pi[a] = child.visit_count / total
    return pi
