"""Local CPU smoke for distill-init MuZero — proves the plumbing without the
real teacher, a GPU, or any cloud spend.

Uses a tiny RANDOM AlphaZeroNet as a stand-in teacher: factor it into h+f,
generate teacher self-play transitions, train g for a few hundred steps, and
confirm the composed (h, g, f) net runs through MCTS and plays legal moves.

    uv run --project experiments/muzero-chess python scripts/smoke_distill_init.py
"""
from __future__ import annotations

import random

import chess
import torch

from muzero_chess.config import MuZeroConfig
from muzero_chess.distill_dynamics import (
    TransitionBuffer,
    distill_step,
    generate_transitions,
)
from muzero_chess.eval import muzero_policy
from muzero_chess.networks import DynamicsNet
from muzero_chess.teacher import (
    DistillInitMuZeroNet,
    TeacherPrediction,
    TeacherRepresentation,
)

from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet


def main() -> None:
    torch.manual_seed(0)
    # Stand-in teacher: tiny + random (real run loads the 20x256 distilled net).
    teacher = AlphaZeroNet(Config(n_res_blocks=2, n_filters=32)).eval()
    cfg = MuZeroConfig(
        latent_channels=32, dyn_n_filters=32, dyn_n_res_blocks=2,
        num_simulations=16, mcts_batch_size=4, mcts_top_k=16,
        num_unroll_steps=4, max_plies=40, temp_moves=10, epsilon_random=0.15,
        batch_size=16, lr=2e-3,
    )

    # Factoring sanity: f(h(obs)) must equal teacher(obs).
    h = TeacherRepresentation(teacher)
    f = TeacherPrediction(teacher)
    obs = torch.randn(2, 19, 8, 8)
    with torch.no_grad():
        tp, tv = teacher(obs)
        fp, fv = f(h(obs))
    assert torch.allclose(tp, fp, atol=1e-5) and torch.allclose(tv, fv, atol=1e-5)
    print("[smoke] factoring exact: f(h(obs)) == teacher(obs) ✓", flush=True)

    composed = DistillInitMuZeroNet(h, DynamicsNet(cfg), f)
    opt = torch.optim.Adam(composed.g.parameters(), lr=cfg.lr)

    print("[smoke] generating teacher self-play transitions...", flush=True)
    games = generate_transitions(teacher, cfg, n_games=12, seed=0)
    buf = TransitionBuffer(K=cfg.num_unroll_steps)
    buf.add(games)
    print(f"[smoke] buffer: {len(buf)} games, {buf.n_windows} K-step windows", flush=True)

    rng = random.Random(0)
    first = None
    for step in range(200):
        batch = buf.sample(cfg.batch_size, rng=rng)
        stats = distill_step(composed, opt, batch, cfg)
        if first is None:
            first = stats
        if step % 40 == 0:
            print(f"[smoke] step {step:3d}  total={stats['loss']:.4f}  "
                  f"latent={stats['latent_loss']:.4f}  pred={stats['pred_loss']:.4f}  "
                  f"reward={stats['reward_loss']:.4f}", flush=True)
    print(f"[smoke] latent loss {first['latent_loss']:.4f} -> {stats['latent_loss']:.4f}",
          flush=True)
    assert stats["latent_loss"] < first["latent_loss"], "g did not learn the dynamics"

    # Composed net must drive MCTS and produce legal moves.
    composed.eval()
    pol = muzero_policy(composed, cfg)
    board = chess.Board()
    for _ in range(6):
        move = pol(board)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break
    print("[smoke] composed net played legal moves through MCTS ✓", flush=True)
    print("[smoke] PASS — plumbing works end to end.", flush=True)


if __name__ == "__main__":
    main()
