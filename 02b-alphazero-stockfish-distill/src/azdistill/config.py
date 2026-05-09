"""Single source of truth for AlphaZero-chess hyperparameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Board / move encoding
    n_input_planes: int = 19  # 6 piece types × 2 colors + 7 meta planes
    n_move_planes: int = 73
    policy_size: int = 8 * 8 * 73  # 4672

    # Network
    n_res_blocks: int = 5
    n_filters: int = 64

    # MCTS
    sims_train: int = 50
    sims_eval: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # Self-play
    temp_moves: int = 30  # plies with τ=1; afterwards τ→0
    max_plies: int = 400  # safety cap (chess can technically hit 50-move/draw rules first)

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_capacity: int = 100_000
    train_steps_per_iter: int = 200

    # Loop
    games_per_iter: int = 20
    iters: int = 50
    checkpoint_every: int = 5
