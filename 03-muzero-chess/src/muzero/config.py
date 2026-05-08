"""MuZero-chess hyperparameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Board / move encoding (same as 02)
    n_input_planes: int = 19
    policy_size: int = 4672

    # Latent state
    latent_channels: int = 64
    latent_size: int = 8  # hidden state stays 8x8 spatially for chess

    # Networks
    n_res_blocks_repr: int = 5
    n_res_blocks_dyn: int = 3
    n_res_blocks_pred: int = 3

    # MCTS (same as AlphaZero)
    sims_train: int = 50
    sims_eval: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # Self-play
    temp_moves: int = 30
    max_plies: int = 400

    # Training
    unroll_steps: int = 5  # K
    n_step_return: int = 10  # for value bootstrap target
    discount: float = 1.0  # chess: terminal-only reward, no discount
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_capacity: int = 1000  # games, not positions
    train_steps_per_iter: int = 200

    # Loop
    games_per_iter: int = 20
    iters: int = 50
    checkpoint_every: int = 5
