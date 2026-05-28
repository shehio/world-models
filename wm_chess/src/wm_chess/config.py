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

    # Arena gating (AlphaGo Zero evaluator). gate_every=0 disables gating
    # (workers self-play from the latest net; no champion, no promotion).
    gate_every: int = 0
    gate_games: int = 60
    gate_sims: int = 200
    gate_threshold: float = 0.55

    # KL-anchor to the distilled teacher: trust region for warm-started RL.
    # kl_beta=0 disables it (loss = policy CE + value MSE only).
    kl_beta: float = 0.0

    # In-loop Stockfish-panel yardstick (run on each promotion when gating).
    sf_eval_games: int = 20
    sf_eval_sims: int = 400
    sf_eval_depth: int = 8
