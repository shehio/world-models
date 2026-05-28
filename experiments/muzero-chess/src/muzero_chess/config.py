"""MuZero on chess — hyperparameters.

MuZero differs from AlphaZero in that the dynamics function is learned,
not the real environment. The hidden-state shape is the central
architectural decision: it has to be small enough to train but big enough
that the dynamics network can predict the next state of chess after a
move from it. We mirror AlphaZero-family practice: an 8x8 spatial map
with N channels, since chess positions are naturally 8x8.

Defaults target a small educational config — fast to train on a single
GPU, big enough to actually learn something on chess at scale, small
enough that the spike fits in a session.
"""
from __future__ import annotations

from dataclasses import dataclass


# Chess board encoding constants from wm_chess.board.
# Re-stated here so the config is self-contained; verified by a test.
INPUT_PLANES = 19          # 12 piece planes + STM + 4 castling + EP + halfmove
ACTION_DIM = 4672          # 8x8x73 — AlphaZero's move encoding
BOARD_SIZE = 8
N_MOVE_PLANES = 73         # AZ move encoding: action = plane * 64 + from_square


@dataclass(frozen=True)
class MuZeroConfig:
    # Hidden state shape: (latent_channels, BOARD_SIZE, BOARD_SIZE).
    # The dynamics network preserves this shape across recurrent calls.
    latent_channels: int = 64

    # Representation network (h_θ).
    repr_n_res_blocks: int = 5
    repr_n_filters: int = 64       # internal conv channels; output projects to latent_channels

    # Dynamics network (g_θ).
    # Inputs: (latent (B, C, 8, 8), action_index int → action plane (B, 1, 8, 8))
    # Outputs: (next_latent, reward)
    dyn_n_res_blocks: int = 3
    dyn_n_filters: int = 64

    # Prediction network (f_θ).
    pred_n_res_blocks: int = 2
    pred_n_filters: int = 64

    # MCTS over the learned dynamics.
    num_simulations: int = 50      # cheap for the educational spike; AZ defaults to 800
    c_puct_base: float = 19_652    # MuZero paper's pb_c_base
    c_puct_init: float = 1.25      # MuZero paper's pb_c_init
    dirichlet_alpha: float = 0.3   # chess-tuned alpha from AlphaZero paper
    dirichlet_eps: float = 0.25
    discount: float = 1.0          # board games have no temporal discount
    # Batched MCTS: K parallel descents per network call, diversified by virtual
    # loss. 1 = sequential reference path; the wall-clock win is on GPU where
    # batching amortizes Python overhead + the per-call CUDA launch.
    mcts_batch_size: int = 8
    virtual_loss: float = 1.0      # penalty added to Q for each pending visit
    # Top-K expansion below the root: only keep the K highest-prior children
    # at non-root nodes. The full 4672-action expansion is the paper algorithm
    # but makes _select_child a 4672-iteration Python loop and dominates the
    # wall clock. Most actions have vanishingly small prior anyway, so the
    # search-quality loss is negligible. Root expansion is unchanged
    # (legal-action mask). Use 0 / None for "no top-K, expand all".
    mcts_top_k: int = 32

    # K-step unroll for training.
    num_unroll_steps: int = 5
    n_step_return: int = 10        # n-step bootstrap for value targets

    # Training.
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Self-play.
    temp_moves: int = 30
    max_plies: int = 200

    # --- Distill-init mode (reuse a distilled AlphaZeroNet as frozen h + f,
    # train only the dynamics g). Ignored by the from-scratch loop. ---
    teacher_ckpt: str = ""          # local path to the distilled checkpoint
    teacher_n_blocks: int = 20      # AlphaZeroNet trunk depth of the teacher
    teacher_n_filters: int = 256    # = latent_channels when distill-init is on
    latent_loss_weight: float = 1.0   # MSE(g-rollout latent, h(real next obs))
    pred_loss_weight: float = 0.25    # consistency through frozen f (policy CE + value MSE)
    epsilon_random: float = 0.1     # fraction of random moves in transition data gen

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, BOARD_SIZE, BOARD_SIZE)

    @property
    def input_planes(self) -> int:
        return INPUT_PLANES

    @property
    def action_dim(self) -> int:
        return ACTION_DIM


def distill_init_config(**overrides) -> MuZeroConfig:
    """Config for the distill-init experiment: the latent is the teacher's
    trunk output, so latent_channels must equal the teacher's filter width.
    The dynamics net runs at that same width. Everything h/f-shaped is
    irrelevant here (those come from the frozen teacher, not from cfg)."""
    base = dict(
        latent_channels=256,
        dyn_n_filters=256,
        dyn_n_res_blocks=4,
        teacher_n_blocks=20,
        teacher_n_filters=256,
        num_unroll_steps=5,
        num_simulations=800,    # match the teacher's sims=800 eval anchor
        batch_size=256,
        lr=1e-3,
    )
    base.update(overrides)
    return MuZeroConfig(**base)
