"""Hyperparameters for the Go pipeline.

Single source of truth — passed to network, MCTS, train loop. Defaults
target the 9x9 demo; switch board_size + adjust filters/blocks for 19x19.

KataGo efficiency flags (off by default for backward compat with the
distilled prior that ships with this repo):

  use_global_pool      — KataGo's global-pooling residual blocks. Mixes
                         channel-pooled features (mean + max) back into
                         the trunk so convolutions can condition on
                         global board context. ~1.60× speedup per
                         the KataGo paper ablations.
  use_aux_ownership    — KataGo's auxiliary ownership head. Predicts a
                         per-point softmax over {black, white, empty}
                         + a scalar final score margin. ~1.65× speedup.
                         Densest learning signal in the bag of tricks —
                         every point on the board becomes a label.
  use_aux_opp_policy   — KataGo's auxiliary opponent-policy head.
                         Predicts the move the opponent will play next.
                         ~1.30× speedup; trivially backward-compatible.
  use_game_features    — KataGo's atari/ladder/ko input planes (stubbed
                         out; full implementation requires re-running
                         distillation since input channel count
                         changes). Off-by-default placeholder.

When loading a pre-existing checkpoint (e.g. the distilled prior), use
`net.load_state_dict(..., strict=False)` — new aux heads / pooled
layers will start randomly and train from scratch through self-play
while the trunk + main policy/value heads inherit the prior.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoConfig:
    # Board / move encoding
    board_size: int = 9
    n_input_planes: int = 4   # 4 (no history) or 17 (8-step history × 2 + STM)
    komi: float = 7.5
    rules: str = "tromp-taylor"

    # Network
    n_res_blocks: int = 4     # 9x9 demo size; 19x19 paper-sized ~ 20
    n_filters: int = 64

    # KataGo efficiency tricks — all default off to preserve compat
    # with the existing distilled prior. New runs can opt in.
    use_global_pool: bool = False
    use_aux_ownership: bool = False
    use_aux_opp_policy: bool = False
    use_game_features: bool = False   # placeholder — input-plane stub only

    # MCTS
    sims_train: int = 100
    sims_eval: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.03   # AlphaGo-Zero uses 0.03 for 19x19; ~0.15 for 9x9
    dirichlet_eps: float = 0.25

    # Self-play / eval
    temp_moves: int = 30
    max_plies: int = 400           # 9x9 games rarely exceed 100; pad for safety

    # Training
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Auxiliary loss weights (KataGo-style; only applied when the
    # corresponding flags are on)
    aux_ownership_weight: float = 1.5    # KataGo paper Sec 5.2 default
    aux_score_weight: float = 0.02       # paper default; small because
                                          # score scales ~10× larger than ±1
    aux_opp_policy_weight: float = 0.15  # paper default

    @property
    def policy_size(self) -> int:
        """board_size² + 1 (pass)."""
        return self.board_size * self.board_size + 1

    @property
    def pass_move(self) -> int:
        return self.board_size * self.board_size
