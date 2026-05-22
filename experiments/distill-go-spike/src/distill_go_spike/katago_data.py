"""KataGo analysis-engine integration for the distillation spike.

Mirrors the structure of `distill_soft.stockfish_data` so the training
loop in `experiments/distill-soft` can be reused with minimal changes —
the on-disk .npz schema is identical:

    states:           (n, N_INPUT_PLANES, size, size) float32
    moves:            (n,)                            int64  (flat board idx + pass)
    multipv_indices:  (n, K)                          int64
    multipv_logprobs: (n, K)                          float32
    zs:               (n,)                            float32  (game outcome from current STM POV)

KataGo's analysis engine takes JSON requests on stdin and emits JSON
responses on stdout — schema documented at
https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md.

For each board position we ask KataGo to evaluate it with N visits and
return the top-K candidate moves with their visit counts, winrates, and
score leads. We softmax the visit counts to produce a policy distribution
(directly equivalent to Stockfish's softmax-of-centipawn-scores in the
chess pipeline), which becomes the soft target the student learns to
match.

Spike scope: a single self-play game on a small board (9×9 by default).
Production scope (out of this file): self-play parallelism, crash-safe
chunking, KataGo weights pinning, GPU allocation. All of that can lift
straight from the chess infra once this proves the data shape lines up.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from .board import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    GoBoard,
    N_INPUT_PLANES,
    WHITE,
    board_to_planes,
    gtp_move_to_xy,
    xy_to_gtp_move,
)


DEFAULT_MULTIPV = 8
DEFAULT_VISITS = 400
DEFAULT_TEMPERATURE = 1.0  # softmax temperature on visit counts


@dataclass
class AnalysisMove:
    """One candidate move from a KataGo analysis response."""

    gtp: str            # GTP coordinate, e.g. "D4" or "pass"
    flat_idx: int       # flat board index = y * size + x; pass = size * size
    visits: int         # MCTS visit count (used as the soft target logit pre-softmax)
    winrate: float      # P(this side wins) ∈ [0, 1]
    score_lead: float   # expected score margin from this side's POV


class KataGoAnalysisEngine:
    """Thin subprocess wrapper around `katago analysis`.

    Usage:
        with KataGoAnalysisEngine("/usr/local/bin/katago", "/opt/katago/model.bin.gz") as eng:
            resp = eng.query(moves=[("B","D4"),("W","Q16")], board_size=19, visits=400, top_k=8)

    KataGo's stdout interleaves analysis results with engine log lines
    (`PV` summaries, etc.), so we filter to JSON lines only. Each request
    gets a fresh `uuid4` id so we don't mismatch responses if the engine
    queues queries.
    """

    def __init__(
        self,
        katago_binary: str,
        model_path: str,
        config_path: str | None = None,
    ) -> None:
        self.katago_binary = katago_binary
        self.model_path = model_path
        self.config_path = config_path
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "KataGoAnalysisEngine":
        cmd = [self.katago_binary, "analysis", "-model", self.model_path]
        if self.config_path:
            cmd += ["-config", self.config_path]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    def query(
        self,
        moves: list[tuple[str, str]],
        board_size: int = BOARD_SIZE,
        visits: int = DEFAULT_VISITS,
        top_k: int = DEFAULT_MULTIPV,
        komi: float = 7.5,
        rules: str = "tromp-taylor",
    ) -> dict:
        """Send one analysis request and return the parsed response dict.

        `moves` is the prefix history of the game in GTP form, e.g.
        [("B", "D4"), ("W", "Q16"), ...]. KataGo will analyze the
        position after the last move and return the top moves for the
        next-to-play side.
        """
        if self._proc is None:
            raise RuntimeError("engine not started (use as context manager)")
        req_id = uuid.uuid4().hex
        request = {
            "id": req_id,
            "moves": moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "maxVisits": visits,
            "analyzeTurns": [len(moves)],
            # `includePolicy: True` returns the raw network policy too,
            # but for the soft-multipv target we want the MCTS-improved
            # visit distribution. That's what `moveInfos` carries.
        }
        self._proc.stdin.write(json.dumps(request) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]
        # Read lines until we see one matching this request's id.
        while True:
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise RuntimeError("katago analysis stdout closed unexpectedly")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # log line, skip
            if msg.get("id") == req_id:
                return msg


def _gtp_to_flat_idx(gtp: str, board_size: int) -> int:
    """Encode a GTP move as a flat int64 index. Pass = board_size**2."""
    if gtp.lower() == "pass":
        return board_size * board_size
    x, y = gtp_move_to_xy(gtp, board_size)
    return y * board_size + x


def katago_analysis_to_soft_targets(
    response: dict,
    board_size: int = BOARD_SIZE,
    top_k: int = DEFAULT_MULTIPV,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a KataGo `moveInfos` array into soft policy targets.

    Returns
    -------
    multipv_indices : (top_k,) int64
        Flat board indices of the top-K candidate moves. Padded with -1
        if KataGo returned fewer than top_k.
    multipv_logprobs : (top_k,) float32
        Log-softmax of `visits` over the top-K. Pads with -inf.
    value : float
        winrate of the side to move (used directly as the value target z).

    The softmax-over-visits is the Go analog of our chess pipeline's
    softmax-over-centipawns: it produces a distribution that is sharper
    when one move dominates and softer when several moves are roughly
    equal. `temperature` scales the logits (visits) the same way `T_PAWNS`
    does in chess.
    """
    move_infos = response.get("moveInfos", [])
    if not move_infos:
        # No legal moves analyzed — return all-pass policy and value 0.5
        indices = np.full(top_k, -1, dtype=np.int64)
        indices[0] = board_size * board_size
        logprobs = np.full(top_k, -np.inf, dtype=np.float32)
        logprobs[0] = 0.0
        return indices, logprobs, 0.5

    sorted_moves = sorted(move_infos, key=lambda m: m["visits"], reverse=True)
    top = sorted_moves[:top_k]

    indices = np.full(top_k, -1, dtype=np.int64)
    visits = np.full(top_k, 0.0, dtype=np.float64)
    for i, mv in enumerate(top):
        indices[i] = _gtp_to_flat_idx(mv["move"], board_size)
        visits[i] = float(mv["visits"])

    mask = indices != -1
    logits = visits[mask] / max(temperature, 1e-6)
    logits = logits - logits.max()  # for numerical stability
    log_z = np.log(np.exp(logits).sum())
    logprobs_pop = (logits - log_z).astype(np.float32)
    logprobs = np.full(top_k, -np.inf, dtype=np.float32)
    logprobs[mask] = logprobs_pop

    root_info = response.get("rootInfo", {})
    value = float(root_info.get("winrate", 0.5))
    return indices, logprobs, value


def play_one_game(
    katago_binary: str,
    model_path: str,
    *,
    config_path: str | None = None,
    board_size: int = 9,
    max_moves: int = 200,
    visits: int = DEFAULT_VISITS,
    top_k: int = DEFAULT_MULTIPV,
    temperature: float = DEFAULT_TEMPERATURE,
    komi: float = 7.5,
    rules: str = "tromp-taylor",
) -> dict:
    """Self-play one game between two copies of KataGo, returning training tensors.

    For each move:
      1. Send the full move history to the analysis engine.
      2. Capture (state planes, top-K move indices, softmax logprobs, value).
      3. Sample the move actually played from the visit distribution
         (greedy for now — same as chess pipeline's deterministic move).
      4. Apply the move locally + extend the GTP history.

    The game ends at `max_moves` or when both sides pass consecutively.

    Output dict has keys matching the chess pipeline's npz schema:
      states (n, N_INPUT_PLANES, size, size), moves (n,),
      multipv_indices (n, K), multipv_logprobs (n, K), zs (n,)
    """
    board = GoBoard(size=board_size)
    history: list[tuple[str, str]] = []

    states: list[np.ndarray] = []
    moves: list[int] = []
    mpv_idx: list[np.ndarray] = []
    mpv_logp: list[np.ndarray] = []
    values: list[float] = []  # KataGo's value for STM at each position

    passes_in_a_row = 0
    color_to_gtp = {BLACK: "B", WHITE: "W"}

    t0 = time.time()
    with KataGoAnalysisEngine(katago_binary, model_path, config_path) as eng:
        for _ in range(max_moves):
            resp = eng.query(
                moves=history,
                board_size=board_size,
                visits=visits,
                top_k=top_k,
                komi=komi,
                rules=rules,
            )
            indices, logprobs, value = katago_analysis_to_soft_targets(
                resp, board_size=board_size, top_k=top_k, temperature=temperature
            )
            states.append(board_to_planes(board))
            mpv_idx.append(indices)
            mpv_logp.append(logprobs)
            values.append(value)

            # Greedy move: argmax over the soft policy.
            chosen_local = int(np.argmax(logprobs))
            chosen_flat = int(indices[chosen_local])
            moves.append(chosen_flat)

            if chosen_flat == board_size * board_size:  # pass
                board.play(-1, -1)
                history.append((color_to_gtp[BLACK if board.to_move == WHITE else WHITE], "pass"))
                passes_in_a_row += 1
                if passes_in_a_row >= 2:
                    break
            else:
                y, x = divmod(chosen_flat, board_size)
                gtp = xy_to_gtp_move(x, y, board_size)
                played_color = color_to_gtp[board.to_move]
                board.play(x, y)
                history.append((played_color, gtp))
                passes_in_a_row = 0

    # KataGo's value is already from-STM perspective; convert to ±1 z per
    # position using a final game-result query (simplest spike: derive z
    # from the value at the final position).
    final_value = values[-1] if values else 0.5
    final_z = 1.0 if final_value > 0.5 else (-1.0 if final_value < 0.5 else 0.0)
    zs = []
    for i, _ in enumerate(states):
        # Alternate sign so each row carries z from THAT row's STM POV.
        sign = 1.0 if (i % 2 == 0) else -1.0
        zs.append(final_z * sign)

    return {
        "states": np.stack(states) if states else np.zeros((0, N_INPUT_PLANES, board_size, board_size), dtype=np.float32),
        "moves": np.asarray(moves, dtype=np.int64),
        "multipv_indices": np.stack(mpv_idx) if mpv_idx else np.zeros((0, top_k), dtype=np.int64),
        "multipv_logprobs": np.stack(mpv_logp) if mpv_logp else np.zeros((0, top_k), dtype=np.float32),
        "zs": np.asarray(zs, dtype=np.float32),
        "_meta": {
            "board_size": board_size,
            "visits": visits,
            "top_k": top_k,
            "temperature": temperature,
            "komi": komi,
            "rules": rules,
            "n_moves": len(moves),
            "elapsed_s": time.time() - t0,
            "final_winrate_stm": final_value,
        },
    }
