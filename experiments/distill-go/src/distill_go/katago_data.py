"""KataGo analysis-engine wrapper + game-playing dataset generator.

Lifted from `experiments/distill-go-spike/src/distill_go_spike/katago_data.py`
and adapted to:
  - Use the new rules-aware `GoBoard` (no longer just bookkeeping).
  - Support both 4-plane and 17-plane state encoding via `n_input_planes`.
  - Be importable from the parallel orchestrator + tests.

Schema written per game (identical to chess pipeline):
    states           (n, n_input_planes, S, S)  float32
    moves            (n,)                       int64    flat idx, pass = S²
    multipv_indices  (n, K)                     int64
    multipv_logprobs (n, K)                     float32
    zs               (n,)                       float32  ±1 from STM's POV

KataGo's analysis-engine protocol:
  https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md
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
    WHITE,
    GoBoard,
    board_to_history_planes,
    board_to_planes,
    flat_to_gtp,
    gtp_to_flat,
)


DEFAULT_MULTIPV = 8
DEFAULT_VISITS = 400
DEFAULT_TEMPERATURE = 1.0


@dataclass
class AnalysisMove:
    gtp: str
    flat_idx: int
    visits: int
    winrate: float
    score_lead: float


class KataGoAnalysisEngine:
    """Subprocess wrapper around `katago analysis`.

    Usage:
        with KataGoAnalysisEngine(BIN, MODEL, CONFIG) as eng:
            resp = eng.query(moves=[("B","D4")], board_size=9, visits=400, top_k=8)
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
        board_size: int = 9,
        visits: int = DEFAULT_VISITS,
        top_k: int = DEFAULT_MULTIPV,
        komi: float = 7.5,
        rules: str = "tromp-taylor",
    ) -> dict:
        if self._proc is None:
            raise RuntimeError("engine not started — use as context manager")
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
        }
        self._proc.stdin.write(json.dumps(request) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]
        while True:
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise RuntimeError("katago analysis stdout closed unexpectedly")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == req_id:
                return msg

    def best_move(
        self,
        moves: list[tuple[str, str]],
        board_size: int,
        visits: int,
        komi: float = 7.5,
        rules: str = "tromp-taylor",
    ) -> str:
        """Return KataGo's top-visited move as a GTP coord (or 'pass')."""
        resp = self.query(moves, board_size=board_size, visits=visits,
                          top_k=1, komi=komi, rules=rules)
        move_infos = resp.get("moveInfos", [])
        if not move_infos:
            return "pass"
        top = max(move_infos, key=lambda m: m["visits"])
        return top["move"]


def katago_analysis_to_soft_targets(
    response: dict,
    board_size: int,
    top_k: int = DEFAULT_MULTIPV,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert KataGo moveInfos → (indices, logprobs, value).

    indices  (top_k,) int64  — flat board idx, -1 padding
    logprobs (top_k,) float32 — softmax-over-visits, -inf padding
    value    float — winrate of side-to-move at this position
    """
    move_infos = response.get("moveInfos", [])
    if not move_infos:
        indices = np.full(top_k, -1, dtype=np.int64)
        indices[0] = board_size * board_size  # pass
        logprobs = np.full(top_k, -np.inf, dtype=np.float32)
        logprobs[0] = 0.0
        return indices, logprobs, 0.5

    sorted_moves = sorted(move_infos, key=lambda m: m["visits"], reverse=True)
    top = sorted_moves[:top_k]

    indices = np.full(top_k, -1, dtype=np.int64)
    visits = np.full(top_k, 0.0, dtype=np.float64)
    for i, mv in enumerate(top):
        indices[i] = gtp_to_flat(mv["move"], board_size)
        visits[i] = float(mv["visits"])

    mask = indices != -1
    logits = visits[mask] / max(temperature, 1e-6)
    logits = logits - logits.max()
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
    n_input_planes: int = 4,
    max_moves: int = 200,
    visits: int = DEFAULT_VISITS,
    top_k: int = DEFAULT_MULTIPV,
    temperature: float = DEFAULT_TEMPERATURE,
    komi: float = 7.5,
    rules: str = "tromp-taylor",
) -> dict:
    """KataGo-vs-KataGo self-play. Returns the standard 5-key dict.

    `n_input_planes` ∈ {4, 17}: which encoder to use. 17 stacks the last 8
    board states (AlphaGo-Zero convention).
    """
    board = GoBoard(size=board_size, komi=komi)
    history_gtp: list[tuple[str, str]] = []
    history_boards: list[GoBoard] = [board.copy()]

    states: list[np.ndarray] = []
    moves: list[int] = []
    mpv_idx: list[np.ndarray] = []
    mpv_logp: list[np.ndarray] = []
    values: list[float] = []
    stm_at_step: list[int] = []

    color_to_gtp = {BLACK: "B", WHITE: "W"}

    def _encode(b: GoBoard) -> np.ndarray:
        if n_input_planes == 17:
            return board_to_history_planes(history_boards + [b], n_history=8)
        return board_to_planes(b)

    t0 = time.time()
    with KataGoAnalysisEngine(katago_binary, model_path, config_path) as eng:
        for _ in range(max_moves):
            resp = eng.query(
                moves=history_gtp, board_size=board_size, visits=visits,
                top_k=top_k, komi=komi, rules=rules,
            )
            indices, logprobs, value = katago_analysis_to_soft_targets(
                resp, board_size=board_size, top_k=top_k, temperature=temperature,
            )
            states.append(_encode(board))
            mpv_idx.append(indices)
            mpv_logp.append(logprobs)
            values.append(value)
            stm_at_step.append(board.to_move)

            chosen_local = int(np.argmax(logprobs))
            chosen_flat = int(indices[chosen_local])
            moves.append(chosen_flat)

            played_color = color_to_gtp[board.to_move]
            gtp = flat_to_gtp(chosen_flat, board_size)
            history_gtp.append((played_color, gtp))
            try:
                board.play(chosen_flat)
            except Exception:
                # KataGo proposed an illegal move (rare) — pass instead so
                # the game still terminates cleanly.
                board.play(board.size * board.size)
                history_gtp[-1] = (played_color, "pass")

            history_boards.append(board.copy())
            if len(history_boards) > 9:
                history_boards = history_boards[-9:]
            if board.is_game_over:
                break

    # Use the local rules engine's score, not KataGo's value head, so the
    # z target is grounded in actual game outcome.
    winner = board.winner()
    zs = np.array(
        [1.0 if stm == winner else -1.0 for stm in stm_at_step],
        dtype=np.float32,
    )

    n_planes = states[0].shape[0] if states else n_input_planes
    return {
        "states": (np.stack(states) if states
                   else np.zeros((0, n_planes, board_size, board_size), dtype=np.float32)),
        "moves": np.asarray(moves, dtype=np.int64),
        "multipv_indices": (np.stack(mpv_idx) if mpv_idx
                            else np.zeros((0, top_k), dtype=np.int64)),
        "multipv_logprobs": (np.stack(mpv_logp) if mpv_logp
                             else np.zeros((0, top_k), dtype=np.float32)),
        "zs": zs,
        "_meta": {
            "board_size": board_size,
            "n_input_planes": n_input_planes,
            "visits": visits,
            "top_k": top_k,
            "komi": komi,
            "rules": rules,
            "n_moves": len(moves),
            "elapsed_s": time.time() - t0,
            "winner": "B" if winner == BLACK else "W",
        },
    }
