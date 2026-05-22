"""Spike package — proves KataGo data generation can produce data in the
same format as our chess pipeline. Not a finished product.

Public API:
    play_one_game(katago_path, model_path, ...) -> dict
    katago_analysis_to_soft_targets(analysis_json, multipv=8, temperature_pawns=...) -> ndarray
    board_to_planes(state) -> ndarray
"""
from .board import BOARD_SIZE, N_INPUT_PLANES, board_to_planes, GoBoard
from .katago_data import (
    DEFAULT_MULTIPV,
    KataGoAnalysisEngine,
    katago_analysis_to_soft_targets,
    play_one_game,
)

__all__ = [
    "BOARD_SIZE",
    "N_INPUT_PLANES",
    "board_to_planes",
    "GoBoard",
    "DEFAULT_MULTIPV",
    "KataGoAnalysisEngine",
    "katago_analysis_to_soft_targets",
    "play_one_game",
]
