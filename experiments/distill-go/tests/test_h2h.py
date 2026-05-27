"""Smoke tests for the Go head-to-head script.

Heavy work (multi-process workers, real game loop) is tested via the
manual run on AWS — these are the building-block tests that run in <5s.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from distill_go.board import BLACK, WHITE, GoBoard
from distill_go.config import GoConfig
from distill_go.network import AlphaZeroGoNet
from h2h import play_one_game, score_to_elo, wilson_ci


def test_play_one_game_terminates_with_winner_or_none():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8,
                   max_plies=30)
    device = torch.device("cpu")
    net_a = AlphaZeroGoNet(cfg).to(device); net_a.eval()
    net_b = AlphaZeroGoNet(cfg).to(device); net_b.eval()
    res = play_one_game(net_a, net_b, cfg, device, sims=4, a_color=BLACK,
                         max_moves=20)
    assert res["winner"] in (BLACK, WHITE, None)
    assert res["draw"] == (res["winner"] is None)
    assert res["a_color"] == BLACK
    assert res["ply"] > 0


def test_play_one_game_a_won_flag_matches_color():
    cfg = GoConfig(board_size=5, n_input_planes=4, n_res_blocks=1, n_filters=8)
    device = torch.device("cpu")
    net_a = AlphaZeroGoNet(cfg).to(device); net_a.eval()
    net_b = AlphaZeroGoNet(cfg).to(device); net_b.eval()
    res = play_one_game(net_a, net_b, cfg, device, sims=4, a_color=WHITE,
                         max_moves=20)
    if res["draw"]:
        assert res["a_won"] is False
    elif res["winner"] == WHITE:
        assert res["a_won"] is True   # A was WHITE, winner is WHITE
    else:
        assert res["a_won"] is False  # A was WHITE, winner is BLACK


def test_score_to_elo_half_is_zero():
    assert score_to_elo(0.5) == pytest.approx(0.0, abs=1e-9)


def test_score_to_elo_extremes_clamp_to_inf():
    assert score_to_elo(0.0) == -math.inf
    assert score_to_elo(1.0) == math.inf


def test_score_to_elo_sign_convention():
    assert score_to_elo(0.6) > 0      # A winning means positive gap (A − B)
    assert score_to_elo(0.4) < 0


def test_wilson_ci_at_zero_games_is_full_range():
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_wilson_ci_at_full_sweep_is_finite_below_one():
    # 100 wins of 100: Wilson CI is not [0,1] — has a non-trivial lower bound.
    lo, hi = wilson_ci(100, 100)
    assert lo > 0.9
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_ci_at_half_centers_near_half():
    lo, hi = wilson_ci(50, 100)
    assert 0.4 < lo < 0.5
    assert 0.5 < hi < 0.6
