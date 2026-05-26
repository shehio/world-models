"""Tests for the Elo-gap helper in scripts/eval.py.

Pin down the boundary behavior at score=0 / score=1 / score=0.5.
The function is short (4 lines) but it's the only thing converting raw
scores into Elo gaps in the eval pipeline — a sign error or missing
guard would silently corrupt every reported Elo.

We import the script module directly (eval.py is a CLI but the helper
is module-level) — no Stockfish/torch needed to test this function.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval as eval_script   # noqa: A001


def test_gap_at_half_is_zero():
    assert eval_script.gap(0.5) == pytest.approx(0.0, abs=1e-9)


def test_gap_at_score_zero_returns_minus_infinity():
    """Player lost every game — no finite Elo gap representable.

    Guard MUST trigger before log(0) blows up.
    """
    assert eval_script.gap(0.0) == -math.inf


def test_gap_at_score_one_returns_plus_infinity():
    """Player won every game — same guard, other side."""
    assert eval_script.gap(1.0) == math.inf


def test_gap_negative_score_returns_minus_infinity():
    """Defensive: caller passes nonsense, function doesn't crash."""
    assert eval_script.gap(-0.1) == -math.inf


def test_gap_above_one_returns_plus_infinity():
    """Defensive on the other side."""
    assert eval_script.gap(1.5) == math.inf


def test_gap_near_extremes_still_finite():
    """Scores very close to 0 / 1 must NOT hit the guard — they compute
    finite Elo gaps. Guards against an over-eager clamp regression."""
    assert math.isfinite(eval_script.gap(0.001))
    assert math.isfinite(eval_script.gap(0.999))
    assert eval_script.gap(0.001) < -1000   # ~ -1200 Elo
    assert eval_script.gap(0.999) > 1000    # ~ +1200 Elo


def test_gap_typical_distillation_value():
    """A reality-check: a score of 0.933 (the d15 baseline vs UCI=1,350)
    converts to ~+457 Elo gap — the actual published number on the site."""
    elo = eval_script.gap(0.933)
    assert 450 < elo < 465


def test_gap_sign_convention_higher_score_means_positive_gap():
    assert eval_script.gap(0.6) > 0
    assert eval_script.gap(0.4) < 0
    assert abs(eval_script.gap(0.4) + eval_script.gap(0.6)) < 1e-9   # symmetric around 0.5
