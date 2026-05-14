"""Unit tests for the binary-search Elo bracketer.

Covers the pure functions (implied_elo_gap / update_bracket / should_stop)
and a faked end-to-end loop that swaps run_match out for a closed-form
"Stockfish at strength X" model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package — add it to sys.path the same way the
# train-supervised tests add src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import elo_bisect


# ---------- implied_elo_gap --------------------------------------------------

def test_implied_elo_gap_half_is_zero():
    """score=0.5 means the agent and opponent are matched — zero gap."""
    assert elo_bisect.implied_elo_gap(0.5) == pytest.approx(0.0, abs=1e-9)


def test_implied_elo_gap_standard_value():
    """The canonical 76% score corresponds to +200 Elo."""
    # 1 / (1 + 10^(-200/400)) ≈ 0.760
    assert elo_bisect.implied_elo_gap(0.76) == pytest.approx(200.0, abs=1.0)


def test_implied_elo_gap_sign_convention():
    """Scoring above 0.5 means *stronger* → positive gap."""
    assert elo_bisect.implied_elo_gap(0.7) > 0
    assert elo_bisect.implied_elo_gap(0.3) < 0


def test_implied_elo_gap_clamps_extremes():
    """Score=0 or score=1 would blow up log10(1/s - 1). Both must
    return finite values."""
    g_zero = elo_bisect.implied_elo_gap(0.0)
    g_one = elo_bisect.implied_elo_gap(1.0)
    assert math.isfinite(g_zero)
    assert math.isfinite(g_one)
    # Clamp is at 1e-3 / 1-1e-3 → |gap| ≈ 400 * log10(999) ≈ 1200.
    assert g_zero == pytest.approx(-g_one, abs=1e-9)
    assert abs(g_zero) < 2000


# ---------- update_bracket --------------------------------------------------

def test_update_bracket_score_above_half_raises_lo():
    lo, hi = elo_bisect.update_bracket(score=0.7, mid=2000, lo=1350, hi=2800)
    assert (lo, hi) == (2000, 2800)


def test_update_bracket_score_below_half_lowers_hi():
    lo, hi = elo_bisect.update_bracket(score=0.3, mid=2000, lo=1350, hi=2800)
    assert (lo, hi) == (1350, 2000)


def test_update_bracket_exact_half_treated_as_at_least():
    """A score of exactly 0.5 means the agent is at the probe Elo —
    the half-step convention is to raise lo."""
    lo, hi = elo_bisect.update_bracket(score=0.5, mid=2000, lo=1350, hi=2800)
    assert (lo, hi) == (2000, 2800)


# ---------- should_stop -----------------------------------------------------

def test_should_stop_when_bracket_tight():
    assert elo_bisect.should_stop(lo=1700, hi=1750, score=0.0) is True
    assert elo_bisect.should_stop(lo=1700, hi=1751, score=0.0) is True  # ≤100
    assert elo_bisect.should_stop(lo=1700, hi=1900, score=0.0) is False


def test_should_stop_when_score_in_bracket():
    """Even with a wide Elo bracket, an in-bracket score means we've
    found a well-matched opponent — stop and report."""
    assert elo_bisect.should_stop(lo=1350, hi=2800, score=0.50) is True
    assert elo_bisect.should_stop(lo=1350, hi=2800, score=0.55) is True
    assert elo_bisect.should_stop(lo=1350, hi=2800, score=0.45) is True
    assert elo_bisect.should_stop(lo=1350, hi=2800, score=0.44) is False
    assert elo_bisect.should_stop(lo=1350, hi=2800, score=0.56) is False


def test_should_stop_honors_custom_bracket_threshold():
    assert elo_bisect.should_stop(lo=1700, hi=1799, score=0.0,
                                  stop_bracket=50) is False
    assert elo_bisect.should_stop(lo=1700, hi=1799, score=0.0,
                                  stop_bracket=100) is True


# ---------- integration: a fake agent at a known Elo ------------------------

def _make_fake_run_match(true_agent_elo: float):
    """Return a run_match-shaped function that pretends the agent has
    Elo `true_agent_elo` and is playing Stockfish at UCI=uci_elo. Score
    follows the Elo curve exactly."""
    def fake(ckpt, uci_elo, n_games, workers, sims, n_blocks, n_filters):
        gap = true_agent_elo - uci_elo
        score = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))
        return {"uci_elo": uci_elo, "n_games": n_games, "score": score,
                "wdl": f"fake @{true_agent_elo}"}
    return fake


def _bisect_with(fake_match):
    """Run the same bisect loop main() runs, but with the fake match
    function. Returns (final_bracket, estimated_elo, n_probes)."""
    lo, hi = elo_bisect.ELO_MIN, elo_bisect.ELO_MAX
    estimated_elo = None
    n_probes = 0
    while hi - lo > elo_bisect.STOP_WHEN_BRACKET_LEQ:
        mid = (lo + hi) // 2
        r = fake_match(None, mid, 40, 8, 800, 20, 256)
        n_probes += 1
        estimated_elo = mid + elo_bisect.implied_elo_gap(r["score"])
        lo, hi = elo_bisect.update_bracket(r["score"], mid, lo, hi)
        if elo_bisect.should_stop(lo, hi, r["score"]):
            break
    return (lo, hi), estimated_elo, n_probes


def test_bisect_converges_on_agent_at_1800():
    bracket, est_elo, n_probes = _bisect_with(_make_fake_run_match(1800))
    lo, hi = bracket
    # Final bracket must contain the truth, OR (if we stopped on
    # in-bracket score) the estimated Elo must.
    assert lo <= 1800 <= hi or abs(est_elo - 1800) < 50
    # Bracket fully traverses 1350..2800 with halves → at most 5 probes
    # to reach ≤100 width; in-bracket early-stop usually finishes sooner.
    assert n_probes <= 6


def test_bisect_converges_on_strong_agent_at_2400():
    bracket, est_elo, n_probes = _bisect_with(_make_fake_run_match(2400))
    lo, hi = bracket
    assert lo <= 2400 <= hi or abs(est_elo - 2400) < 50
    assert n_probes <= 6


def test_bisect_converges_on_weak_agent_at_1400():
    bracket, est_elo, n_probes = _bisect_with(_make_fake_run_match(1400))
    lo, hi = bracket
    # At the bottom of the range, the agent may score very low against
    # the first midpoint probe — but the estimate from the implied gap
    # still localises within the bottom bracket.
    assert lo <= 1400 + 100  # within 100 of the bottom
    assert n_probes <= 6
