"""Unit tests for scoring. No AWS needed — synthetic CapacityProbes."""
from __future__ import annotations

import pytest

from spot_rover.capacity import CapacityProbe
from spot_rover.score import best_per_region, rank


def _probe(region: str, az: str, itype: str, price: float,
           on_demand: float | None = None, stddev: float = 0.0,
           interrupt: str | None = "<5%", vcpus: int = 32,
           tier: int = 1) -> CapacityProbe:
    return CapacityProbe(
        region=region, az=az, instance_type=itype,
        spot_price=price, on_demand_price=on_demand,
        price_stddev_7d=stddev, interrupt_band=interrupt, vcpus=vcpus,
        tier=tier,
    )


def test_cheapest_with_stable_price_wins():
    """Two identical interrupt bands, same vCPU: cheaper one ranks higher."""
    a = _probe("us-east-1", "a1", "c7a.8xlarge", 0.30, on_demand=1.50)
    b = _probe("eu-central-1", "b1", "c7a.8xlarge", 0.45, on_demand=1.50)
    out = rank([a, b])
    assert out[0].probe.region == "us-east-1"
    assert out[1].probe.region == "eu-central-1"


def test_volatile_price_loses_to_stable_at_same_cost():
    """Same price, but one has high 7-day stddev → other wins."""
    stable = _probe("us-east-1", "a1", "c7a.8xlarge", 0.30,
                    on_demand=1.50, stddev=0.005)
    volatile = _probe("us-east-1", "a2", "c7a.8xlarge", 0.30,
                      on_demand=1.50, stddev=0.10)
    out = rank([stable, volatile])
    assert out[0].probe.az == "a1"


def test_high_interrupt_band_loses():
    a = _probe("us-east-1", "a1", "c7a.8xlarge", 0.30,
               on_demand=1.50, interrupt="<5%")
    b = _probe("us-east-1", "a1", "c7a.8xlarge", 0.30,
               on_demand=1.50, interrupt=">20%")
    out = rank([a, b])
    assert out[0].interrupt_score > out[1].interrupt_score
    assert out[0].total > out[1].total


def test_spot_at_on_demand_price_kills_discount():
    no_discount = _probe("us-east-1", "a1", "c7a.8xlarge", 1.50,
                         on_demand=1.50)
    deep_discount = _probe("us-east-1", "a1", "c7a.8xlarge", 0.30,
                           on_demand=1.50)
    out = rank([no_discount, deep_discount])
    assert out[0].probe.spot_price < out[1].probe.spot_price
    assert out[1].discount_score < out[0].discount_score


def test_best_per_region_dedupes():
    rs = rank([
        _probe("us-east-1", "a1", "c7a.8xlarge", 0.30, on_demand=1.50),
        _probe("us-east-1", "a2", "c7a.8xlarge", 0.45, on_demand=1.50),
        _probe("eu-central-1", "b1", "c7a.8xlarge", 0.50, on_demand=1.50),
    ])
    out = best_per_region(rs)
    regions = [c.probe.region for c in out]
    assert regions == ["us-east-1", "eu-central-1"]
    # winner per region is the higher-scoring one
    assert out[0].probe.az == "a1"


def test_rank_empty_returns_empty():
    assert rank([]) == []


def test_zero_vcpu_doesnt_crash():
    """Unknown instance type (vcpus=0) shouldn't div-by-zero."""
    p = CapacityProbe(
        region="us-east-1", az="a1", instance_type="unknown.unknown",
        spot_price=0.30, on_demand_price=None, price_stddev_7d=0.0,
        interrupt_band=None, vcpus=0, tier=1,
    )
    out = rank([p])
    assert len(out) == 1
    # Should rank — total is finite (just not necessarily great).
    assert out[0].total >= 0


def test_tier_filter_default_excludes_m_family():
    """rank() with default min_tier=1 must drop tier-2 (general-purpose)."""
    c_family = _probe("us-east-2", "a1", "c7a.8xlarge", 0.28,
                      on_demand=1.50, tier=1)
    m_family = _probe("eu-central-1", "b1", "m7i.8xlarge", 0.35,
                      on_demand=1.61, tier=2)
    out = rank([c_family, m_family])
    assert len(out) == 1
    assert out[0].probe.instance_type == "c7a.8xlarge"


def test_tier_filter_relaxed_includes_m_family():
    """min_tier=2 lets tier-2 in (with the documented 4x perf warning)."""
    c_family = _probe("us-east-2", "a1", "c7a.8xlarge", 0.28,
                      on_demand=1.50, tier=1)
    m_family = _probe("eu-central-1", "b1", "m7i.8xlarge", 0.35,
                      on_demand=1.61, tier=2)
    out = rank([c_family, m_family], min_tier=2)
    assert len(out) == 2


def test_tier_filter_none_disables():
    """min_tier=None lets everything through, including unknown tier=9."""
    unknown = _probe("us-east-1", "a1", "unknown.unknown", 0.30, tier=9)
    out = rank([unknown], min_tier=None)
    assert len(out) == 1


def test_tier_filter_empty_when_no_tier1_present():
    """If only tier-2 probes are passed, default-filtered rank returns []."""
    m1 = _probe("eu-central-1", "a", "m7i.8xlarge", 0.35, tier=2)
    m2 = _probe("eu-west-1", "b", "m7i.8xlarge", 0.42, tier=2)
    assert rank([m1, m2]) == []                  # default filter: nothing
    assert len(rank([m1, m2], min_tier=2)) == 2  # explicit relax: both pass
