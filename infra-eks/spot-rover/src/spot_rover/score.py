"""Rank CapacityProbes by composite score.

Score factors (higher = more favorable):
  - **Cost**: lower spot $/vCPU is better. The dominant factor for CPU-bound
    Stockfish-style datagen since wall-clock scales linearly with vCPUs.
  - **Stability**: lower 7-day price stddev is better. High variance ≈ tight
    capacity ≈ likely interruption.
  - **Discount**: spot deeply below on-demand suggests slack; near-parity
    means the pool is hot.
  - **Interruption rate**: Spot Advisor band, when available.

Composite score = cost_score × stability_score × discount_score × interrupt_score.
All factors normalized to [0, 1] so the product is interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .capacity import CapacityProbe


_INTERRUPT_TO_FACTOR = {
    "<5%":   1.00,
    "5-10%": 0.70,
    "10-15%": 0.40,
    "15-20%": 0.20,
    ">20%":  0.05,
    None:    0.50,    # unknown: neutral
}


@dataclass(frozen=True)
class RankedCandidate:
    probe: CapacityProbe
    cost_score: float
    stability_score: float
    discount_score: float
    interrupt_score: float

    @property
    def total(self) -> float:
        return (self.cost_score
                * self.stability_score
                * self.discount_score
                * self.interrupt_score)


def _cost_score(probe: CapacityProbe, cheapest_per_vcpu: float) -> float:
    """Best (cheapest) probe = 1.0. Worst (10× the cheapest) = 0.1."""
    per_vcpu = probe.spot_per_vcpu
    # Unknown vCPU count → spot_per_vcpu is inf; score it 0 (won't win but won't NaN).
    if per_vcpu == float("inf") or per_vcpu <= 0:
        return 0.0
    ratio = cheapest_per_vcpu / per_vcpu
    return min(max(ratio, 0.0), 1.0)


def _stability_score(probe: CapacityProbe) -> float:
    """7-day stddev as fraction of price. <5% stddev = 1.0; >50% = ~0.0."""
    if probe.spot_price <= 0:
        return 0.0
    rel = probe.price_stddev_7d / probe.spot_price
    return max(0.0, 1.0 - (rel / 0.5))


def _discount_score(probe: CapacityProbe) -> float:
    """Spot at 30% of OD = 1.0; spot at OD = 0.0."""
    if probe.on_demand_price is None:
        return 0.5  # unknown: neutral
    discount = probe.discount_vs_on_demand
    if discount is None:
        return 0.5
    return max(0.0, min(1.0, (discount - 0.3) / 0.7 + 0.3))


def rank(
    probes: Iterable[CapacityProbe],
    *,
    min_tier: int | None = 1,
) -> list[RankedCandidate]:
    """Return probes sorted by composite score, best first.

    `min_tier` filters out probes worse than the given workload tier
    (lower number = better tier). Default 1 = compute-optimized only;
    callers that genuinely want to allow general-purpose fallback must
    pass `min_tier=2` explicitly. Pass `None` to disable the filter
    entirely (e.g. for GPU workloads — set the workload config's
    `min_instance_tier: 0` to do this).

    Rationale: in 2026-05-20's eu-central-1 failover, the cluster
    silently took m7i.8xlarge spot (tier 2) because c-family was out,
    paying a 4× per-core Stockfish penalty for the whole 30+ hour
    datagen. The default `min_tier=1` makes that failure mode loud:
    the rover returns an empty ranking, the operator sees it, and
    moves regions instead of degrading silently.
    """
    probes = list(probes)
    if min_tier is not None:
        probes = [p for p in probes if p.tier <= min_tier]
    if not probes:
        return []
    cheapest = min(p.spot_per_vcpu for p in probes if p.spot_per_vcpu > 0)
    out: list[RankedCandidate] = []
    for p in probes:
        out.append(RankedCandidate(
            probe=p,
            cost_score=_cost_score(p, cheapest),
            stability_score=_stability_score(p),
            discount_score=_discount_score(p),
            interrupt_score=_INTERRUPT_TO_FACTOR.get(p.interrupt_band, 0.5),
        ))
    out.sort(key=lambda c: c.total, reverse=True)
    return out


def best_per_region(
    ranked: list[RankedCandidate],
) -> list[RankedCandidate]:
    """One winner per region — top-scoring (AZ, instance-type) for each."""
    seen: set = set()
    out: list[RankedCandidate] = []
    for c in ranked:
        if c.probe.region in seen:
            continue
        seen.add(c.probe.region)
        out.append(c)
    return out
