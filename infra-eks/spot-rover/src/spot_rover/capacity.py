"""AWS capacity discovery via boto3.

For each (region, AZ, instance-type) tuple we want to know:
  - Current spot price (latest sample from DescribeSpotPriceHistory)
  - Price stability (stddev over last 7 days — high variance ≈ tight capacity)
  - On-demand price (baseline, from a static table or Pricing API)
  - Interruption-rate band (optional, from the Spot Advisor dataset)
  - vCPU count (for $/core normalization)

These are all proxies — AWS doesn't expose "available capacity right now."
The only ground-truth signal is to actually attempt to fulfill a fleet
request and handle `InsufficientCapacity`. We may add that as a
last-mile confirmation, but for scoring across N×M combos it's too slow
and would itself burn spot capacity.
"""
from __future__ import annotations

import json
import statistics
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import boto3


# Static on-demand prices ($/hr) for the instance types we care about.
# Pulled from AWS pricing on 2026-05-20 — not load-bearing for scoring
# (used only as a sanity baseline). Update via the Pricing API if drift
# becomes a problem; for the MVP this is fine.
ON_DEMAND_PRICES = {
    # c7a (AMD EPYC, 4th gen): chess datagen recommended instance family
    "c7a.4xlarge":  0.7344,
    "c7a.8xlarge":  1.4688,
    "c7a.16xlarge": 2.9376,
    # c7i (Intel Sapphire Rapids): similar perf-per-$
    "c7i.4xlarge":  0.7140,
    "c7i.8xlarge":  1.4280,
    "c7i.16xlarge": 2.8560,
    # c6i (Intel Ice Lake): slightly older, often more available
    "c6i.4xlarge":  0.6800,
    "c6i.8xlarge":  1.3600,
    "c6i.16xlarge": 2.7200,
    # c6a (AMD Milan):
    "c6a.4xlarge":  0.6120,
    "c6a.8xlarge":  1.2240,
    "c6a.16xlarge": 2.4480,
    # m7i (mixed CPU/mem for KataGo-style workloads):
    "m7i.4xlarge":  0.8064,
    "m7i.8xlarge":  1.6128,
    # GPU (for student-side training, not datagen): g5 family
    "g5.xlarge":    1.006,
    "g5.2xlarge":   1.212,
    "g5.4xlarge":   1.624,
    "g5.8xlarge":   2.448,
    "g6.xlarge":    0.8048,
    "g6.2xlarge":   0.9776,
    "g6.4xlarge":   1.3232,
    "g6.8xlarge":   2.0144,
}


VCPUS = {
    "c7a.4xlarge": 16, "c7a.8xlarge": 32, "c7a.16xlarge": 64,
    "c7i.4xlarge": 16, "c7i.8xlarge": 32, "c7i.16xlarge": 64,
    "c6i.4xlarge": 16, "c6i.8xlarge": 32, "c6i.16xlarge": 64,
    "c6a.4xlarge": 16, "c6a.8xlarge": 32, "c6a.16xlarge": 64,
    "m7i.4xlarge": 16, "m7i.8xlarge": 32,
    "g5.xlarge": 4,  "g5.2xlarge": 8,  "g5.4xlarge": 16, "g5.8xlarge": 32,
    "g6.xlarge": 4,  "g6.2xlarge": 8,  "g6.4xlarge": 16, "g6.8xlarge": 32,
}


SPOT_ADVISOR_URL = (
    "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"
)


@dataclass(frozen=True)
class CapacityProbe:
    """A single (region, AZ, instance-type) capacity sample."""

    region: str
    az: str
    instance_type: str
    spot_price: float
    on_demand_price: float | None
    price_stddev_7d: float
    interrupt_band: str | None    # "<5%" | "5-10%" | "10-15%" | "15-20%" | ">20%"
    vcpus: int

    @property
    def spot_per_vcpu(self) -> float:
        return self.spot_price / self.vcpus if self.vcpus else float("inf")

    @property
    def discount_vs_on_demand(self) -> float | None:
        """Fraction off on-demand: 0.7 means spot is 30% of on-demand."""
        if self.on_demand_price is None:
            return None
        return 1.0 - (self.spot_price / self.on_demand_price)


def _fetch_spot_advisor() -> dict | None:
    """Fetch the Spot Advisor interruption-rate dataset.

    Best-effort: returns None if the fetch fails. The score will still
    work without it; interruption-rate-aware ranking is just a bonus.
    """
    try:
        with urllib.request.urlopen(SPOT_ADVISOR_URL, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


_INTERRUPT_INDEX_TO_BAND = {
    0: "<5%",
    1: "5-10%",
    2: "10-15%",
    3: "15-20%",
    4: ">20%",
}


def _interrupt_band_for(
    advisor: dict | None,
    region: str,
    instance_type: str,
) -> str | None:
    if advisor is None:
        return None
    try:
        r = advisor["spot_advisor"][region]["Linux"][instance_type]
        return _INTERRUPT_INDEX_TO_BAND.get(int(r["r"]))
    except (KeyError, TypeError, ValueError):
        return None


def _latest_spot_per_az(
    region: str,
    instance_types: list[str],
    window_days: int = 7,
) -> dict[tuple[str, str], list[float]]:
    """Return all spot prices in the last `window_days` per (AZ, instance-type).

    AWS returns one event per price-change, oldest-first (within paginated
    pages). We collect them all and let the caller take the latest + stddev.
    """
    ec2 = boto3.client("ec2", region_name=region)
    start_time = datetime.now(timezone.utc) - timedelta(days=window_days)
    out: dict[tuple[str, str], list[float]] = {}
    paginator = ec2.get_paginator("describe_spot_price_history")
    pages = paginator.paginate(
        InstanceTypes=instance_types,
        ProductDescriptions=["Linux/UNIX"],
        StartTime=start_time,
    )
    for page in pages:
        for evt in page.get("SpotPriceHistory", []):
            key = (evt["AvailabilityZone"], evt["InstanceType"])
            try:
                price = float(evt["SpotPrice"])
            except (TypeError, ValueError):
                continue
            out.setdefault(key, []).append(price)
    return out


def probe(
    regions: Iterable[str],
    instance_types: list[str],
    *,
    window_days: int = 7,
    skip_advisor: bool = False,
) -> list[CapacityProbe]:
    """Probe spot capacity across (region × AZ × instance-type) combinations.

    Returns one CapacityProbe per combo with at least one price sample in
    the window. Combos with no spot history (typically meaning the instance
    type is unavailable in that AZ) are silently dropped — they'd be
    unrankable.
    """
    advisor = None if skip_advisor else _fetch_spot_advisor()
    probes: list[CapacityProbe] = []
    for region in regions:
        per_az = _latest_spot_per_az(region, instance_types,
                                     window_days=window_days)
        for (az, itype), prices in per_az.items():
            if not prices:
                continue
            latest = prices[0]   # AWS returns newest-first within page
            stddev = (statistics.pstdev(prices)
                      if len(prices) > 1 else 0.0)
            probes.append(CapacityProbe(
                region=region,
                az=az,
                instance_type=itype,
                spot_price=latest,
                on_demand_price=ON_DEMAND_PRICES.get(itype),
                price_stddev_7d=stddev,
                interrupt_band=_interrupt_band_for(advisor, region, itype),
                vcpus=VCPUS.get(itype, 0),
            ))
    return probes
