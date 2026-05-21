"""One-shot spot-capacity rover: scan → score → render → apply.

Default mode is dry-run (no eksctl/kubectl invocation). Pass --execute to
actually create the cluster + apply the Job.

Examples:

    # Recommend a region, print what it would do, don't touch AWS:
    uv run python scripts/rove.py --config configs/chess-d15-250k.yaml

    # Same, but actually provision (eksctl create cluster takes ~15 min):
    uv run python scripts/rove.py --config configs/chess-d15-250k.yaml --execute

    # Skip the Spot Advisor fetch (offline / firewalled):
    uv run python scripts/rove.py --config configs/chess-d15-250k.yaml --skip-advisor

    # Just the report — no template rendering, no state write:
    uv run python scripts/rove.py --config configs/chess-d15-250k.yaml --report-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from spot_rover.capacity import probe
from spot_rover.provision import apply
from spot_rover.score import RankedCandidate, best_per_region, rank
from spot_rover.state import Deployment, upsert
from spot_rover.template import render_cluster, render_job


def _print_report(ranked: list[RankedCandidate], top_n: int = 10) -> None:
    print(f"\n{'='*72}", flush=True)
    print(f"top {top_n} candidates across all regions (best first):", flush=True)
    print(f"{'='*72}", flush=True)
    print(
        f"{'rank':>4}  {'region':<14}  {'AZ':<14}  {'instance':<14}  "
        f"{'$/hr':>7}  {'$/vcpu':>7}  {'disc%':>5}  {'stab':>4}  "
        f"{'intr':>6}  {'total':>5}",
        flush=True,
    )
    for i, c in enumerate(ranked[:top_n], 1):
        disc = c.probe.discount_vs_on_demand
        disc_str = f"{disc*100:>4.0f}%" if disc is not None else "  ?"
        intr = c.probe.interrupt_band or "?"
        print(
            f"{i:>4}  {c.probe.region:<14}  {c.probe.az:<14}  "
            f"{c.probe.instance_type:<14}  "
            f"{c.probe.spot_price:>7.4f}  "
            f"{c.probe.spot_per_vcpu:>7.4f}  "
            f"{disc_str:>5}  "
            f"{c.stability_score:>4.2f}  "
            f"{intr:>6}  "
            f"{c.total:>5.3f}",
            flush=True,
        )


def _print_per_region(ranked: list[RankedCandidate]) -> None:
    per_r = best_per_region(ranked)
    print(f"\n{'='*72}", flush=True)
    print("best per region:", flush=True)
    print(f"{'='*72}", flush=True)
    for i, c in enumerate(per_r, 1):
        print(
            f"  {i}. {c.probe.region:<14}  "
            f"{c.probe.instance_type:<14} in {c.probe.az}  "
            f"@ ${c.probe.spot_price:.4f}/hr  (score={c.total:.3f})",
            flush=True,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--execute", action="store_true",
                   help="actually run eksctl + kubectl (default: dry-run)")
    p.add_argument("--report-only", action="store_true",
                   help="capacity report only; skip rendering and state write")
    p.add_argument("--skip-advisor", action="store_true",
                   help="don't fetch the Spot Advisor interruption-rate data")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write rendered cluster + job YAML (default: tempdir)")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    print(
        f"workload: {cfg['workload']}\n"
        f"regions:  {', '.join(cfg['regions'])}\n"
        f"types:    {', '.join(cfg['instance_types'])}\n",
        flush=True,
    )
    print("probing AWS spot-price history (this takes ~1-2 min) ...", flush=True)
    probes = probe(
        regions=cfg["regions"],
        instance_types=cfg["instance_types"],
        skip_advisor=args.skip_advisor,
    )
    if not probes:
        print("ERROR: no spot price history in any region × instance-type combo. "
              "Try widening --instance-types or check IAM perms.", flush=True)
        return 2
    print(f"  {len(probes)} (region, AZ, type) combos sampled", flush=True)

    # min_instance_tier defaults to 1 (compute-optimized only). Workload
    # configs can lower it (e.g. 0 for GPU) or raise it (2 to explicitly
    # allow general-purpose fallback). See score.rank docstring + the
    # project_spot_capacity_snapshot memory for context.
    min_tier = cfg.get("min_instance_tier", 1)
    ranked = rank(probes, min_tier=min_tier)
    if not ranked:
        tiers_seen = sorted({p.tier for p in probes})
        print(
            f"\nERROR: no candidates passed min_instance_tier={min_tier} "
            f"(seen tiers: {tiers_seen}). Either widen the regions, add "
            f"higher-tier instance types to the config, or relax "
            f"`min_instance_tier` if you really mean to accept a lower "
            f"tier (warning: tier 2 is ~4x slower per core for Stockfish).",
            flush=True,
        )
        return 3
    _print_report(ranked, top_n=args.top_n)
    _print_per_region(ranked)

    if args.report_only:
        return 0

    # Pick the winner (best overall composite score), then collect same-region
    # siblings as fallback instance types for the nodegroup.
    winner = ranked[0]
    same_region = [c for c in ranked
                   if c.probe.region == winner.probe.region]

    rendered = render_cluster(
        winner=winner,
        siblings=same_region,
        workload=cfg["workload"],
        desired_capacity=cfg["desired_capacity"],
        min_size=cfg.get("min_size", 0),
        max_size=cfg.get("max_size"),
    )

    job_template_path = Path(cfg["job_template"])
    if not job_template_path.is_absolute():
        job_template_path = args.config.resolve().parent.parent.parent.parent / job_template_path
    if not job_template_path.exists():
        print(f"\nWARN: job template not found at {job_template_path} — skipping job render.",
              flush=True)
        job_yaml = ""
    else:
        job_yaml = render_job(
            job_template_path,
            ecr_uri=cfg["ecr_uri"],
            s3_region=cfg["s3_region"],
            s3_bucket=cfg["s3_bucket"],
            s3_prefix=cfg["s3_prefix"],
        )

    print(
        f"\n{'='*72}\n"
        f"selected: {rendered.cluster_name} in {rendered.region}\n"
        f"  primary instance type: {rendered.primary_instance_type}\n"
        f"  fallback types:        {', '.join(rendered.fallback_instance_types) or '(none)'}\n"
        f"  preferred AZ:          {rendered.az}\n"
        f"  desired capacity:      {cfg['desired_capacity']}\n"
        f"{'='*72}",
        flush=True,
    )

    result = apply(
        cluster_yaml=rendered.cluster_yaml,
        job_yaml=job_yaml,
        cluster_name=rendered.cluster_name,
        region=rendered.region,
        dry_run=not args.execute,
        work_dir=args.out_dir,
    )

    print("\nprovision log:", flush=True)
    for line in result.log:
        print(f"  {line}", flush=True)
    print(
        f"\ncluster yaml: {result.cluster_yaml_path}\n"
        f"job yaml:     {result.job_yaml_path}\n"
        f"created:      {result.cluster_created}  applied: {result.job_applied}",
        flush=True,
    )

    if args.execute and result.cluster_created and result.job_applied:
        deployment = Deployment(
            deployment_id=rendered.cluster_name,
            workload=cfg["workload"],
            region=rendered.region,
            cluster_name=rendered.cluster_name,
            job_name=cfg["workload"],
            instance_type=rendered.primary_instance_type,
            primary_az=rendered.az,
            fallback_instance_types=list(rendered.fallback_instance_types),
        )
        uri = upsert(
            deployment,
            bucket=cfg["state_bucket"],
            prefix=cfg.get("state_prefix", "spot-rover"),
        )
        print(f"\nstate recorded: {uri}", flush=True)

    return 0 if result.job_applied or not args.execute else 1


if __name__ == "__main__":
    sys.exit(main())
