"""Render the eksctl ClusterConfig template from a winning candidate.

The template uses `{{var}}` placeholders so it stays readable as plain YAML
when inspected. We deliberately avoid Jinja or any templating dep — the
substitutions are simple string replaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .score import RankedCandidate

# Cluster naming: <workload>-<region-suffix> so two workloads can co-exist.
_REGION_SUFFIX = {
    "us-east-1": "use1",
    "us-east-2": "use2",
    "us-west-2": "usw2",
    "eu-central-1": "euc1",
    "eu-west-1": "euw1",
    "ap-northeast-1": "apne1",
    "ap-southeast-1": "apse1",
}


@dataclass(frozen=True)
class RenderedCluster:
    cluster_yaml: str
    cluster_name: str
    region: str
    az: str
    primary_instance_type: str
    fallback_instance_types: list[str]


def cluster_name(workload: str, region: str) -> str:
    suffix = _REGION_SUFFIX.get(region, region.replace("-", ""))
    return f"{workload}-{suffix}"


def render_cluster(
    winner: RankedCandidate,
    siblings: list[RankedCandidate],
    *,
    workload: str,
    desired_capacity: int,
    min_size: int = 0,
    max_size: int | None = None,
    template_path: Path | None = None,
    extra_tags: dict[str, str] | None = None,
) -> RenderedCluster:
    """Render the cluster YAML for `winner` with `siblings` as fallback types.

    `siblings` should be other RankedCandidates IN THE SAME REGION that ranked
    well — they become the nodegroup's fallback instance types so eksctl
    can use them if the winner's family is unavailable. Pass the rover's
    full ranked list (filtered to same region) here.
    """
    if template_path is None:
        template_path = (
            Path(__file__).resolve().parents[2] / "templates" / "cluster.yaml.tmpl"
        )
    text = template_path.read_text()

    name = cluster_name(workload, winner.probe.region)
    fallback_types: list[str] = []
    seen = {winner.probe.instance_type}
    for sib in siblings:
        if sib.probe.region != winner.probe.region:
            continue
        if sib.probe.instance_type in seen:
            continue
        seen.add(sib.probe.instance_type)
        fallback_types.append(sib.probe.instance_type)
        if len(fallback_types) >= 5:   # keep the list bounded
            break

    instance_types = [winner.probe.instance_type] + fallback_types
    instance_types_yaml = "\n".join(f"      - {t}" for t in instance_types)

    if max_size is None:
        max_size = max(desired_capacity + 2, desired_capacity)

    tags_yaml = ""
    if extra_tags:
        tags_yaml = "\n".join(f"    {k}: {v}" for k, v in extra_tags.items())

    rendered = (
        text.replace("{{name}}", name)
            .replace("{{region}}", winner.probe.region)
            .replace("{{az}}", winner.probe.az)
            .replace("{{workload}}", workload)
            .replace("{{instance_types}}", instance_types_yaml)
            .replace("{{desired_capacity}}", str(desired_capacity))
            .replace("{{min_size}}", str(min_size))
            .replace("{{max_size}}", str(max_size))
            .replace("{{tags}}", tags_yaml)
    )
    return RenderedCluster(
        cluster_yaml=rendered,
        cluster_name=name,
        region=winner.probe.region,
        az=winner.probe.az,
        primary_instance_type=winner.probe.instance_type,
        fallback_instance_types=fallback_types,
    )


def render_job(
    job_template_path: Path,
    *,
    ecr_uri: str,
    s3_region: str,
    s3_bucket: str,
    s3_prefix: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Render a Job YAML from a template with envsubst-style placeholders.

    Matches the existing `infra-eks/k8s/job-*.yaml` convention. `overrides`
    can patch arbitrary placeholders for workload-specific knobs.
    """
    text = job_template_path.read_text()
    subs = {
        "ECR_URI": ecr_uri,
        "S3_REGION": s3_region,
        "S3_BUCKET": s3_bucket,
        "S3_PREFIX": s3_prefix,
    }
    if overrides:
        subs.update(overrides)
    for k, v in subs.items():
        text = text.replace("${" + k + "}", v)
    return text
