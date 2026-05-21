"""S3-backed deployment registry.

A single JSON object at `s3://<bucket>/spot-rover/state.json` lists every
deployment the rover has created. The continuous-daemon mode (phase 2)
reads this to know which clusters to monitor; for the one-shot MVP it
serves as a record so the user can tear down deployments later.

The state object is small (one record per deployment, dozens at most), so
S3 PutObject + last-writer-wins semantics are fine. Concurrent rovers
across regions are not a target use case.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import boto3
from botocore.exceptions import ClientError


@dataclass
class Deployment:
    deployment_id: str          # cluster name (unique)
    workload: str               # e.g. "wm-chess-d15-250k"
    region: str
    cluster_name: str
    job_name: str
    instance_type: str
    primary_az: str
    fallback_instance_types: list[str] = field(default_factory=list)
    created_at: str = ""
    status: str = "active"      # "active" | "torn-down" | "interrupted"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


def _s3_state_uri(bucket: str, prefix: str = "spot-rover") -> tuple[str, str]:
    return bucket, f"{prefix.strip('/')}/state.json"


def load_state(bucket: str, prefix: str = "spot-rover") -> list[Deployment]:
    """Read the state JSON from S3. Returns [] if not present yet."""
    b, key = _s3_state_uri(bucket, prefix)
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=b, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise
    payload = json.loads(obj["Body"].read())
    return [Deployment(**d) for d in payload.get("deployments", [])]


def save_state(
    deployments: Iterable[Deployment],
    bucket: str,
    prefix: str = "spot-rover",
) -> str:
    """Overwrite the state JSON. Returns the s3:// URI for confirmation."""
    b, key = _s3_state_uri(bucket, prefix)
    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "deployments": [asdict(d) for d in deployments],
    }
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=b, Key=key,
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )
    return f"s3://{b}/{key}"


def upsert(
    deployment: Deployment,
    bucket: str,
    prefix: str = "spot-rover",
) -> str:
    """Read state, replace-or-append by deployment_id, write back."""
    existing = load_state(bucket, prefix=prefix)
    existing = [d for d in existing if d.deployment_id != deployment.deployment_id]
    existing.append(deployment)
    return save_state(existing, bucket, prefix=prefix)
