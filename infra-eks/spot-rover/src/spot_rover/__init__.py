"""Spot-rover: scan regions for available spot capacity, deploy an EKS job there.

Use case: when us-east-1 spot pool went hostile on 2026-05-20 (pods reclaimed
within 30s of spawning), the failover to eu-central-1 was manual — pick
region by hand, render `cluster-gen-d15-250k-eu.yaml` by hand, apply by hand,
kick the job by hand. This package automates that decision + apply.

Workflow:

    1. `capacity.probe(regions, instance_families)` →
       list[CapacityProbe], one per (region, AZ, instance-type).
    2. `score.rank(probes)` → sorted by composite score (cost × stability ×
       interruption rate).
    3. `template.render(workload, winner)` → eksctl ClusterConfig YAML.
    4. `provision.apply(rendered_yaml, job_yaml)` → eksctl create cluster
       + kubectl apply -f job.
    5. `state.record(deployment)` → S3 deployment registry so a future run
       knows what's already deployed where.

MVP scope: one-shot. A future continuous-daemon mode would watch the
deployed fleet for spot interruptions and re-roll to a new region.
"""
from .capacity import CapacityProbe, probe
from .score import RankedCandidate, rank
from .state import Deployment, load_state, save_state

__all__ = [
    "CapacityProbe",
    "RankedCandidate",
    "Deployment",
    "probe",
    "rank",
    "load_state",
    "save_state",
]
