"""Tests for the deprecation stubs that replaced the Terraform-based
orchestrators.

After infra/ was removed in favor of infra-eks/, two scripts that drove
the old Terraform single-box pipeline became dead weight:

  - wm_chess/scripts/overnight_datagen.sh    (overnight datagen orchestrator)
  - experiments/distill-soft/scripts/run_split_pipeline.sh (split CPU/GPU)

We replaced both with deprecation stubs that exit non-zero and print a
pointer to the new EKS-based flow. These tests are defensive — they
fail if someone reintroduces a working invocation path without thinking
about whether the deprecation is still warranted, or if a future edit
silently breaks the "fail-fast" contract.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

STUBS = [
    REPO_ROOT / "wm_chess" / "scripts" / "overnight_datagen.sh",
    REPO_ROOT / "experiments" / "distill-soft" / "scripts" / "run_split_pipeline.sh",
]


@pytest.mark.parametrize("stub", STUBS, ids=lambda p: p.name)
def test_stub_exists_and_is_executable(stub):
    assert stub.exists(), f"missing deprecation stub: {stub}"
    assert stub.stat().st_mode & 0o111, f"stub not executable: {stub}"


@pytest.mark.parametrize("stub", STUBS, ids=lambda p: p.name)
def test_stub_exits_nonzero(stub):
    """Anything that habitually invoked these (cron, manual, scripts)
    should fail fast rather than silently succeed."""
    result = subprocess.run(
        ["bash", str(stub)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode != 0, (
        f"deprecation stub {stub.name} returned 0 — must exit non-zero so "
        f"callers fail fast"
    )


@pytest.mark.parametrize("stub", STUBS, ids=lambda p: p.name)
def test_stub_points_at_infra_eks(stub):
    """Stub output must include a clear pointer to the EKS replacement
    so the operator knows where to look."""
    result = subprocess.run(
        ["bash", str(stub)], capture_output=True, text=True, timeout=10
    )
    combined = (result.stdout + result.stderr).lower()
    assert "infra-eks" in combined, (
        f"deprecation stub {stub.name} doesn't mention infra-eks; "
        f"operators won't know where to look. Got: {combined[:300]!r}"
    )
