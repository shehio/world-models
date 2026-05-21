"""Drive eksctl + kubectl to provision the chosen region.

The rover writes rendered YAML to a temp dir, then either prints the
commands (dry-run) or invokes them. Provisioning is intentionally a thin
shell over `eksctl` and `kubectl` — both are mature, well-tested tools
and there's no reason to reimplement their idempotency/retry handling.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProvisionResult:
    cluster_name: str
    region: str
    cluster_yaml_path: Path
    job_yaml_path: Path
    cluster_created: bool
    job_applied: bool
    log: list[str]


def _run(cmd: list[str], log: list[str], dry_run: bool) -> int:
    """Run a command, capturing into the log. Returns exit code."""
    pretty = " ".join(cmd)
    log.append(f"$ {pretty}")
    if dry_run:
        log.append("  (dry-run, not executed)")
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log.append(f"  {line}")
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log.append(f"  ! {line}")
    return proc.returncode


def apply(
    cluster_yaml: str,
    job_yaml: str,
    cluster_name: str,
    region: str,
    *,
    dry_run: bool = True,
    work_dir: Path | None = None,
) -> ProvisionResult:
    """Provision the cluster + apply the Job.

    If `dry_run`, the rendered YAML is written but no eksctl/kubectl
    is invoked — useful for review before committing real cloud spend.
    """
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="spot-rover-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    cluster_path = work_dir / f"cluster-{cluster_name}.yaml"
    job_path = work_dir / f"job-{cluster_name}.yaml"
    cluster_path.write_text(cluster_yaml)
    job_path.write_text(job_yaml)

    log: list[str] = []
    log.append(f"wrote {cluster_path}")
    log.append(f"wrote {job_path}")

    rc = _run(
        ["eksctl", "create", "cluster", "-f", str(cluster_path)],
        log, dry_run,
    )
    cluster_created = (rc == 0)
    if not cluster_created and not dry_run:
        return ProvisionResult(
            cluster_name=cluster_name, region=region,
            cluster_yaml_path=cluster_path, job_yaml_path=job_path,
            cluster_created=False, job_applied=False, log=log,
        )

    # eksctl create cluster also configures kubectl. Be explicit anyway in
    # case the user's kube-config is shared across regions.
    context = f"{cluster_name}.{region}.eksctl.io"
    _run(
        ["aws", "eks", "update-kubeconfig",
         "--name", cluster_name, "--region", region],
        log, dry_run,
    )
    rc = _run(
        ["kubectl", "--context", f"iam-root-account@{context}",
         "apply", "-f", str(job_path)],
        log, dry_run,
    )
    job_applied = (rc == 0)

    return ProvisionResult(
        cluster_name=cluster_name, region=region,
        cluster_yaml_path=cluster_path, job_yaml_path=job_path,
        cluster_created=cluster_created, job_applied=job_applied, log=log,
    )
