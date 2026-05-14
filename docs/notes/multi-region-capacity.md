# Multi-region capacity

us-east-1 + eu-central-1 fallback doubles effective G/VT vCPU quota.
Cross-region ECR pull and S3 read "just work" — no code changes
needed.

AWS quota **L-DB2E81BA** (G/VT instances) caps per-region usage at
32 vCPU. This collides with our workload patterns:

- `g6.4xlarge` eval = 16 vCPU. Two concurrent evals = 32 vCPU → next
  launch hits `VcpuLimitExceeded`.
- `g6e.xlarge` training (L40S) = 4 vCPU. Fine alone.
- `g6e.8xlarge` full-30M training = 32 vCPU. Won't even start if any
  other G/VT instance is alive in the same region.

**Pattern:** run in us-east-1 first, fall back to eu-central-1. The
two regions have independent quotas → ~64 effective vCPU.

## Cross-region wiring is invisible to the workload

- ECR image lives in
  `594561963943.dkr.ecr.us-east-1.amazonaws.com` and is
  cross-region-pulled (docker login + docker pull from any region).
- S3 buckets live in us-east-1 and are cross-region read/written
  (`--region us-east-1` on the `aws s3 cp` inside user-data).
- IAM instance profile (`wm-chess-merge-instance-profile`) is global.

## Region constants

- **us-east-1**: `subnet-042fb3c497e2631a7`, AMI
  `ami-027c3ae8019fc0d3a` (DL Base GPU AL2023).
- **eu-central-1**: `subnet-0a4bed60`, AMI
  `ami-01e9d13d4c5e54237` (same family).

## Where this pattern is implemented

- `infra-eks/daemons/wm-autoeval-daemon.sh` —
  `REGION_ORDER=(us-east-1 eu-central-1)`, `region_config()` returns
  subnet+AMI per region, `try_launch_in_region()` iterates with
  claim-marker rollback on total failure.
- `infra-eks/launchers/*.sh` — each is hardcoded to one region but
  easy to sed-edit `LAUNCH_REGION` / `AMI` / `SUBNET` for a
  retry-across-regions watcher; see
  [`bare-ec2-launchers.md`](./bare-ec2-launchers.md).

**Why:** d10/d15 training week constantly fought `VcpuLimitExceeded`
— adding eu-central-1 as a quota-pool partner doubled throughput
without any quota-increase tickets.

**How to apply:** Any new wm-chess GPU EC2 launch (eval, training,
ablation) should be parameterizable for region. If a single region's
quota is tight, just launch in the other one — no other changes
needed for cross-region S3 + ECR.
