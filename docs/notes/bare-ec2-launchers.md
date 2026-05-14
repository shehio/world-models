# Bare-EC2 launchers

`infra-eks/launchers/` — bare-EC2 one-shots (L40S training, deep-sim
eval, capacity ablation). Used when EKS can't get the instance type or
for one-off experiments.

`infra-eks/launchers/` holds bare-EC2 launchers — one-shot scripts
that launch a single EC2, run a job, and self-terminate. Used when:

- EKS can't get a specific instance type (e.g. `g6e` in us-east-1 was
  out of L40S capacity in our cluster's AZ).
- A one-off ablation doesn't justify a new EKS Job submission.

## The pattern (every launcher)

1. Constants block: `ACCOUNT_ID`, `IMAGE_REGION` (us-east-1),
   `LAUNCH_REGION`, `AMI`, `SUBNET`, `INSTANCE_TYPE`,
   `INSTANCE_PROFILE`.
2. `USER_DATA` heredoc that:
   - Sets `trap cleanup EXIT` (uploads `/var/log/*.log` to S3,
     `shutdown -h +1`).
   - `aws ecr get-login-password | docker login` + `docker pull`.
   - `docker run --rm --gpus all ... --entrypoint /entrypoint-train.sh ...`
     (training) or
     `--entrypoint bash ... -lc 'cd /work/... && python scripts/eval.py ...'`
     (eval).
3. `aws ec2 run-instances` with
   `--instance-initiated-shutdown-behavior terminate`.

## Existing launchers

- `d10-l40s-eu.sh` — d10 cross-run resume on L40S in eu-central-1
  (uses `INIT_FROM_S3` + `START_EPOCH`).
- `d15-40x256-eu.sh` — Experiment A: 40-block ResNet on d15. Tests
  capacity-limited hypothesis.
- `d10-full30m.sh` — Experiment C: d10 on the full ~30M positions
  (no `MAX_POSITIONS`). Needs `g6e.8xlarge` (256 GB RAM).
- `eval-deep-sims.sh` — Experiment E: re-eval an existing ckpt with
  `--sims 4000`. Writes to a distinct filename
  (`eval_results-<ckpt>-sims4000.txt`) so the daemon's idempotency
  check still skips the slow path.

## Retry-until-quota recipe

```bash
while true; do
    for region in us-east-1 eu-central-1; do
        # sed in LAUNCH_REGION/AMI/SUBNET, run, check for i-... in output
    done
    sleep 120
done
```

**Why:** EKS gives spot resilience and a clean Job lifecycle, but our
32 vCPU per-region G/VT quota is shared with auto-eval EC2s. Bare EC2
in a second region is the simplest way around capacity contention.

**How to apply:** Need a one-off GPU run? If the EKS cluster's region
is quota-busy or doesn't have the instance type, copy an existing
launcher, edit the `docker run` env vars + `INSTANCE_TYPE`, and
`bash infra-eks/launchers/<name>.sh`. See
[`multi-region-capacity.md`](./multi-region-capacity.md) for the
cross-region wiring.
