# Bare-EC2 launchers

One-shot scripts that bring up a single EC2, run a job, and self-terminate via
`shutdown -h` + `instance-initiated-shutdown-behavior=terminate`. Used when EKS
is unavailable for the capacity we need (e.g. an L40S type with no quota in our
cluster region) or when a one-off doesn't justify a Job submission.

## Pattern

All launchers share the same shape:

1. Constants at top: `ACCOUNT_ID`, `IMAGE_REGION`, `LAUNCH_REGION`, `AMI`,
   `SUBNET`, `INSTANCE_TYPE`, `INSTANCE_PROFILE`.
2. A `USER_DATA` heredoc that:
   - traps EXIT to upload the host log to S3 and `shutdown -h +1`,
   - logs in to the us-east-1 ECR, pulls `wm-chess-gpu:latest`,
   - `docker run --rm --gpus all` with the appropriate entrypoint
     (`/entrypoint-train.sh` for training, `bash -lc '…eval.py…'` for eval).
3. `aws ec2 run-instances` with `--instance-initiated-shutdown-behavior terminate`
   so the EC2 disappears when the workload finishes.

Cross-region wiring works because:
- The ECR image lives in `us-east-1` and is cross-region-pulled.
- The S3 bucket (`wm-chess-library-…`) lives in `us-east-1` and is cross-region
  read/written.
- The IAM instance profile is global.

So the LAUNCH_REGION can move freely (us-east-1, eu-central-1) to dodge G/VT
vCPU quota in any single region.

## Launchers

| Script | What it does |
| --- | --- |
| `d10-l40s-eu.sh` | Resume d10 training from `distilled_epoch004` (ep 5) on an L40S in eu-central-1. Bypasses EKS when us-east has no L40S capacity. |
| `d15-40x256-eu.sh` | Experiment A: train a 40-block ResNet on d15 — tests whether the ~1800 Elo plateau is capacity-limited. |
| `d10-full30m.sh` | Experiment C: train d10 on the FULL ~30M positions (vs 5M subsample) — tests whether the plateau is dataset-size-limited. Needs g6e.8xlarge (256 GB RAM). |
| `eval-deep-sims.sh` | Experiment E: re-eval d15 ep 20 with `--sims 4000` (vs daily 800) — tests whether the model is under-read by routine eval depth. |

## Retry-until-quota pattern

These launchers fail loudly if G/VT vCPU quota is exhausted. A common retry
loop:

```bash
# Retry the C launch every 2 min until quota frees.
while true; do
    out=$(bash infra-eks/launchers/d10-full30m.sh 2>&1)
    if echo "$out" | grep -qE '^i-[0-9a-f]+'; then echo "launched: $out"; break; fi
    sleep 120
done
```

For multi-region retry, override `LAUNCH_REGION`/`AMI`/`SUBNET` per attempt
(see `daemons/wm-autoeval-daemon.sh` for the daemon-side version with region
fallback baked in).

## Logs and results

Each launcher writes:
- `/var/log/{train,eval}.log` on the EC2 (mirror of stdout/stderr).
- An EXIT trap copies that log to S3 — for training, to
  `s3://<bucket>/<prefix>/checkpoints/net-<NxF>/<RUN_ID>/host-launch.log`;
  for eval, to `s3://<…>/eval_results-<ckpt>.log` next to the result.

Look there first when an experiment "didn't produce a checkpoint" — the EC2
has already terminated.
