# Training-side daemons

Two long-running shell daemons that run on the operator's laptop (or any
machine with `aws` + `kubectl --context wm-train-us`) and provide
crash-safe checkpoints + per-checkpoint Elo measurement.

Both are idempotent — safe to restart, will not double-process work.

## `wm-pod-sync-daemon.sh`

For training pods running an OLD image (built before per-checkpoint S3
sync was added to `train.py`). Periodically `kubectl exec`s into the
pod and `aws s3 sync`s `/work/checkpoints/` to the indexed S3 path.

The label filter is `job-name=wm-chess-train` so it does NOT touch the
d10 pod (whose newer image syncs itself). Exits cleanly when the d15
pod is gone.

```
nohup bash infra-eks/daemons/wm-pod-sync-daemon.sh >/dev/null 2>&1 &
```

Not needed for any run built from `a3c6576` or later — `train.py`
syncs each checkpoint to S3 inline.

## `wm-autoeval-daemon.sh`

Polls every S3 bucket in `BUCKETS` for new `distilled_epoch*.pt` files.
For each one without an `eval_results-<ckpt>.txt` sibling, launches a
one-shot g6.xlarge EC2 in us-east-1 that:

1. Pulls the `wm-chess-gpu` image from ECR
2. Downloads the `.pt` from S3
3. Runs `eval.py` vs Stockfish UCI=1350 (100 games) and depth=1 (100 games)
4. Uploads `eval_results-<ckpt>.txt` and `.log` next to the checkpoint
5. Calls `shutdown -h +1` — the instance self-terminates via
   `instance-initiated-shutdown-behavior=terminate`

Idempotency: writes `.claimed-eval-<ckpt>` markers in S3 before
launching, so concurrent pollers won't double-launch.

```
nohup bash infra-eks/daemons/wm-autoeval-daemon.sh >/dev/null 2>&1 &
```

### Requirements

- AMI: `Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)`
  — comes with docker + nvidia-container-toolkit pre-installed
- Instance profile: `wm-chess-merge-instance-profile`, attached role
  `wm-chess-merge-instance-role` with the `wm-chess-eval-s3` inline
  policy (GetObject + PutObject on `**/checkpoints/**`)
- Subnet: `subnet-042fb3c497e2631a7` (us-east-1 default VPC public subnet)
