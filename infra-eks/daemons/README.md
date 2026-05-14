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
one-shot **g6.4xlarge** EC2 that:

1. Pulls the `wm-chess-gpu` image from ECR (us-east-1)
2. Downloads the `.pt` from S3 (us-east-1)
3. Runs `eval.py` vs Stockfish **UCI=1350** (~100 games, 8 workers)
   and **UCI=1800** (~100 games, calibrated opponent near current model
   strength)
4. Uploads `eval_results-<ckpt>.txt` and `.log` next to the checkpoint
5. Calls `shutdown -h +1` — the instance self-terminates via
   `instance-initiated-shutdown-behavior=terminate`

The daemon parses the architecture (`net-NxF`) out of the checkpoint
path and passes the right `--n-blocks` / `--n-filters` to `eval.py`, so
20×256 and 40×256 checkpoints both work transparently.

### Multi-region fallback

`REGION_ORDER=(us-east-1 eu-central-1)`. When us-east-1 hits the 32 vCPU
G/VT quota, the daemon retries the same launch in eu-central-1. The
ECR image and S3 buckets stay in us-east-1; cross-region pull/read just
works. This roughly doubles concurrent eval throughput.

### Idempotency

- **`.claimed-eval-<ckpt>`** marker — written *before* launch so two
  concurrent pollers won't double-launch. Removed automatically if every
  region fails to launch.
- **`eval_results-<ckpt>.txt`** sibling — the final result file. If it
  exists, the checkpoint is skipped permanently. Per-experiment one-off
  evals use a distinct filename (e.g. `eval_results-<ckpt>-sims4000.txt`)
  so they don't collide with the daemon's check.

```
nohup bash infra-eks/daemons/wm-autoeval-daemon.sh >/dev/null 2>&1 &
```

### Requirements

- **AMI**: *Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)*
  — comes with docker + nvidia-container-toolkit pre-installed.
  IDs are wired per-region in `region_config`:
  `us-east-1=ami-027c3ae8019fc0d3a`, `eu-central-1=ami-01e9d13d4c5e54237`.
- **Instance profile**: `wm-chess-merge-instance-profile`. The attached
  role needs S3 GetObject + PutObject on the library bucket and
  ECR-read on `wm-chess-gpu`.
- **Subnets**: `subnet-042fb3c497e2631a7` (us-east-1a default VPC
  public subnet), `subnet-0a4bed60` (eu-central-1a default).
