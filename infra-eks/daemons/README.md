# Training-side daemons

Six long-running shell daemons that run on the operator's laptop (or any
machine with `aws` + `kubectl --context wm-train-us`) and provide
crash-safe checkpoints, per-checkpoint Elo measurement, eviction
recovery, and pipeline-stage handoffs.

All are idempotent — safe to restart, will not double-process work.

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

## `wm-d15-30m-handoff-daemon.sh`

One-shot handoff daemon for the d15 250K → d15-30M training pipeline.
Walks the three stages of the handoff so we don't have to babysit the
boundaries:

1. **Wait for datagen complete.** Polls
   `kubectl get job wm-chess-gen-d15-250k -o jsonpath='{.status.succeeded}'`
   until it hits `8`.
2. **Apply merge + wait for `merged/data.npz`.** Runs
   `bash infra-eks/launchers/gen-d15-250k.sh merge` to apply the merge
   K8s Job, then polls S3 for `s3://wm-chess-library-…/d15-mpv8-T1-g250000-20260519T0412Z/merged/data.npz`
   to appear (typically ~5 min after merge Job applied).
3. **Fire training.** Runs `bash infra-eks/launchers/d15-full30m.sh` —
   bare-EC2 g6e.8xlarge in us-east-1 against the merged dataset.

Idempotent via per-step state files under `/tmp/wm-d15-30m-handoff.state/`:
`datagen.done` → `merge.done` → `training.fired`. Restart the daemon and
it resumes at the next pending step.

To stop cleanly: `echo stop > /tmp/wm-d15-30m-handoff.stop`. To force a
full re-run after success: `rm -rf /tmp/wm-d15-30m-handoff.state`.

```
nohup bash infra-eks/daemons/wm-d15-30m-handoff-daemon.sh > /tmp/wm-d15-30m-handoff.log 2>&1 &
tail -F /tmp/wm-d15-30m-handoff.log
```

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

## `wm-deep-eval-daemon.sh`

Second-stage eval daemon. Polls S3 for the sims=800 eval results the
autoeval daemon writes; when a checkpoint's sims=800 Elo crosses
`ELO_THRESHOLD` (default 1950), it fires a one-shot sims=4,000 eval EC2
(g6.4xlarge, same multi-region fallback order) for a sharper
measurement. Rationale: sims=800 systematically under-reports strength
vs sims=4,000 (+277 Elo measured on d15 ep 20), so the cheap sims=800
curve is used to pick *which* ckpts deserve the expensive deep read.
Idempotent via an `eval_results-<ckpt>-sims4000.txt` sibling (permanent
skip) and a `.claimed-sims4000-<ckpt>` marker (no double-launch).

```
nohup bash infra-eks/daemons/wm-deep-eval-daemon.sh >/dev/null 2>&1 &
```

## `wm-d15-run1-watcher.sh`

Eviction watcher for the d15 Run 1 (40×256) spot training instance.
Polls the instance state every 5 min; if the spot is evicted, re-fires
`infra-eks/launchers/d15-full30m.sh`, which launches a fresh spot
g6e.8xlarge and auto-resumes from the latest S3 checkpoint (the
launcher pins `RUN_ID`). Tracks the current instance id in
`/tmp/wm-d15-run1.id`; if the instance is still running it does
nothing. Start it with the instance id to watch:

```
nohup bash infra-eks/daemons/wm-d15-run1-watcher.sh <instance-id> >/dev/null 2>&1 &
```

## `wm-go-gpu-handoff-daemon.sh`

Go-pipeline sibling of `wm-d15-30m-handoff-daemon.sh`. Walks the three
stages of the Go 9×9 GPU pipeline so the boundaries don't need
babysitting: (1) polls the `wm-go-gpu-9x9` datagen Job until all
completions succeed, (2) merges the per-pod
`shards_partial/pod-*/worker_*_chunk_*.npz` chunks into one
`merged/data.npz` on S3 using the Go-native merge (the chess streaming
`merge_chunks.py` expects a different on-disk layout), then (3) fires
the Go training launcher on the merged dataset. Idempotent and
crash-safe via per-step marker files under
`/tmp/wm-go-gpu-handoff.state`; `echo stop > /tmp/wm-go-gpu-handoff.stop`
to exit cleanly.

```
nohup bash infra-eks/daemons/wm-go-gpu-handoff-daemon.sh > /tmp/wm-go-gpu-handoff.log 2>&1 &
```
