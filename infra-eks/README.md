# EKS — Kubernetes-native parallel datagen + merge

Replaces the `infra/` Terraform module and the bash orchestrator at
`wm_chess/scripts/parallel_datagen.sh`. Solves the two pain points
of the raw-EC2 approach:

1. **Spot reclamation resilience.** The managed node group spans
   six instance types (`c7i.8xlarge`, `c6i.8xlarge`, `c7a.8xlarge`,
   `c5.9xlarge`, `m7i.8xlarge`, `m6i.8xlarge`). The K8s Job
   controller restarts reclaimed pods with the same completion index
   automatically (up to `backoffLimit: 24`). No bespoke retry loops.

2. **No `ssh` / `tmux` / laptop in the loop.** Pods are container-
   native, scheduled by K8s, communicate via S3 for shard staging.
   The merge step runs as a second Job on the same cluster; the
   laptop only pulls the final merged dataset.

## Topology

```
laptop                              EKS cluster (wm-chess)
─────                               ─────────────────────────
                                    ┌──────────────────────┐
eksctl apply ────────► CloudFormation │  control plane     │
                                      │  ($0.10/h)         │
docker build + push ──► ECR           └──────────┬─────────┘
                                                 │ schedules
                                    ┌────────────▼─────────┐
                                    │ managed nodegroup    │
                                    │ workers-spot         │
                                    │ • 0..8 nodes         │
                                    │ • 6 instance types   │
                                    │   (spot diversified) │
                                    │ • 32 vCPU each       │
                                    └──────────┬───────────┘
                                               │ pods (8 indexed)
                                    ┌──────────▼───────────┐
                                    │  wm-chess-gen Job    │
                                    │  parallelism=8       │
                                    │  completionMode=     │
                                    │    Indexed           │
                                    │  each pod writes one │
                                    │  shard to S3         │
                                    └──────────┬───────────┘
                                               │ shards
                                    ┌──────────▼───────────┐
                                    │  S3: wm-chess-library│
                                    │  /<prefix>/shards/   │
                                    │    pod-0/ pod-1/ ... │
                                    └──────────┬───────────┘
                                               │ on gen complete
                                    ┌──────────▼───────────┐
                                    │  wm-chess-merge Job  │
                                    │  parallelism=1       │
                                    │  reads 8 shards,     │
                                    │  runs merge_shards.py│
                                    └──────────┬───────────┘
                                               │ merged dataset
                                    ┌──────────▼───────────┐
                                    │  S3: wm-chess-library│
                                    │  /<prefix>/merged/   │
                                    │     data.npz         │
                                    │     games.pgn        │
                                    │     metadata.json    │
                                    └──────────┬───────────┘
                                               │ aws s3 sync
laptop ←─────────────────────────────────────────┘
  library/games/sf-18/d15-mpv8-T1/g100000-merged-<ts>/data.npz
```

## Files

| File | Role |
|---|---|
| `cluster.yaml` | eksctl spec: control plane + spot-diversified nodegroup |
| `Dockerfile` | Container image (stockfish + uv + project) |
| `entrypoint.sh` | `gen` / `merge` mode dispatcher inside the container |
| `k8s/job-gen.yaml` | Indexed parallel Job (8 pods, one per shard) |
| `k8s/job-merge.yaml` | Single-pod Job that concatenates shards |
| `k8s/service-account.yaml` | SA bound to IAM role with S3 RW (created via eksctl) |
| `run.sh` | End-to-end launcher (build, push, apply, wait, pull) |

## Cost (vs raw EC2)

| Component | Cost/hour | Notes |
|---|---:|---|
| EKS control plane | **$0.10** | always-on while cluster exists |
| Worker nodes (8 × c7i-class spot) | ~$2.72 | same as raw EC2 |
| S3 storage | < $0.01 | ~10 GB shards × $0.023/GB-month |
| ECR storage | < $0.01 | <1 GB image × $0.10/GB-month |
| Data transfer S3 ↔ EC2 | $0 | same region |
| Laptop S3 download | one-time | ~$0.09/GB egress |

For a single 12-hour run: ~$34 vs ~$32 raw EC2. **$2 / 6 % premium**
for the durability of `parallelism=8 backoffLimit=24` retries against
spot reclamation. Worth it.

## Usage

```bash
# 0. one-time per laptop
brew install eksctl kubectl
# (also need docker daemon running)

# 1. provision (~15-20 min)
eksctl create cluster -f infra-eks/cluster.yaml

# 2. fire one run end-to-end
bash infra-eks/run.sh

# 3. when totally done, tear down to stop the $0.10/h control-plane bleed
eksctl delete cluster -f infra-eks/cluster.yaml
# (or: TEARDOWN_AFTER=1 bash infra-eks/run.sh)
```

Each `run.sh` invocation:
1. Builds the container image, pushes to ECR.
2. Ensures S3 bucket + IAM service-account exist.
3. Renders the Job templates with the image / bucket / prefix.
4. `kubectl apply` the gen Job, `kubectl wait` for completion.
5. `kubectl apply` the merge Job, `kubectl wait`.
6. `aws s3 sync` the merged dataset to `library/games/.../g100000-merged-<ts>/`.
7. Optionally `eksctl delete cluster`.

## Why this beats `parallel_datagen.sh`

| Failure class | bash orchestrator (old) | EKS Job (new) |
|---|---|---|
| Spot reclamation mid-run | Pod is gone; lost up to chunk_size games | Pod gets re-scheduled with the same completion index, retries from scratch (so each retry might cost the chunk_size games it had, but the *shard* completes) |
| AZ capacity exhaustion | Bash loops through types in series, all in one AZ | Managed nodegroup spans 6 instance types; K8s scheduler picks any AZ |
| Laptop sleeps / wifi dies | tmux on remote keeps gen alive, but no orchestration | K8s runs entirely in AWS; laptop only used for `kubectl wait` and final download |
| Bash bug (e.g., `ssh` reading stdin) | Yes, I shipped two of these today | Doesn't exist — k8s Job spec is declarative |
| Observability | `tail -F /tmp/parallel.log` | `kubectl logs job/wm-chess-gen --tail=100 -f --all-containers` plus standard K8s tooling (CloudWatch via the agent, Lens, etc.) |
