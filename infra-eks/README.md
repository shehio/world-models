# EKS — Kubernetes-native parallel datagen + GPU training + auto-eval

Two pipelines, one cluster pattern:

1. **Datagen** (this file, top half) — Indexed-Job parallel data
   generation on spot CPU, merge as a second Job. Replaces the
   `infra/` Terraform module and `wm_chess/scripts/parallel_datagen.sh`.
2. **Training + eval** (this file, [training section](#gpu-training)) —
   GPU EKS Job for the long training run, **`daemons/`** for crash-safe
   per-checkpoint S3 sync + auto-launched per-checkpoint EC2 evals,
   **`launchers/`** for one-off bare-EC2 runs (L40S training, deep-sim
   ablations).

The datagen pipeline solves the two pain points of the raw-EC2 approach:

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

### Datagen pipeline

| File | Role |
|---|---|
| `cluster.yaml` | eksctl spec: control plane + spot-diversified nodegroup |
| `Dockerfile` | Container image (stockfish + uv + project) |
| `entrypoint.sh` | `gen` / `merge` mode dispatcher inside the container |
| `k8s/job-gen.yaml` | Indexed parallel Job (8 pods, one per shard) |
| `k8s/job-merge.yaml` | Single-pod Job that concatenates shards |
| `k8s/service-account.yaml` | SA bound to IAM role with S3 RW (created via eksctl) |
| `run.sh` | End-to-end launcher (build, push, apply, wait, pull) |
| `buildspec.yml` / `codebuild.json` | CodeBuild image-build pipeline |

### GPU training + eval pipeline

| File | Role |
|---|---|
| `cluster-train-us.yaml` / `cluster-train-eu.yaml` | eksctl specs for GPU clusters (1× g6.xlarge default + add-on nodegroups) |
| `nodegroup-fast-us.yaml` | `g6.4xlarge` (16 vCPU, L4) for in-RAM training of mid-size datasets |
| `nodegroup-l40s-us.yaml` / `nodegroup-l40s-b-us.yaml` | `g6e.xlarge` (L40S) nodegroups for the d10 long training |
| `Dockerfile.train` | GPU image (CUDA + uv + project + entrypoint-train.sh) |
| `entrypoint-train.sh` | Trainer entrypoint: honors `INIT_FROM_S3` / `START_EPOCH` for cross-run resume |
| `k8s/job-train.yaml` | GPU Job template (env-driven; see `run-train.sh`) |
| `run-train.sh` | CLI wrapper: `submit us|eu`, render Job, apply, tail logs |
| `buildspec-train.yml` | CodeBuild spec for the GPU image |
| `daemons/wm-autoeval-daemon.sh` | Polls S3 for new ckpts, launches per-checkpoint eval EC2s (multi-region) |
| `daemons/wm-pod-sync-daemon.sh` | Legacy: kubectl-exec'd S3 sync for old-image pods |
| `launchers/d10-l40s-eu.sh` | Bare-EC2 L40S launch in eu-central-1 (bypass EKS when capacity is short) |
| `launchers/d15-40x256-eu.sh` | Experiment A: 40-block ResNet on d15 |
| `launchers/d10-full30m.sh` | Experiment C: d10 on the full ~30M positions |
| `launchers/eval-deep-sims.sh` | Experiment E: re-eval a ckpt with `--sims 4000` |

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

## GPU training

The training side reuses the same cluster pattern with a GPU nodegroup
and `Dockerfile.train`.

```bash
# 1. provision a GPU cluster (us-east-1 default)
eksctl create cluster -f infra-eks/cluster-train-us.yaml

# 2. submit a training Job (env-driven — see run-train.sh for knobs)
BUCKET=wm-chess-library-... \
PREFIX=d15-mpv8-T1-g100000-... \
MAX_POSITIONS=5000000 \
IN_RAM=1 EPOCHS=20 SAVE_EVERY=5 \
N_BLOCKS=20 N_FILTERS=256 \
./infra-eks/run-train.sh submit us

# 3. each epoch the trainer syncs distilled_epochNNN.pt to S3.
#    Start the auto-eval daemon to pick those up and run eval EC2s.
nohup bash infra-eks/daemons/wm-autoeval-daemon.sh >/dev/null 2>&1 &

# 4. tear down when results land
eksctl delete cluster -f infra-eks/cluster-train-us.yaml
```

### Crash-safe checkpointing + cross-run resume

`entrypoint-train.sh` honors two env vars for explicit cross-run resume:

- `INIT_FROM_S3=s3://…/distilled_epochNNN.pt` — initial weights to load.
- `START_EPOCH=N` — number of epochs to skip in the new run's loop. The
  training loop's RNG is advanced N times so the data permutation
  matches what the original run would have produced.

This means a spot reclamation or an L40S-shortage migration doesn't lose
progress: drop the latest ckpt into `INIT_FROM_S3`, set `START_EPOCH` to
one past it, and the next run picks up the same trajectory under a fresh
`RUN_ID` (so checkpoints land in a new directory and the auto-eval
daemon picks them up).

### When to bypass EKS — `launchers/`

EKS gives spot resilience and a clean Job lifecycle, but the cluster
shares a 32 vCPU G/VT quota with the auto-eval EC2s. When we need an
L40S type that's out of capacity in our cluster region, or when we just
want a one-shot ablation run that doesn't justify a new Job submission,
we use the bare-EC2 launchers in `launchers/`. Each one is a single
`aws ec2 run-instances` with a user-data heredoc that pulls the same
image, runs the same entrypoint, and self-terminates via
`shutdown -h` + `instance-initiated-shutdown-behavior=terminate`.

See [`launchers/README.md`](./launchers/README.md) for the full list and
the retry-until-quota pattern.
