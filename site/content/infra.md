---
title: "Infrastructure"
subtitle: "eks · bare-ec2 fallback · multi-region · self-terminating"
next: "/failures/"
---

## The Shape

Everything runs on AWS. The pipeline has three roles:

1. **Datagen** — EKS Indexed Job, 8 spot-CPU pods × N games each.
   Each pod writes a shard to S3; a second Job merges shards into a
   single `data.npz`. Cluster spec:
   [`cluster.yaml`](https://github.com/shehio/world-models/blob/main/infra-eks/cluster.yaml).
2. **Training** — EKS single-pod Job on a GPU node, or a bare-EC2
   self-terminating launcher when EKS can't get the instance type.
   Cluster spec: [`cluster-train-us.yaml`](https://github.com/shehio/world-models/blob/main/infra-eks/cluster-train-us.yaml).
3. **Eval + self-play** — per-checkpoint auto-eval EC2s
   ([`daemons/wm-autoeval-daemon.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/daemons/wm-autoeval-daemon.sh)),
   and a self-play loop in its own EKS cluster
   ([`cluster-selfplay-us.yaml`](https://github.com/shehio/world-models/blob/main/infra-eks/cluster-selfplay-us.yaml)).

Everything is committed and self-contained. Tear-down is one
`eksctl delete cluster -f <spec>` per cluster.

## The Key Constraint: G/VT vCPU Quota

AWS caps "G and VT instances" (the GPU families we use) at 32 vCPU
per region by default. Two concurrent g6.4xlarge evals (16 vCPU
each) saturate that. A g6e.8xlarge full-30M training (32 vCPU) won't
even start if any other G/VT instance is alive in the same region.

The pipeline works around this with **multi-region fallback**:

- The auto-eval daemon walks `REGION_ORDER=(us-east-1 eu-central-1)`.
  First successful launch wins.
- Cross-region wiring is invisible to the workload. ECR image lives
  in us-east-1; S3 buckets live in us-east-1; EC2 lives in whichever
  region has quota. `aws s3 cp --region us-east-1` and
  `aws ecr get-login-password --region us-east-1` work from any
  region.
- IAM is global. The instance profile
  `wm-chess-merge-instance-profile` works in every region.

Net effect: ~64 effective vCPU of G/VT capacity, no quota tickets
required.

## The EKS Pattern

Every cluster follows the same shape:

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata: { name: wm-chess-*, region: us-east-1, version: "1.31" }
iam:
  withOIDC: true
  serviceAccounts:
    - metadata: { name: wm-chess-train-sa, namespace: default }
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
managedNodeGroups:
  - name: gpu-*
    instanceTypes: [g6.*, g6e.*]
    spot: false   # on-demand for predictability under tight quota
    minSize: 0
    desiredCapacity: 1
    maxSize: 1
    volumeSize: 100   # bump to 300 if extracting the full 30M dataset
    labels: { role: gpu-* }
    # NO TAINT — see failures/ for what happens otherwise
```

Bring up: `eksctl create cluster -f <spec>` (~15-20 min).
Tear down: `eksctl delete cluster -f <spec>` (~10 min).

## The Bare-EC2 Pattern

When EKS can't get a specific instance type (e.g. g6e was out of
L40S capacity in our cluster's AZ), we bypass K8s with a bare-EC2
launcher. Each script:

1. Constants at top: `LAUNCH_REGION`, `AMI`, `SUBNET`,
   `INSTANCE_TYPE`, `INSTANCE_PROFILE`.
2. `USER_DATA` heredoc that `docker run`s the same image with the
   same entrypoint, then `shutdown -h +1` on EXIT.
3. `aws ec2 run-instances --instance-initiated-shutdown-behavior terminate`
   so the EC2 self-deletes when the workload finishes.

Catalog of existing launchers
([`infra-eks/launchers/`](https://github.com/shehio/world-models/tree/main/infra-eks/launchers)):

| script | what |
|---|---|
| `d10-l40s-eu.sh` | resume d10 training on L40S in eu-central-1 |
| `d15-40x256-eu.sh` | Experiment A (40×256 net on d15) |
| `d10-full30m.sh` | Experiment C (d10 on full ~30M positions, g6e.8xlarge) |
| `eval-deep-sims.sh` | Experiment E (sims=4000 eval) |

## Crash-Safety + Cross-Run Resume

`entrypoint-train.sh` honors two env vars for explicit cross-run
resume:

- `INIT_FROM_S3=s3://…/distilled_epochNNN.pt` — initial weights.
- `START_EPOCH=N` — number of epochs to skip in the loop. The RNG
  advances N times so the data permutation matches what the original
  run would have produced.

This means a spot reclaim, an L40S-shortage migration, or a manual
"resume after a fix" doesn't lose progress. Drop the latest ckpt
into `INIT_FROM_S3`, set `START_EPOCH` to one past it, and the next
run picks up under a fresh `RUN_ID` (so checkpoints land in a new
directory and the auto-eval daemon picks them up).

## Image Build

`infra-eks/Dockerfile.train` is a single image used by both
supervised training (`entrypoint-train.sh`) and self-play
(`entrypoint-selfplay.sh`). Base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9`.
Adds AWS CLI, Stockfish 17, and our editable packages via
`pip install --no-deps -e .` (the `--no-deps` is critical — without
it, pip reinstalls torch as a CPU wheel from PyPI and the CUDA image
becomes useless). Build via CodeBuild:

```bash
aws codebuild start-build --region us-east-1 \
    --project-name wm-chess-image \
    --buildspec-override infra-eks/buildspec-train.yml \
    --environment-variables-override \
        name=ECR_REPO,value=wm-chess-gpu \
        name=AWS_REGION,value=us-east-1
```
