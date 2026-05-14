# Training infrastructure layout

GPU training pipeline on EKS for wm-chess — image, clusters, Jobs, the
bugs we hit on first deploy and how they were fixed.

The training side lives entirely under `infra-eks/`:

- **`Dockerfile.train`** — `pytorch/pytorch:2.5.1-cuda12.1` base + AWS
  CLI + our editable packages (`pip install --no-deps -e` so the bundled
  CUDA torch isn't replaced by a CPU wheel). No stockfish.
- **`entrypoint-train.sh`** — fetch `s3://.../merged/data.npz`, run
  `train.py`, push checkpoints. Includes resume-from-ckpt logic that
  looks under the indexed S3 path, plus an `INIT_FROM_S3` /
  `START_EPOCH` cross-run resume path.
- **`cluster-train-us.yaml`** / **`cluster-train-eu.yaml`** — one EKS
  cluster per region. Both have IRSA service account
  `wm-chess-train-sa` with `AmazonS3FullAccess`. **No node taint**
  (originally had `nvidia.com/gpu:NoSchedule` — broke CoreDNS in
  single-node clusters; see
  [`eks-gpu-taint-breaks-coredns.md`](./eks-gpu-taint-breaks-coredns.md)).
- **`k8s/job-train.yaml`** — single-pod Job, `nvidia.com/gpu: 1`
  resource limit, `backoffLimit: 0` during debug (keeps failing pods
  around for inspection), `restartPolicy: Never`.
- **`buildspec-train.yml`** — CodeBuild spec; pushes to `wm-chess-gpu`
  ECR repo. ECR replication mirrors us-east-1 → eu-west-1
  automatically (PREFIX_MATCH `wm-chess`).
- **`run-train.sh`** — orchestrator: `up | image | submit | logs |
  status | down  us|eu`.

**Indexed checkpoint path** (mirrors `library/games/` tree):

```
s3://<bucket>/<run-prefix>/checkpoints/net-<R>x<F>/<run-id>/
  distilled_epoch{NNN}.pt
  run_metadata.json
  train_history.json
```

`RUN_ID` is generated in `entrypoint-train.sh` as a UTC timestamp.

## Region-specific gotchas

- **us-east-1**: spot G/VT capacity for g6/g5/g4dn was unfulfillable
  repeatedly. Now uses **on-demand** (32 vCPU OD G/VT quota; g6.xlarge
  = 4 vCPU = fine).
- **eu-west-1**: g6 NOT offered; on-demand G/VT quota is 0. **Must use
  spot.** Pool: g5.xlarge + g4dn.xlarge.
- **eu-central-1**: used as a quota-pool partner with us-east-1 for
  bare-EC2 launches — see
  [`multi-region-capacity.md`](./multi-region-capacity.md).

## Required volumeSize for the GPU nodegroups

- us-east: 100 GB (us-east d15 dataset extracts to ~25 GB).
- eu: **250 GB** (eu d10 dataset extracts to ~145 GB).
- For runs that extract the full 30M-position dataset (Experiment C),
  bump to **300 GB**.

**Why:** The training pipeline took ~half a session of iteration to
get working. Saving this so the next time GPU training is needed, all
the lessons are in one place.

**How to apply:** Next training run, start by sourcing `.env` then
`./infra-eks/run-train.sh up <region>`, `image <region>`,
`submit <region>`. The four common failures are documented in
[`eks-gpu-taint-breaks-coredns.md`](./eks-gpu-taint-breaks-coredns.md)
(CoreDNS taint),
[`training-dataloader-memmap.md`](./training-dataloader-memmap.md)
(dataset-too-big OOM), and the `run-train.sh` script comments
(`envsubst ${VAR:-default}` doesn't work, kubectl context must be
explicit). See [`world-models-datagen-state.md`](./world-models-datagen-state.md)
for the broader datagen → merge → train pipeline.
