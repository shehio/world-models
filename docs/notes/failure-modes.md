# Failure modes hit during the May 2026 training week

Concrete failures from the d10 / d15 training week, with the symptom,
root cause, and fix. Reading this catalog before a new training run
saves the 30-45 min it took to diagnose each one in real time.

## 1. NPZ extraction filled the 150 GB EBS volume

**Symptom:** d10 training pod's epoch-0 stalled with no progress. The
container's `/work-tmp` filled up; the OS killed the extraction process
silently.

**Root cause:** `MultipvDataset` extracts the `.npz` into a flat
directory of `.npy` files for memmap. The d10 dataset has ~30 M
positions → `states.npy` alone is ~145 GB. The default training
volume was 100-150 GB.

**Fix (two parts):**

- Added `--max-positions N` truncation to the extractor (`MAX_POSITIONS`
  env var on the entrypoint). 5 M positions → ~25 GB extracted.
- Bumped the volume to 300 GB on the bare-EC2 launcher for the full
  30 M run (`d10-full30m.sh`) and 250 GB on `cluster-train-eu.yaml`.

See [`training-dataloader-memmap.md`](./training-dataloader-memmap.md).

## 2. Bare-EC2 launcher referenced an in-container path that didn't exist

**Symptom:** d10-l40s launcher v1 logged

```
docker: ... exec: "/work/infra-eks/entrypoint-train.sh":
stat /work/infra-eks/entrypoint-train.sh: no such file or directory
```

The EC2 self-terminated within minutes; no checkpoints produced.

**Root cause:** Wrong path in `--entrypoint`. The image's Dockerfile
installs the entrypoint at `/entrypoint-train.sh`, not under `/work/`.
The launcher's user-data pointed at the wrong place.

**Fix:** All launchers in [`infra-eks/launchers/`](../../infra-eks/launchers/)
now use `--entrypoint /entrypoint-train.sh` (or
`/entrypoint-selfplay.sh`). Verified by running
`bash -n` on each launcher before commit.

## 3. SAVE_EVERY=5 cost epochs 5-8 on every reclaim

**Symptom:** A reclaim or manual termination at epoch 8 of a 20-epoch
run loses ckpts 5/6/7/8 even though they were computed; only the ep4
ckpt is in S3.

**Root cause:** `--save-every 5` writes ckpts at epochs 4, 9, 14, 19.
Any interruption between save points is wasted compute.

**Workaround:** For long runs (overnight, full 30M positions),
`SAVE_EVERY=1` is cheap (each save is ~1 GB to S3, ~30 s) and turns
"lost work between reclaims" from up to 5 epochs into at most 1.

**Status:** Not yet patched as a default — the auto-eval daemon's
idempotency check runs per-epoch-ckpt either way. Worth changing the
default in `entrypoint-train.sh` if more long runs are coming.

## 4. eval EC2 `docker run --gpus all` failed on plain AL2023

**Symptom:**

```
docker: Error response from daemon: could not select device driver
"" with capabilities: [[gpu]]
```

eval EC2 self-terminated before running anything.

**Root cause:** The default AL2023 AMI doesn't ship with
`nvidia-container-toolkit` configured for `--gpus all`. Plain docker
+ NVIDIA driver isn't enough.

**Fix:** Use the **Deep Learning Base OSS Nvidia Driver GPU AMI
(Amazon Linux 2023)** — comes pre-configured. IDs are wired in
[`wm-autoeval-daemon.sh`](../../infra-eks/daemons/wm-autoeval-daemon.sh)
per region:

- us-east-1: `ami-027c3ae8019fc0d3a`
- eu-central-1: `ami-01e9d13d4c5e54237`

## 5. Stockfish `UCI_Elo=1320` rejected

**Symptom:** Eval pod died with

```
RuntimeError: invalid argument UCI_Elo: value 1320 less than minimum 1350
```

The first auto-eval after upgrading to Stockfish 17 in the image
broke; older code targeted 1320 (02b's anchor).

**Root cause:** Stockfish 17 bumped the `UCI_Elo` minimum from 1320 to
1350.

**Fix:** Daemon's `STOCKFISH_ELO` default is now 1350. Old 1185-Elo
02b numbers are normalized in
[`distill-soft/results.md`](../../experiments/distill-soft/results.md).

## 6. Repeated `VcpuLimitExceeded` in us-east-1

**Symptom:** Concurrent eval EC2s + training EC2 in us-east-1 → next
launch fails with the 32 vCPU G/VT quota message.

**Root cause:** AWS quota L-DB2E81BA caps us-east G/VT usage at 32
vCPU. Three g6.4xlarge evals (48 vCPU) saturate it.

**Fix:** Daemon now walks
`REGION_ORDER=(us-east-1 eu-central-1)` per launch — first success
wins. Cross-region ECR pull + S3 read just work; see
[`multi-region-capacity.md`](./multi-region-capacity.md).

## 7. EKS GPU taint broke CoreDNS → no DNS in training pod

Documented in detail in
[`eks-gpu-taint-breaks-coredns.md`](./eks-gpu-taint-breaks-coredns.md).
TL;DR: single-node GPU cluster + the default `nvidia.com/gpu:NoSchedule`
taint leaves CoreDNS unscheduled, so the training pod can't resolve
STS / S3 endpoints. Removed the taint from both `cluster-train-*.yaml`.

## 8. `finalize_library_path` / `merge_shards.py` OOM on small instances

Documented in detail in
[`finalize-merge-oom.md`](./finalize-merge-oom.md). TL;DR: both
materialize all chunks in RAM, which doesn't fit on `m6i.large`.
Workaround: streaming `wm_chess/scripts/merge_chunks.py`.

## 9. eval `--stockfish-elo -1 --stockfish-depth 1` was unreliable

**Symptom:** The "secondary" eval (depth=1 top-skill Stockfish) gave
wildly different W/D/L distributions across nearby checkpoints —
ep14 of d15 was 8/8/88, ep19 was 93/11/0. Inconsistent with the
otherwise-monotonic UCI=1350 trajectory.

**Root cause:** depth=1 produces many drawn positional games where the
result depends on whether the agent finds a tactical win the depth-1
Stockfish blunders into. High variance, no Elo calibration.

**Fix:** Replaced the secondary sub-test with **UCI=1800**, a
calibrated opponent near current model strength. CI is now tightest
where the model is well-matched. See
[`EVALS.md`](../../EVALS.md#the-two-sub-tests-the-daemon-runs).

## Takeaways for the next training run

1. **Size the volume for the dataset, not the ckpts.** 100 GB is fine
   for 5 M positions; bump to 300 GB for full-30M.
2. **Set `SAVE_EVERY=1`** on overnight runs unless you have a reason
   not to. Disk and S3 are essentially free at this scale.
3. **Use the DL Base GPU AMI** for any new bare-EC2 launcher, not
   plain AL2023.
4. **Parameterize `LAUNCH_REGION` / `AMI` / `SUBNET`** so retry-on-
   quota across regions is a one-line sed change.
5. **Don't add `nvidia.com/gpu:NoSchedule` to single-purpose
   clusters.**
6. **Pick a calibrated sub-test opponent**, not `depth=1`. Bump the
   anchor as the model gets stronger.
