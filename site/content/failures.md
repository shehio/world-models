---
title: "Failure modes"
subtitle: "nine real bugs we hit, with the fixes"
---

Reading this catalog before a fresh training run is much faster than
rediscovering each one in real time. Each entry: **symptom**, **root
cause**, **fix**.

## 1 — NPZ Extraction Filled the Volume

**Symptom.** Training pod's epoch 1 stalled with no progress. The
container's `/work-tmp` filled up; the OS killed extraction silently.

**Root cause.** `MultipvDataset` extracts the `.npz` into a flat
directory of `.npy` files for memmap. The d10 dataset (~30M positions)
extracts to ~145 GB. The default training volume was 100–150 GB.

**Fix.** `--max-positions N` truncation on the extractor (env var
`MAX_POSITIONS`). 5M → ~25 GB. Bumped the bare-EC2 volume to 300 GB
for the full 30M run.

## 2 — Bare-EC2 Launcher Used the Wrong Entrypoint Path

**Symptom.** d10-l40s launcher v1:

```
docker: ... exec: "/work/infra-eks/entrypoint-train.sh":
stat /work/infra-eks/entrypoint-train.sh: no such file or directory
```

EC2 self-terminated within minutes; no checkpoints produced.

**Root cause.** Wrong path. The image installs the entrypoint at
`/entrypoint-train.sh`, not under `/work/`.

**Fix.** All launchers in `infra-eks/launchers/` now use the
in-container path. `bash -n <launcher.sh>` lint runs before commit.

## 3 — SAVE_EVERY=5 Cost Epochs 6–9 on Every Reclaim

**Symptom.** A reclaim or manual termination at epoch 9 of a 20-epoch
run loses ckpts 6/7/8/9 even though they were computed; only ep 5 is
in S3.

**Root cause.** `--save-every 5` writes ckpts at epochs 5, 10, 15, 20.
Any interruption between save points is wasted compute.

**Fix.** For long runs, set `SAVE_EVERY=1`. Each save is ~1 GB to S3,
~30 s — essentially free at this scale.

## 4 — Eval EC2 `docker run --gpus all` Failed on Plain AL2023

**Symptom.**

```
docker: Error response from daemon: could not select device driver
"" with capabilities: [[gpu]]
```

**Root cause.** The default AL2023 AMI doesn't ship with
`nvidia-container-toolkit` configured for `--gpus all`.

**Fix.** Use the **Deep Learning Base OSS Nvidia Driver GPU AMI**:

- us-east-1: `ami-027c3ae8019fc0d3a`
- eu-central-1: `ami-01e9d13d4c5e54237`

Wired into the auto-eval daemon's `region_config`.

## 5 — Stockfish UCI_Elo=1320 Rejected

**Symptom.** Eval pod died with `value 1320 less than minimum 1350`.
The first auto-eval after upgrading to Stockfish 17 broke.

**Root cause.** Stockfish 17 bumped the `UCI_Elo` minimum from 1320
to 1350.

**Fix.** Daemon default is now 1,350. Old 1,185-Elo 02b numbers are
normalized in the experiment's `results.md`.

## 6 — Repeated VcpuLimitExceeded in us-east-1

**Symptom.** Two concurrent eval EC2s + one training EC2 → next
launch fails with the 32 vCPU G/VT quota message.

**Root cause.** AWS quota L-DB2E81BA caps us-east G/VT at 32 vCPU.

**Fix.** Daemon walks `REGION_ORDER=(us-east-1 eu-central-1)` per
launch. Cross-region pull/read just works. See [infra](/infra/).

## 7 — EKS GPU Taint Broke CoreDNS

**Symptom.** Training pod couldn't reach
`sts.us-east-1.amazonaws.com` → `aws s3 cp` failed with
`Could not connect to the endpoint URL`. Job hit
`BackoffLimitExceeded`.

**Root cause.** The default eksctl GPU node spec adds
`nvidia.com/gpu:NoSchedule`. On a single-node cluster, that taints
the only node; system pods (CoreDNS, metrics-server) don't tolerate
the taint and stay `Pending` forever. With CoreDNS unscheduled, no
DNS resolution.

**Fix.** Removed the taint from both `cluster-train-{us,eu}.yaml`.
Single-purpose clusters don't need it.

## 8 — Finalize / Merge OOM on Small Instances

**Symptom.** Salvage pods OOM-killed silently while finalizing
per-pod data.

**Root cause.** `finalize_library_path` and `merge_shards.py` both:

```python
arrays = []
for cf in chunks: arrays.append(np.load(cf)[k])
np.concatenate(arrays, axis=0)
```

Holds every chunk in RAM, then `np.concatenate` doubles peak. For
the d10 pods, ~5 GB raw → ~11 GB peak → exceeds the 8 GB on an
`m6i.large`.

**Fix.** `wm_chess/scripts/merge_chunks.py` — a streaming script
that bypasses both finalize and merge. For each array key, it opens
one zip entry, writes the npy header, then streams each chunk's
`tobytes()` directly into the deflate stream. Peak RAM ~100 MB
regardless of dataset size.

## 9 — Depth=1 Secondary Eval Was Unreliable

**Symptom.** The "depth=1 top-skill" secondary eval gave wildly
different W/D/L across nearby checkpoints — d15 ep 15 was 8/8/88,
ep 20 was 93/11/0. Inconsistent with the otherwise-monotonic UCI=1350
trajectory.

**Root cause.** depth=1 produces drawish positional games where the
result depends on whether the agent finds a tactical win the depth-1
Stockfish blunders into. High variance, no Elo calibration.

**Fix.** Replaced with **UCI=1800**, a calibrated opponent near
current model strength. CI is now tightest where the model is
well-matched.

## Takeaways for the Next Training Run

1. **Size the volume for the dataset, not the ckpts.** 100 GB is fine
   for 5M positions; bump to 300 GB for full-30M.
2. **Set `SAVE_EVERY=1`** on overnight runs unless you have a reason
   not to.
3. **Use the DL Base GPU AMI**, not plain AL2023.
4. **Parameterize `LAUNCH_REGION` / `AMI` / `SUBNET`** so retry-on-
   quota across regions is a one-line sed change.
5. **Don't add `nvidia.com/gpu:NoSchedule`** to single-purpose
   clusters.
6. **Pick a calibrated sub-test opponent**, not `depth=1`.
