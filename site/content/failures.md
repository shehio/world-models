---
title: "Failure modes"
subtitle: "every real bug we hit, with the fixes"
---

Reading this catalog before a fresh training run is much faster than
rediscovering each one in real time. Each entry: **symptom**, **root
cause**, **fix**.

## 1: NPZ Extraction Filled the Volume

**Symptom.** Training pod's epoch 1 stalled with no progress. The
container's `/work-tmp` filled up; the OS killed extraction silently.

**Root cause.** `MultipvDataset` extracts the `.npz` into a flat
directory of `.npy` files for memmap. The d10 dataset (~30M positions)
extracts to ~145 GB. The default training volume was 100–150 GB.

**Fix.** `--max-positions N` truncation on the extractor (env var
`MAX_POSITIONS`). 5M → ~25 GB. Bumped the bare-EC2 volume to 300 GB
for the full 30M run.

## 2: Bare-EC2 Launcher Used the Wrong Entrypoint Path

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

## 3: SAVE_EVERY=5 Cost Epochs 6–9 on Every Reclaim

**Symptom.** A reclaim or manual termination at epoch 9 of a 20-epoch
run loses ckpts 6/7/8/9 even though they were computed; only ep 5 is
in S3.

**Root cause.** `--save-every 5` writes ckpts at epochs 5, 10, 15, 20.
Any interruption between save points is wasted compute.

**Fix.** For long runs, set `SAVE_EVERY=1`. Each save is ~1 GB to S3,
~30 s, essentially free at this scale.

## 4: Eval EC2 `docker run --gpus all` Failed on Plain AL2023

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

## 5: Stockfish UCI_Elo=1320 Rejected

**Symptom.** Eval pod died with `value 1320 less than minimum 1350`.
An auto-eval crashed: the image's Stockfish enforced a 1350 `UCI_Elo`
minimum while the eval code targeted 1320 (02b's anchor).

**Root cause.** A 1350 minimum is the *historical* floor. Stockfish
actually **lowered** the minimum from 1350 → 1320 in 2023 (PR #4341,
first shipped in SF16), so the crashing binary was SF ≤ 15.1 (the
version `apt` ships on Debian/Ubuntu stable), not the "17" we'd
assumed. We originally recorded this backwards (as "SF17 raised it").

**Fix.** Daemon default is now 1,350, safe across every version. Old
1,185-Elo 02b numbers are normalized in the experiment's `results.md`.

## 6: Repeated VcpuLimitExceeded in us-east-1

**Symptom.** Two concurrent eval EC2s + one training EC2 → next
launch fails with the 32 vCPU G/VT quota message.

**Root cause.** AWS quota L-DB2E81BA caps us-east G/VT at 32 vCPU.

**Fix.** Daemon walks `REGION_ORDER=(us-east-1 eu-central-1)` per
launch. Cross-region pull/read just works. See [infra](/infra/).

## 7: EKS GPU Taint Broke CoreDNS

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

## 8: Finalize / Merge OOM on Small Instances

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

**Fix.** `wm_chess/scripts/merge_chunks.py`, a streaming script
that bypasses both finalize and merge. For each array key, it opens
one zip entry, writes the npy header, then streams each chunk's
`tobytes()` directly into the deflate stream. Peak RAM ~100 MB
regardless of dataset size.

## 9: Depth=1 Secondary Eval Was Unreliable

**Symptom.** The "depth=1 top-skill" secondary eval gave wildly
different W/D/L across nearby checkpoints: d15 ep 15 was 8/8/88,
ep 20 was 93/11/0. Inconsistent with the otherwise-monotonic UCI=1350
trajectory.

**Root cause.** depth=1 produces drawish positional games where the
result depends on whether the agent finds a tactical win the depth-1
Stockfish blunders into. High variance, no Elo calibration.

**Fix.** Replaced with **UCI=1800**, a calibrated opponent near
current model strength. CI is now tightest where the model is
well-matched.

## 10: d15 R1 (40×256) Plateaued at Constant LR=1e-3

**Symptom.** R1's `train_history.json` showed loss / top-1 / top-K
essentially frozen across epochs 7–13: loss moved 0.0013 across 6
epochs, top-1 changed by 0.4 pp. sims=800 evals confirmed: ckpts
ep 7–11 all sat in the 1,864–1,910 band, no upward trend after the
ep 6 spike to 1,941.

**Root cause.** Bigger net (50M params) + constant Adam LR=1e-3 +
no decay → optimizer can't *settle* into a precise minimum, just
oscillates around one. The d10 20×256 run with the same constant LR
got further only because the smaller network's loss surface
tolerates a noisier optimizer.

**Fix.** Added `--lr-scheduler cosine` + `--warmup-epochs N` to
`train.py`. R1 v2 launched 2026-05-24 with linear warmup
(3 epochs, 1e-5 → 1e-3) then cosine decay to 1e-5 over the
remaining 37 epochs. If R1 v2 ep 10–15 sims=800 lands clearly above
R1 v1's 1,941 ceiling, cosine was the missing piece.

**Damage.** Original R1 ran 32h (boot + ep 0–11) on `i-0baf68ab2fa93e155`
(g6e.8xlarge spot, us-east-1a) before eviction, then 13h resume on
`i-058bea691f419c69c` (us-east-1d) before manual kill. ~45h × ~$1.74/h
spot = **~$78 in compute, with ep 6's 2,146 sims=4000 as the keep-able
output**. The "discovery via observability" pattern paid for itself;
without per-epoch `train_history.json` syncs we'd have thrown the
whole run at it longer.

## 11: Docker `$ECR` Escaped in Heredoc Killed the First R1 v2 Launch

**Symptom.** R1 v2's first instance produced no checkpoints. The
host-launch log showed:
```
docker: invalid reference format.
```

**Root cause.** The launcher's user-data heredoc had
`\$ECR/wm-chess-gpu:latest` in the `docker run` line. The backslash
escaped the dollar, so the heredoc emitted a literal `$ECR/wm-chess-gpu:latest`
into the EC2's bash script. `$ECR` wasn't defined in the EC2's
environment (only in the launcher script's environment), so the
image arg resolved to `/wm-chess-gpu:latest`, no registry prefix.

**Fix.** Drop the backslash so the launcher-script heredoc interpolates
`$ECR` at user-data generation time, giving the EC2 the full
`594561963943.dkr.ecr.us-east-1.amazonaws.com/wm-chess-gpu:latest`.

**Damage.** Two failed boots: `i-08bc4347aaa36e74a` (eu-central-1c
g6e.8xlarge spot, manual launch) and `i-0874256bc9a24d7b1`
(eu-central-1c g6e.8xlarge spot, watcher relaunch). Each ran ~2 min
before the cleanup trap fired and self-terminated. **~$0.20 in EC2
time + ~30 min of debugging.**

## 12: Baked Docker Image Was Stale for h2h_mp.py

**Symptom.** The d10-vs-d15 head-to-head EC2 immediately errored:
```
h2h_mp.py: error: unrecognized arguments: --n-blocks-a 20 --n-filters-a 256
```

**Root cause.** The `--n-blocks-a` / `--n-filters-b` flags landed in
`h2h_mp.py` *after* the wm-chess-gpu image was last built. The image
shipped the older argparser that didn't know the new flags.

**Fix.** Have the launcher `curl` the latest `h2h_mp.py` from `main`
at runtime *before* invoking it. Cleaner than rebuilding the image
for a one-script change. Same pattern works for any script the image
ships pre-baked but main has since modified, used again for the
cosine-LR `train.py` patch.

**Damage.** First h2h instance `i-096d1209fdb836a99` (g6.4xlarge OD
us-east-1) ran ~2 min then failed cleanly. Re-fired the launcher with
the curl trampoline as `i-05a66aaaf5c27cffb` (g6.4xlarge OD us-east-1),
which produced the actual 0/104/0 result in 6h 24min.
**~$0.20 wasted + 6h delay on the headline H2H number.**

## 13: Spot Eviction at ep 6 with No Optimizer Resume

**Symptom.** R1 v1 was spot-evicted in us-east-1a around ep 6 with
"no Spot capacity available." The watcher daemon re-launched and
the entrypoint auto-resumed from the latest S3 ckpt, but the new
process started training again at LR=1e-3 from scratch in
optimizer-state terms, just with the ep 6 weights pre-loaded.

**Root cause.** `train.py` saves model weights but not optimizer
state. Adam's per-parameter momentum / variance estimates rebuild
from zero, which biases the first few post-resume steps. With a
*scheduled* LR this would also mean the schedule resets, handled
in the cosine update via `scheduler.step()` matching `--start-epoch`.

**Fix.** Pin `RUN_ID` in the launcher so a relaunch lands in the
same S3 path and the auto-resume picks up the latest ckpt. The
new spot watcher (`wm-d15-run1-watcher.sh`) re-fires the launcher
on eviction, costing ~58 min for dataset reload but zero model
weight loss per eviction.

**Damage.** Original R1 (`i-0baf68ab2fa93e155`) evicted after ep 6.
Watcher fired the resume as `i-058bea691f419c69c` (us-east-1d spot)
which trained ep 7–11 before manual kill. **~1h dataset reload =
~$1.74**, the post-resume training itself was productive.

## 14: Cancelled Spot Request Didn't Free Quota Immediately

**Symptom.** After terminating R1 v1, every `RunInstances` call
returned `MaxSpotInstanceCountExceeded` for ~5 minutes. AWS still
counted the cancelled `sir-awgffx7m` (state `request-canceled-and-instance-running`)
against the spot quota.

**Root cause.** Spot request lifecycle is independent of the
instance lifecycle. Even after the instance is terminated, the
spot request needs to transition to `cancelled` before its vCPU
count is released.

**Fix.** `aws ec2 cancel-spot-instance-requests --spot-instance-request-ids <id>`
explicitly. Then poll for state `cancelled` before retrying the
launch. Adds ~30s but avoids the noisy quota errors.

**Damage.** Orphaned request `sir-awgffx7m` (after `i-058bea691f419c69c`
terminated) blocked R1 v2 cosine launch for ~10 min. **No $ cost,
but a wasted 10-minute debug cycle and three "MaxSpotInstanceCountExceeded"
errors that looked like a quota problem before we realized.**

## 15: eu-central-1c Lost g6e.8xlarge Spot Capacity Mid-Run

**Symptom.** R1 v2 successfully launched in eu-central-1c spot,
then the watcher's relaunch attempt 6h later (after the docker bug
failure) returned `InsufficientInstanceCapacity` from the same AZ.
AWS suggested 1a or 1b instead, but earlier we'd tried 1a (out)
and 1b (out) before settling on 1c.

**Root cause.** Spot capacity in g6e.8xlarge is *highly* AZ-local
and time-of-day-volatile. The AZ that fulfilled at launch may not
have capacity 6h later.

**Fix.** When a launcher fails with `InsufficientInstanceCapacity`,
try other AZs in the region before falling back regions. The
deep-eval daemon also got a third tier (eu-central-1 spot) added
to its `REGION_ORDER` for the same reason. A real fix would be
spot-rover-style proactive placement scoring, already exists at
`infra-eks/spot-rover/`, just not wired into the training launchers
yet.

**Damage.** R1 v2 cosine cycled through five g6e.8xlarge spot
instances chasing capacity: `i-08bc4347aaa36e74a` and
`i-0874256bc9a24d7b1` (eu-central-1c, both died on the docker bug
above), `i-05726d09d2b5fcaed` (us-east-1d, evicted ~27 min into
dataset load), `i-03610018666046c42` and `i-0eb7640cf94250fe8`
(us-east-1d watcher relaunches, also evicted before ep 0),
`i-0556fbe333082ead0` (us-east-1b, evicted within an hour). **0
ckpts produced across all attempts. ~$15 in spot wastage** before
giving up and switching to OD as `i-0a4c27118ff4e62ba`
(eu-central-1 OD) the next morning.

## 16: Autoeval Queue Stuck When Both OD Quotas Were Full

**Symptom.** Run 2 ckpts ep 2/3/4 didn't get sims=800 evals for
hours. The autoeval daemon kept logging `LAUNCH eval EC2 for ...`
followed by `us-east-1 launch failed` and `eu-central-1 launch
failed` every ~6 minutes, then deleting the claim marker.

**Root cause.** Both regions' OD G/VT quotas (32 vCPU each) were
saturated: us-east-1 by two long-running sims=4000 EC2s
(2 × 16 vCPU), eu-central-1 by the R2 training instance
(32 vCPU). Spot quotas were also full (R1 training in us-east-1
spot). No slot anywhere.

**Fix.** Added `eu-central-1:spot` as a third tier in
`wm-autoeval-daemon.sh`'s `REGION_ORDER`. Spot quotas are
*separate* from OD quotas, and eu-central-1 spot had 32 free vCPU
because R2 was on OD. Patch shipped in commit `2e809e7`. Drained
the queue overnight.

**Damage.** ~2h of stuck-queue while the patch was being written
and shipped. R2 ep 2/3/4 evals delayed but eventually drained;
**no $ cost** beyond the daemon's S3 list calls. The lesson
was-cheap, the patch was-cheap, the consequence-was-cheap, a
nice cluster of "just plumbing" wins.

## 17: One Run 2 ckpt (ep 2) Never Got Evaluated

**Symptom.** Run 2's `eval_results-distilled_epoch002.txt` is the
only file missing from an otherwise contiguous ep0–ep12 sims=800
sequence.

**Root cause.** The eu-central-1 spot fallback wasn't yet deployed
when ep 2 was first polled. The autoeval daemon claim-launched-
failed-released the marker repeatedly during the quota crunch.
Once the fallback shipped, the daemon polled ahead to ep 3/4 and
later ckpts; ep 2 fell out of the "new ckpts" loop because no
ckpt was newer than its claim marker.

**Fix.** Manual re-fire if the data point matters. (For Run 2
specifically: with 20+ other epochs evaluated, ep 2 isn't
load-bearing for the trajectory.) Future improvement: have the
daemon distinguish "permanently failed to claim" from "released
on launch failure" and aggressively re-try the latter set.

**Damage.** One missing data point in the R2 ep 0–29 sims=800
trajectory. No $ cost: the daemon's S3 calls are negligible. The
hole is visible in the R2 trajectory table on the
[experiments page](/experiments/#d15-46m).

## 18: Parallelizing the Self-Play Gate onto CPU Workers Was Slower

**Symptom.** The gated self-play loop's candidate-vs-champion gate
match (60 games) + Stockfish yardstick, after being "parallelized"
across the 14 CPU self-play workers, ran **~110 min+**, *slower* than
the original single-threaded gate (~45 min).

**Root cause.** The original gate ran in the trainer process on the
**GPU** (~45 s/game). The "fix" fanned the match across the CPU worker
pool (the pattern self-play uses). But for the 20×256 / ~23.7M-param
net, CPU batch-1 MCTS is ~tens-of-times slower *per game* than the GPU,
so 14× worker parallelism didn't come close to offsetting it. Self-play
uses CPU workers because *many concurrent* full games amortize, and the
net was historically small (10×128, where "batch-1 is faster on CPU"
held), false for 20×256.

**Fix.** Reverted (commit `67335fc`): gate match + Stockfish eval run
single-threaded in the trainer on the GPU. The gate was never the
bottleneck; self-play dominates at ~2.7h/iter, so a 45-min GPU gate is
~13% overhead. If gate time matters, cut `gate_games` / `gate_sims`, not
CPU parallelism. Rule: before parallelizing GPU-bound MCTS onto CPU
workers, check the per-game device cost first.

**Damage.** One gated relaunch discarded ~6h in (`i-0a729949eca23ef5d`),
plus the ~110-min gate that triggered the diagnosis before it was killed.
**~$10 in g5.4xlarge time + the relaunch churn.**

## Session-Total Damage

Adding up #10 through #17 from the 2026-05-22 → 2026-05-25 d15
campaign:

| Bucket | Burn |
|---|---|
| Spot wastage chasing capacity (#15) | ~$15 |
| Failed boots from docker bug (#11, #12) | ~$0.40 |
| Dataset reload after eviction (#13) | ~$1.74 |
| Original R1 constant-LR run (#10) | ~$78 (kept ep 6 = 2,146 sims=4000) |
| **Total** | **~$95** |

Plus ~6h of headline-result delay (#12) and ~10 min of misdiagnosed
quota errors (#14). The eventually-productive output: 50+ R2 ckpts,
3 sims=4000 d15 measurements (R1 ep 1, R1 ep 7, R2 ep 6/7/12),
1 H2H measurement, the cosine LR train.py patch, and an autoeval
daemon with proper three-region fallback. **~$95 is cheap for an
overhaul this thorough.**

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
7. **Use cosine LR + warmup for nets >30M params**. Constant LR
   converges 20×256 fine but plateaus 40×256. See #10.
8. **Pin `RUN_ID` + auto-resume + a watcher daemon** turns spot
   eviction from "lose a day's training" into "lose 1 hour of
   dataset reload." See #13.
9. **Cancel spot requests explicitly** when terminating spot
   instances. Cancel is *not* implicit on termination. See #14.
10. **Multiple region:market tiers for any auto-launching daemon.**
    OD quotas in 1 region fill up faster than expected when long
    deep-eval EC2s pile in alongside training. See #16.
11. **`curl` the latest script into the container at runtime** when
    you can't wait for an image rebuild. Works for `train.py`,
    `entrypoint-train.sh`, `h2h_mp.py`, anything else `COPY`-ed
    into the image. See #12.
12. **Don't escape `$` in launcher heredocs** unless you mean to
    defer the variable to the EC2-side bash. See #11.
