# Bare-EC2 launchers

One-shot scripts that bring up a single EC2, run a job, and self-terminate via
`shutdown -h` + `instance-initiated-shutdown-behavior=terminate`. Used when EKS
is unavailable for the capacity we need (e.g. an L40S type with no quota in our
cluster region) or when a one-off doesn't justify a Job submission.

## Pattern

All launchers share the same shape:

1. Constants at top: `ACCOUNT_ID`, `IMAGE_REGION`, `LAUNCH_REGION`, `AMI`,
   `SUBNET`, `INSTANCE_TYPE`, `INSTANCE_PROFILE`.
2. A `USER_DATA` heredoc that:
   - traps EXIT to upload the host log to S3 and `shutdown -h +1`,
   - logs in to the us-east-1 ECR, pulls `wm-chess-gpu:latest`,
   - `docker run --rm --gpus all` with the appropriate entrypoint
     (`/entrypoint-train.sh` for training, `bash -lc '…eval.py…'` for eval).
3. `aws ec2 run-instances` with `--instance-initiated-shutdown-behavior terminate`
   so the EC2 disappears when the workload finishes.

Cross-region wiring works because:
- The ECR image lives in `us-east-1` and is cross-region-pulled.
- The S3 bucket (`wm-chess-library-…`) lives in `us-east-1` and is cross-region
  read/written.
- The IAM instance profile is global.

So the LAUNCH_REGION can move freely (us-east-1, eu-central-1) to dodge G/VT
vCPU quota in any single region.

## Launchers

### Datagen

| Script | What it does |
| --- | --- |
| `gen-d15-250k.sh` | Launch the d15 250K-game datagen pipeline (d15 teacher at d10-30M scale). |
| `gen-d15-10x.sh` | Launch the 10× d15 datagen pipeline: 500K games at depth 15, multipv=8, T=1. |
| `gen-d15-add5m.sh` | Generate ~5M more d15 positions (50K games), staged for merging with the 5M baseline. |

### Chess training

| Script | What it does |
| --- | --- |
| `d10-l40s-eu.sh` | Resume d10 training from `distilled_epoch004` (ep 5) on an L40S in eu-central-1. Bypasses EKS when us-east has no L40S capacity. |
| `d10-full30m.sh` | Experiment C: train d10 on the FULL ~30M positions (vs 5M subsample) — tests whether the plateau is dataset-size-limited. Needs g6e.8xlarge (256 GB RAM). |
| `d15-40x256-eu.sh` | Experiment A: train a 40-block ResNet on d15 — tests whether the ~1800 Elo plateau is capacity-limited. |
| `d15-full30m.sh` | R1: train the 40×256 net on the full d15 dataset (45.9M positions, no MAX_POSITIONS cap). g6e.8xlarge. Prereq: `wm-chess-gen-d15-250k` cluster gen+merge done. |
| `d15-20x256-lowlr.sh` | R2 v1: 20×256 + LR=5e-4 + wd=1e-3 on the d15 46M dataset — the regularization-leaning recipe. |
| `d15-20x256-cosine.sh` | R2 v2: same as `d15-20x256-lowlr.sh` plus the cosine LR schedule — the run that produced the 2,301 project high. |

### Chess eval (one-off deep reads)

| Script | What it does |
| --- | --- |
| `eval-deep-sims.sh` | Experiment E: re-eval d15 ep 20 with `--sims 4000` (vs daily 800) — tests whether the model is under-read by routine eval depth. |
| `eval-c-ep5-sims4000.sh` | Deep-sims eval on C's ep 5 ckpt (d10 full-30M 20×256). |
| `eval-c-ep5-sims2000-uci1800.sh` | Sims-curve fill for C ep 5: 2,000 sims at UCI=1800 (between the 800/4,000 endpoints). |
| `eval-c-ep5-sims4000-uci2000.sh` | Strength-ceiling probe for C ep 5: 4,000 sims vs the stronger UCI=2000 anchor. |
| `eval-c-ep10-sims4000-uci1800.sh` | Epoch-vs-search interaction: C ep 10 at 4,000 sims, UCI=1800. |
| `eval-d15-40x256-ep1-sims4000.sh` | One-off sims=4,000 eval on the d15 40×256 run's ep 1 ckpt (below the deep-eval daemon's threshold). |
| `eval-r2v2-ep4-sims8000-uci2000.sh` | sims=8,000 vs UCI=2,000 on R2 v2 ep 4 — produced the 2,153 tightest-CI measurement. |
| `eval-r2v2-ep14-sims4000.sh` | sims=4,000 on R2 v2 ep 14 (missed by the offline deep-eval daemon) — produced the 2,301 project high. |
| `bisect-all-ckpts.sh` | Run elo-bisect across every (run × epoch) checkpoint: d15 5M baseline, d15 5M 40×256, d10 5M subset. |

### Chess self-play RL

| Script | What it does |
| --- | --- |
| `d10-selfplay.sh` | AlphaZero-style self-play RL bootstrapped from d10 ep 15 (the 2,189 ckpt). |
| `selfplay-chess-r2v2-ep14.sh` | Ungated self-play RL on top of R2 v2 ep 14 (the 2,301 ckpt). |
| `selfplay-chess-gated-r2v2-ep14.sh` | Gated self-play MVP on R2 v2 ep 14: arena gating + KL-anchor + replay window + in-loop Stockfish yardstick. |
| `eval-selfplay-current.sh` | One-shot eval of the live self-play loop's `current.pt` without waiting for the loop's eval cadence. |
| `eval-selfplay-final.sh` | Evaluate the self-play loop's final checkpoint (the autoeval daemon's filename filter skips `net_iter*.pt`/`final.pt`). |
| `eval-selfplay-fix-test.sh` | Verification run for the self-play LR fix (1e-3 → 1e-5) from the distilled prior. |
| `eval-selfplay-hour000.sh` | Diagnostic: eval the first hourly self-play ckpt to pinpoint when the regression started. |
| `h2h-chess-selfplay-vs-prior.sh` | Periodic head-to-head: latest self-play ckpt vs the R2 v2 ep 14 prior, for early Elo signal. |
| `h2h-d10-vs-d15.sh` | Head-to-head: best d10 ckpt vs best d15 ckpt, both at the same MCTS sim count (via `h2h_mp.py`). |

### MuZero

| Script | What it does |
| --- | --- |
| `muzero-chess.sh` | From-scratch MuZero chess training on a bare EC2 (OD by default — spot kept evicting the long runs). |
| `muzero-distill-init.sh` | Distill-init MuZero: train only the dynamics `g` on top of the frozen 2,301 distilled teacher. |

### Go (9×9)

| Script | What it does |
| --- | --- |
| `eval-go-9x9-8x128.sh` | Elo eval of the trained 8×128 Go net vs KataGo at fixed visits. |
| `selfplay-go-9x9.sh` | AlphaZero-style self-play on top of the 8×128 distilled Go prior (the KataGo-parity ckpt). |
| `h2h-go-selfplay-vs-prior.sh` | Head-to-head: latest Go self-play ckpt vs the distilled prior (the vs-random eval saturated at 10/0/0). |
| `calibrate-katago-vs-gnugo.sh` | Calibration match: KataGo@v200 vs GNU Go on 9×9, to anchor KataGo's absolute Elo. |
| `calibrate-pachi.sh` | Calibration match: KataGo@v200 vs Pachi (GNU Go was too weak — swept 100/0). |
| `calibrate-chained.sh` | Chained calibration: derive KataGo@v200's absolute Elo through a measurable intermediate anchor. |

## Retry-until-quota pattern

These launchers fail loudly if G/VT vCPU quota is exhausted. A common retry
loop:

```bash
# Retry the C launch every 2 min until quota frees.
while true; do
    out=$(bash infra-eks/launchers/d10-full30m.sh 2>&1)
    if echo "$out" | grep -qE '^i-[0-9a-f]+'; then echo "launched: $out"; break; fi
    sleep 120
done
```

For multi-region retry, override `LAUNCH_REGION`/`AMI`/`SUBNET` per attempt
(see `daemons/wm-autoeval-daemon.sh` for the daemon-side version with region
fallback baked in).

## Logs and results

Each launcher writes:
- `/var/log/{train,eval}.log` on the EC2 (mirror of stdout/stderr).
- An EXIT trap copies that log to S3 — for training, to
  `s3://<bucket>/<prefix>/checkpoints/net-<NxF>/<RUN_ID>/host-launch.log`;
  for eval, to `s3://<…>/eval_results-<ckpt>.log` next to the result.

Look there first when an experiment "didn't produce a checkpoint" — the EC2
has already terminated.
