#!/usr/bin/env bash
# Auto-fire the Go 9x9 training run when the GPU datagen finishes.
#
# Mirrors wm-d15-30m-handoff-daemon.sh, scoped to the Go GPU pipeline:
#
#   1. Poll the Go GPU datagen Job (wm-go-gpu-9x9) until COMPLETIONS=4/4.
#   2. Merge all shards_partial/pod-*/worker_*_chunk_*.npz into one
#      merged/data.npz on S3, using the Go-native merge.py (the chess
#      streaming merge_chunks.py expects a different on-disk layout).
#   3. Fire infra-eks/launchers/go-gpu-train.sh to launch a g6e.xlarge
#      bare-EC2 training run on the merged dataset.
#
# Idempotent and crash-safe: each step writes a marker file under
# /tmp/wm-go-gpu-handoff.state so re-running picks up where it left off.
# Lives at the laptop / orchestrator level — does not need to run inside
# the cluster.
#
# Usage:
#   nohup bash infra-eks/daemons/wm-go-gpu-handoff-daemon.sh > /tmp/wm-go-gpu-handoff.log 2>&1 &
#   # tail -F /tmp/wm-go-gpu-handoff.log to follow
#   # echo stop > /tmp/wm-go-gpu-handoff.stop to exit cleanly

set -u

LOG=/tmp/wm-go-gpu-handoff.log
STATE_DIR=/tmp/wm-go-gpu-handoff.state
STOP_FILE=/tmp/wm-go-gpu-handoff.stop
INTERVAL_SEC=${INTERVAL_SEC:-60}

# Overridable so the daemon can re-target without code edits.
REGION=${REGION:-us-east-1}
KUBECTL_CTX=${KUBECTL_CTX:-iam-root-account@wm-go-gpu-use1.us-east-1.eksctl.io}
JOB_NAME=${JOB_NAME:-wm-go-gpu-9x9}
COMPLETIONS=${COMPLETIONS:-4}
S3_BUCKET=${S3_BUCKET:-wm-chess-library-594561963943}
S3_PREFIX=${S3_PREFIX:-go-9x9-mpv8-gpu-v400-20260521T1230Z}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$STATE_DIR"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Step 1: wait for the GPU datagen Job to succeed all 4 indices.
wait_for_datagen_complete() {
    if [ -f "$STATE_DIR/datagen.done" ]; then
        log "[1/3] datagen already marked done (state file present)"
        return 0
    fi
    log "[1/3] waiting for $JOB_NAME to reach $COMPLETIONS/$COMPLETIONS succeeded"
    while true; do
        if [ -f "$STOP_FILE" ]; then log "stop file found, exiting"; exit 0; fi
        local succeeded
        succeeded=$(kubectl --context "$KUBECTL_CTX" get job "$JOB_NAME" \
            -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "")
        if [ "$succeeded" = "$COMPLETIONS" ]; then
            log "[1/3] datagen complete ($COMPLETIONS/$COMPLETIONS succeeded)"
            touch "$STATE_DIR/datagen.done"
            return 0
        fi
        sleep "$INTERVAL_SEC"
    done
}

# Step 2: merge all per-pod chunks locally and push merged/data.npz to S3.
# The Go layout is flat (shards_partial/pod-N/worker_NN_chunk_NNNN.npz) —
# no `chunks/` subdir like chess — so we use distill_go's own merge.py.
# Peak RAM is bounded by total dataset size (16k games × 9x9 × 4 planes
# ≈ ~2-4 GB for 9x9). Comfortable on the laptop running this daemon.
wait_for_merge_complete() {
    if [ -f "$STATE_DIR/merge.done" ]; then
        log "[2/3] merge already marked done (state file present)"
        return 0
    fi
    local merged_uri="s3://$S3_BUCKET/$S3_PREFIX/merged/data.npz"
    # If merged is already on S3 (e.g. from a prior daemon run), skip.
    if aws s3 ls "$merged_uri" --region "$REGION" > /dev/null 2>&1; then
        log "[2/3] merged/data.npz already on S3, skipping merge"
        touch "$STATE_DIR/merge.done"
        return 0
    fi

    local work=$STATE_DIR/merge-work
    mkdir -p "$work/chunks"
    log "[2/3] syncing shards_partial/ → $work/chunks/ (flattened)"
    # Flatten pod-N/worker_*.npz → chunks/pod-N_worker_*.npz so merge.py
    # sees them as one dir (it requires the worker_*_chunk_*.npz naming
    # and ignores subdirectories).
    aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/shards_partial/" "$work/raw/" \
        --region "$REGION" --no-progress
    # The merge.py regex is anchored ^worker_..., so reuse the file
    # names directly. Per-pod prefix is unnecessary — chunks are
    # disambiguated by worker idx (pod-N seeded BASE_SEED + N*1000 so
    # worker idx ranges differ between pods).
    find "$work/raw" -name "worker_*_chunk_*.npz" -print0 \
        | xargs -0 -I{} ln -sf {} "$work/chunks/"

    log "[2/3] running distill_go merge.py"
    (
        cd "$REPO_ROOT/experiments/distill-go" \
        && uv run python -c "
from distill_go.merge import merge_chunks
m = merge_chunks('$work/chunks', '$work/data.npz')
print('merged:', m)
"
    ) 2>&1 | tee -a "$LOG"

    if [ ! -f "$work/data.npz" ]; then
        log "[2/3] merge FAILED — data.npz missing"
        return 1
    fi
    log "[2/3] uploading merged/data.npz to $merged_uri"
    aws s3 cp "$work/data.npz" "$merged_uri" --region "$REGION" --no-progress
    touch "$STATE_DIR/merge.done"
}

# Step 3: train locally on the daemon host.
# Go 9x9 dataset is ~1.2M positions / ~50 MB merged — MPS on the Mac runs the
# 4×64 ResNet at ~20s/epoch (per the 2026-05-21 smoke test on 188K positions).
# A cloud GPU launcher is overkill for this size; train inline and upload
# checkpoints to S3.
fire_training() {
    if [ -f "$STATE_DIR/training.fired" ]; then
        log "[3/3] training already fired (state file present)"
        return 0
    fi
    local merged="$STATE_DIR/merge-work/data.npz"
    local ckpt_dir="$STATE_DIR/ckpt"
    mkdir -p "$ckpt_dir"
    # 8x128 ResNet matches the validated v2 local-experiment config (per
    # project_wm_go_cloud_run memory). AlphaZero-9x9-appropriate sizing —
    # don't downscale "for speed"; keep the paper-faithful architecture.
    log "[3/3] training locally: scripts/train.py on $merged (8x128, 20 epochs)"
    (
        cd "$REPO_ROOT/experiments/distill-go" \
        && uv run python scripts/train.py \
            --data "$merged" \
            --board-size 9 --n-input-planes 4 \
            --epochs 20 --batch-size 256 \
            --n-blocks 8 --n-filters 128 \
            --ckpt-dir "$ckpt_dir" \
            --save-every 2
    ) 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" != "0" ]; then
        log "[3/3] training failed rc=$rc; retrying next cycle"
        return 1
    fi
    log "[3/3] uploading checkpoints to s3://$S3_BUCKET/$S3_PREFIX/checkpoints/"
    aws s3 sync "$ckpt_dir" \
        "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/9x9-6x96-$(date -u +%Y%m%dT%H%MZ)/" \
        --region "$REGION" --no-progress
    touch "$STATE_DIR/training.fired"
    log "[3/3] training complete"
}

log "wm-go-gpu-handoff-daemon starting"
log "  job=$JOB_NAME  context=$KUBECTL_CTX  s3=$S3_BUCKET/$S3_PREFIX"
log "  state=$STATE_DIR  interval=${INTERVAL_SEC}s"

wait_for_datagen_complete
while ! wait_for_merge_complete; do sleep "$INTERVAL_SEC"; done
while ! fire_training; do :; done

log "handoff complete. clean up with: rm -rf $STATE_DIR $LOG"
