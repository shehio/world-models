#!/usr/bin/env bash
# Auto-fire the d15-30M training run when the d15 250K datagen finishes.
#
# Watches the gen Job + S3 merged dataset and walks the handoff:
#
#   1. Poll the datagen Job (wm-chess-gen-d15-250k) until COMPLETIONS=8/8.
#   2. Apply the merge Job. Watch S3 for merged/data.npz to appear.
#   3. Fire `infra-eks/launchers/d15-full30m.sh` to launch the bare-EC2
#      g6e.8xlarge training run on the merged dataset.
#
# Idempotent and crash-safe: each step writes a marker file under
# /tmp/wm-d15-30m-handoff.state so re-running picks up where it left off.
# Lives at the laptop / orchestrator level — does not need to run inside
# the cluster.
#
# Usage:
#   nohup bash infra-eks/daemons/wm-d15-30m-handoff-daemon.sh > /tmp/wm-d15-30m-handoff.log 2>&1 &
#   # tail -F /tmp/wm-d15-30m-handoff.log to follow
#   # echo stop > /tmp/wm-d15-30m-handoff.stop to exit cleanly
#
# Why a daemon and not a cron: we want sub-minute responsiveness on the
# final handoff (the merge is fast, ~5 min) so a 5-min cron would add
# unnecessary lag between merge-complete and training-launch.

set -u

LOG=/tmp/wm-d15-30m-handoff.log
STATE_DIR=/tmp/wm-d15-30m-handoff.state
STOP_FILE=/tmp/wm-d15-30m-handoff.stop
INTERVAL_SEC=${INTERVAL_SEC:-60}

# All overridable via env so the daemon can re-target without code edits
# (used during the 2026-05-20 us-east-1 → eu-central-1 migration when
# us-east-1 spot capacity collapsed mid-run).
REGION=${REGION:-us-east-1}
CLUSTER=${CLUSTER:-wm-chess-gen-d15-250k}
KUBECTL_CTX=${KUBECTL_CTX:-wm-gen-d15-250k}
JOB_NAME=${JOB_NAME:-wm-chess-gen-d15-250k}
S3_BUCKET=${S3_BUCKET:-wm-chess-library-594561963943}
S3_PREFIX=${S3_PREFIX:-d15-mpv8-T1-g250000-20260519T0412Z}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$STATE_DIR"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Step 1: wait for the datagen Job to succeed all 8 indices.
wait_for_datagen_complete() {
    if [ -f "$STATE_DIR/datagen.done" ]; then
        log "[1/3] datagen already marked done (state file present)"
        return 0
    fi
    log "[1/3] waiting for $JOB_NAME to reach 8/8 succeeded"
    while true; do
        if [ -f "$STOP_FILE" ]; then log "stop file found, exiting"; exit 0; fi
        local succeeded
        succeeded=$(kubectl --context "$KUBECTL_CTX" get job "$JOB_NAME" \
            -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "")
        if [ "$succeeded" = "8" ]; then
            log "[1/3] datagen complete (8/8 succeeded)"
            touch "$STATE_DIR/datagen.done"
            return 0
        fi
        sleep "$INTERVAL_SEC"
    done
}

# Step 2: apply the merge Job, wait for merged/data.npz on S3.
wait_for_merge_complete() {
    if [ -f "$STATE_DIR/merge.done" ]; then
        log "[2/3] merge already marked done (state file present)"
        return 0
    fi
    log "[2/3] applying merge Job via launchers/gen-d15-250k.sh merge"
    bash "$REPO_ROOT/infra-eks/launchers/gen-d15-250k.sh" merge 2>&1 \
        | tee -a "$LOG"
    log "[2/3] waiting for s3://$S3_BUCKET/$S3_PREFIX/merged/data.npz"
    local merged_uri="s3://$S3_BUCKET/$S3_PREFIX/merged/data.npz"
    while true; do
        if [ -f "$STOP_FILE" ]; then log "stop file found, exiting"; exit 0; fi
        if aws s3 ls "$merged_uri" --region "$REGION" > /dev/null 2>&1; then
            local size
            size=$(aws s3 ls "$merged_uri" --region "$REGION" \
                | awk '{print $3}')
            log "[2/3] merge complete: $merged_uri ($size bytes)"
            touch "$STATE_DIR/merge.done"
            return 0
        fi
        sleep "$INTERVAL_SEC"
    done
}

# Step 3: fire the d15-full30m bare-EC2 training launcher.
fire_training() {
    if [ -f "$STATE_DIR/training.fired" ]; then
        log "[3/3] training already fired (state file present)"
        return 0
    fi
    log "[3/3] firing infra-eks/launchers/d15-full30m.sh"
    local out
    out=$(bash "$REPO_ROOT/infra-eks/launchers/d15-full30m.sh" 2>&1)
    log "[3/3] launcher output: $out"
    # Capture the instance id (matches ^i-[0-9a-f]+ from run-instances output).
    local instance_id
    instance_id=$(echo "$out" | grep -oE '^i-[0-9a-f]+' | head -1)
    if [ -n "$instance_id" ]; then
        log "[3/3] training instance launched: $instance_id"
        echo "$instance_id" > "$STATE_DIR/training.fired"
    else
        log "[3/3] WARNING: no instance id parsed; retrying next cycle"
        sleep "$INTERVAL_SEC"
        return 1
    fi
}

log "wm-d15-30m-handoff-daemon starting"
log "  cluster=$CLUSTER  context=$KUBECTL_CTX  s3=$S3_BUCKET/$S3_PREFIX"
log "  state=$STATE_DIR  interval=${INTERVAL_SEC}s"

wait_for_datagen_complete
wait_for_merge_complete
while ! fire_training; do :; done

log "handoff complete. clean up with: rm -rf $STATE_DIR $LOG"
