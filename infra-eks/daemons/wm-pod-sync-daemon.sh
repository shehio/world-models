#!/usr/bin/env bash
# Periodic kubectl-exec sync of the CURRENTLY-RUNNING us-east d15 pod's
# checkpoint dir to S3. The pod runs the OLD image which only syncs at
# the end of training; this gives us crash-safe per-checkpoint durability.
#
# Future runs use the patched train.py and don't need this daemon.

set -u
LOG=/tmp/wm-pod-sync-daemon.log
INTERVAL_SEC=${INTERVAL_SEC:-600}   # 10 min
PREFIX="d15-mpv8-T1-g100000-20260512T1844Z"
S3_BASE="s3://wm-chess-library-594561963943/$PREFIX/checkpoints/net-20x256/20260513T1946Z"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== started; sync every ${INTERVAL_SEC}s -> $S3_BASE/ ==="

while true; do
    # Filter to the d15 job specifically — `app=wm-chess-train` would
    # also match the concurrent d10 pod (whose train.py already syncs
    # itself per checkpoint, no daemon needed).
    POD=$(kubectl --context wm-train-us get pods -l job-name=wm-chess-train \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -z "$POD" ]; then
        log "no d15 pod found; daemon exiting (training presumably done)"
        exit 0
    fi

    log "syncing /work/checkpoints/ from pod $POD"
    kubectl --context wm-train-us exec "$POD" -- bash -c "
        aws s3 sync /work/checkpoints/ '$S3_BASE/' --no-progress \
            --exclude '*.tmp.*' \
            --exclude '.eval_progress_*' \
            --exclude '.train_progress.json' 2>&1
    " 2>&1 | tail -5 | tee -a "$LOG"

    sleep "$INTERVAL_SEC"
done
