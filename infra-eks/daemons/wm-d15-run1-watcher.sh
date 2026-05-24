#!/usr/bin/env bash
# Watcher for the d15 Run 1 (40x256 spot) training instance. If the spot
# is evicted, re-fires infra-eks/launchers/d15-full30m.sh which:
#   - launches a fresh spot g6e.8xlarge in us-east-1d (best L40S spot score)
#   - auto-resumes from latest S3 ckpt (RUN_ID is pinned in the launcher)
#
# Polls every 5 min. Tracks the current instance id in /tmp/wm-d15-run1.id.
# Idempotent: if the instance is still running, does nothing.

set -u
LOG=/tmp/wm-d15-run1-watcher.log
INSTANCE_ID_FILE=/tmp/wm-d15-run1.id
INTERVAL_SEC=${INTERVAL_SEC:-300}
REGION=${REGION:-us-east-1}    # override via env when the run lives elsewhere

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Seed with the currently-known instance id (passed on first start as $1
# or read from the file on subsequent restarts).
if [ -n "${1:-}" ]; then
    echo "$1" > "$INSTANCE_ID_FILE"
fi

if [ ! -f "$INSTANCE_ID_FILE" ]; then
    log "ERROR: no instance id in $INSTANCE_ID_FILE — start with: bash wm-d15-run1-watcher.sh <instance-id>"
    exit 1
fi

CURRENT_ID=$(cat "$INSTANCE_ID_FILE")
log "=== started; watching $CURRENT_ID in $REGION; poll every ${INTERVAL_SEC}s ==="

while true; do
    state=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$CURRENT_ID" \
            --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)

    case "$state" in
        running|pending)
            : # healthy, nothing to do
            ;;
        terminated|shutting-down|stopped|stopping|"None"|"")
            log "instance $CURRENT_ID is $state — re-launching via d15-full30m.sh"
            cd /Users/shehabyasser/workspace/sandbox/world-models
            new_out=$(bash infra-eks/launchers/d15-full30m.sh 2>&1)
            new_id=$(echo "$new_out" | head -1 | awk '{print $1}')
            if echo "$new_id" | grep -qE '^i-[0-9a-f]+$'; then
                log "  re-launched as $new_id"
                echo "$new_id" > "$INSTANCE_ID_FILE"
                CURRENT_ID=$new_id
            else
                log "  re-launch FAILED: $new_out"
                # Don't update CURRENT_ID; we'll retry the failed launch next cycle.
            fi
            ;;
        *)
            log "instance $CURRENT_ID in unexpected state: $state"
            ;;
    esac

    sleep "$INTERVAL_SEC"
done
