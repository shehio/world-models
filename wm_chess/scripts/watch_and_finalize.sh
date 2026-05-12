#!/usr/bin/env bash
# Watch an already-running fleet of N gen VMs: poll tmux has-session every
# 5 min, rsync library back when each completes, terminate when all done.
# Useful when the launcher died mid-run and you've recovered the VMs.
#
# Inputs:
#   FLEET file format (one VM per line, whitespace-separated):
#       <instance_id> <public_ip> <seed>
#
# Usage:
#   FLEET=/tmp/fleet.txt KEY_NAME=mykey bash watch_and_finalize.sh

set -uo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
LIB_DIR="$REPO_ROOT/library"

: "${FLEET:?Set FLEET=/path/to/fleet.txt}"
: "${KEY_NAME:?Set KEY_NAME=<your-aws-key>}"
: "${REGION:=us-east-1}"
: "${POLL_INTERVAL:=300}"

KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ConnectTimeout=10)

log() { echo "[$(date -u +%H:%M:%S)] $*" >&2; }

cleanup_on_exit() {
    local rc=$?
    if [ -f "$FLEET" ]; then
        log "cleanup: terminating remaining instances"
        awk '{print $1}' "$FLEET" | xargs -r aws ec2 terminate-instances --region "$REGION" --instance-ids > /dev/null 2>&1 || true
    fi
    exit $rc
}
# (trap installed after MARKER_DIR is created so cleanup deletes it too)

log "watching $(wc -l < "$FLEET" | tr -d ' ') VMs from $FLEET"

# Poll loop: every POLL_INTERVAL seconds, count VMs still running gen.
# Marker files (macOS bash 3.2 doesn't have associative arrays).
MARKER_DIR="$(mktemp -d /tmp/watch-markers.XXXXXX)"
trap "rm -rf '$MARKER_DIR'; cleanup_on_exit" INT TERM EXIT

while true; do
    n_done=0; n_total=0
    while IFS=$' \t' read -r IID IP SEED; do
        n_total=$((n_total + 1))
        if [ -f "$MARKER_DIR/$IID.done" ]; then
            n_done=$((n_done + 1)); continue
        fi
        if ssh -n "${SSH_OPTS[@]}" -o ConnectTimeout=5 "ubuntu@$IP" \
                'tmux has-session -t gen 2>/dev/null' </dev/null 2>/dev/null; then
            : # still running
        else
            log "$IID ($IP) seed=$SEED: gen finished"
            touch "$MARKER_DIR/$IID.done"
            n_done=$((n_done + 1))
            # Pull this VM's shard immediately so we don't lose it if it dies later.
            mkdir -p "$LIB_DIR/games"
            rsync -az -e "ssh -n ${SSH_OPTS[*]}" \
                "ubuntu@$IP:~/library/games/" "$LIB_DIR/games/" >/dev/null 2>&1 || \
                log "  rsync from $IP failed"
            ssh -n "${SSH_OPTS[@]}" "ubuntu@$IP" 'cat /tmp/gen.log' \
                > "$LIB_DIR/runs/recover-${IID}.log" 2>/dev/null || true
        fi
    done < "$FLEET"
    log "[poll] $n_done / $n_total VMs done"
    if [ "$n_done" -eq "$n_total" ]; then break; fi
    sleep "$POLL_INTERVAL"
done

log "all VMs finished. Final rsync sweep ..."
while IFS=$' \t' read -r IID IP SEED; do
    rsync -az -e "ssh -n ${SSH_OPTS[*]}" \
        "ubuntu@$IP:~/library/games/" "$LIB_DIR/games/" >/dev/null 2>&1 || true
done < "$FLEET"

log "terminating fleet"
awk '{print $1}' "$FLEET" | xargs aws ec2 terminate-instances --region "$REGION" --instance-ids > /dev/null
trap - INT TERM

log "regenerating CATALOG.md"
uv run --project "$REPO_ROOT/wm_chess" python "$REPO_ROOT/wm_chess/scripts/catalog.py" --quiet || true

log "DONE."
