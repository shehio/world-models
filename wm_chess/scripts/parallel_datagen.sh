#!/usr/bin/env bash
# Launch N c7i.8xlarge spot instances in parallel, each generating
# games_total/N games of a single dataset config (same depth, multipv, T)
# with different seeds. Bypasses Terraform for fleet management — uses
# raw aws CLI so we can fan-out independent instances cleanly.
#
# Each VM writes to its own library leaf:
#     library/games/sf-<v>/d<D>-mpv<K>-T<T>/g<g_per_vm>-seed<S>/
# After all complete, the shards are merged into:
#     library/games/sf-<v>/d<D>-mpv<K>-T<T>/g<total>-merged/data.npz
#
# Usage:
#   KEY_NAME=mykey bash wm_chess/scripts/parallel_datagen.sh
#
# Required env:
#   KEY_NAME                 AWS EC2 key pair name
#
# Optional env (defaults shown):
#   REGION=us-east-1
#   N_VMS=8
#   TOTAL_GAMES=100000
#   DEPTH=15
#   MULTIPV=8
#   T=1.0
#   SF_VERSION=18           (for path construction; the engine detects its own)
#   BASE_SEED=42
#   INSTANCE_TYPE=c7i.8xlarge
#   MAX_SPOT_PRICE=          (empty = use AWS spot market price)
#   VOLUME_GB=80

set -uo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
LIB_DIR="$REPO_ROOT/library"

: "${KEY_NAME:?Set KEY_NAME=<aws-key-pair-name>}"
: "${REGION:=us-east-1}"
: "${N_VMS:=8}"
: "${TOTAL_GAMES:=100000}"
: "${DEPTH:=15}"
: "${MULTIPV:=8}"
: "${T:=1.0}"
: "${SF_VERSION:=18}"
: "${BASE_SEED:=42}"
: "${INSTANCE_TYPE:=c7i.8xlarge}"
: "${VOLUME_GB:=80}"

KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
[ -f "$KEY_PATH" ] || { echo "FATAL: $KEY_PATH not found"; exit 1; }
SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ConnectTimeout=10)

# Derive paths
T_FMT="$(python3 -c "print(f'{float(\"$T\"):g}')")"
GAMES_PER_VM=$(( TOTAL_GAMES / N_VMS ))
CFG_PATH="sf-${SF_VERSION}/d${DEPTH}-mpv${MULTIPV}-T${T_FMT}"
RUN_TS="$(date -u +%Y%m%dT%H%MZ)"
RUN_DIR="$LIB_DIR/runs/parallel-${RUN_TS}-d${DEPTH}-g${TOTAL_GAMES}"
mkdir -p "$RUN_DIR"
RUNLOG="$RUN_DIR/run.log"
exec > >(tee -a "$RUNLOG") 2>&1

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

cleanup_on_exit() {
    local rc=$?
    if [ -f "$RUN_DIR/instances.txt" ]; then
        log "cleanup: terminating any remaining instances"
        awk '{print $1}' "$RUN_DIR/instances.txt" | xargs -r aws ec2 terminate-instances --region "$REGION" --instance-ids 2>/dev/null || true
    fi
    exit $rc
}
trap cleanup_on_exit EXIT INT TERM

log "config: N_VMS=$N_VMS TOTAL_GAMES=$TOTAL_GAMES (=$GAMES_PER_VM/vm)  depth=$DEPTH mpv=$MULTIPV T=$T"
log "run dir: $RUN_DIR"

# ---------- one-time AWS setup ----------
AMI_ID="$(aws ec2 describe-images --region "$REGION" --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
              "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)"
log "AMI: $AMI_ID"

VPC_ID="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
log "default VPC: $VPC_ID"

# Pick subnets that offer the instance type.
SUBNET_AZS="$(aws ec2 describe-instance-type-offerings --region "$REGION" \
    --location-type availability-zone --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
    --query 'InstanceTypeOfferings[*].Location' --output text)"
# Pick the first AZ's default subnet.
FIRST_AZ="$(echo "$SUBNET_AZS" | awk '{print $1}')"
SUBNET_ID="$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$FIRST_AZ" \
              "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text)"
log "subnet (AZ $FIRST_AZ): $SUBNET_ID"

# Security group: dedicated to parallel runs.
MY_CIDR="$(curl -s -4 ifconfig.me)/32"
SG_NAME="wm-chess-parallel-sg"
SG_ID="$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")"
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID="$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SG_NAME" --description "wm-chess parallel datagen SSH" \
        --vpc-id "$VPC_ID" --query 'GroupId' --output text)"
    log "created SG: $SG_ID"
fi
# Ensure SSH rule for our IP (idempotent).
aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_CIDR" 2>/dev/null || true
log "SG: $SG_ID  (SSH from $MY_CIDR)"

# ---------- launch ----------
USER_DATA=$(cat <<'EOF'
#!/usr/bin/env bash
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y stockfish htop tmux jq rsync
sudo -u ubuntu bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
echo 'export PATH="$HOME/.local/bin:$PATH"' | sudo -u ubuntu tee -a /home/ubuntu/.bashrc
mkdir -p /home/ubuntu/work; chown -R ubuntu:ubuntu /home/ubuntu/work
date -u +"%Y-%m-%dT%H:%M:%SZ bootstrap complete (wm-chess-parallel)" \
  | tee /home/ubuntu/BOOTSTRAP_COMPLETE
chown ubuntu:ubuntu /home/ubuntu/BOOTSTRAP_COMPLETE
EOF
)
USER_DATA_B64="$(printf '%s' "$USER_DATA" | base64 | tr -d '\n')"

> "$RUN_DIR/instances.txt"  # format: instance_id  seed  games

log "launching $N_VMS × $INSTANCE_TYPE spot in $REGION ..."
for i in $(seq 0 $((N_VMS - 1))); do
    SEED=$(( BASE_SEED + i * 1000 ))
    IID="$(aws ec2 run-instances --region "$REGION" \
        --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
        --subnet-id "$SUBNET_ID" \
        --associate-public-ip-address \
        --instance-market-options 'MarketType=spot' \
        --user-data "$USER_DATA_B64" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=wm-chess-parallel},{Key=Name,Value=wm-chess-vm$i-d$DEPTH},{Key=RunTimestamp,Value=$RUN_TS}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1)"
    echo "$IID $SEED $GAMES_PER_VM" >> "$RUN_DIR/instances.txt"
    log "  vm$i: $IID  (seed=$SEED  games=$GAMES_PER_VM)"
done

# Wait for all instances to be running and have public IPs.
log "waiting for all instances to enter 'running' state ..."
INSTANCE_IDS=$(awk '{print $1}' "$RUN_DIR/instances.txt")
aws ec2 wait instance-running --region "$REGION" --instance-ids $INSTANCE_IDS
log "all instances running. Resolving public IPs ..."

> "$RUN_DIR/ips.txt"
for line in $(awk '{print $1"|"$2"|"$3}' "$RUN_DIR/instances.txt"); do
    IID="${line%%|*}"; rest="${line#*|}"
    SEED="${rest%%|*}"; GAMES="${rest##*|}"
    IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
    echo "$IID $IP $SEED $GAMES" >> "$RUN_DIR/ips.txt"
done
log "IPs:"; cat "$RUN_DIR/ips.txt"

# ---------- wait for bootstrap on each ----------
log "waiting for /home/ubuntu/BOOTSTRAP_COMPLETE on all VMs (up to 10 min each) ..."
wait_pids=()
while IFS=' ' read -r IID IP SEED GAMES; do
    (
        deadline=$(( $(date +%s) + 600 ))
        while [ $(date +%s) -lt $deadline ]; do
            if ssh "${SSH_OPTS[@]}" "ubuntu@$IP" 'test -f /home/ubuntu/BOOTSTRAP_COMPLETE' 2>/dev/null; then
                echo "[bootstrap] $IID ($IP) ready"
                exit 0
            fi
            sleep 10
        done
        echo "[bootstrap] $IID ($IP) TIMEOUT"
        exit 1
    ) &
    wait_pids+=($!)
done < "$RUN_DIR/ips.txt"
for p in "${wait_pids[@]}"; do wait "$p" || log "  (one VM failed bootstrap; continuing with the rest)"; done

# ---------- rsync project + launch gen on each in parallel ----------
log "rsyncing project + launching gen on each VM in parallel ..."
gen_pids=()
while IFS=' ' read -r IID IP SEED GAMES; do
    (
        rsync -az -e "ssh ${SSH_OPTS[*]}" \
            --exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache' --exclude='.git/' \
            "$REPO_ROOT/wm_chess/" "ubuntu@$IP:~/wm_chess/" >/dev/null
        rsync -az -e "ssh ${SSH_OPTS[*]}" \
            --exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache' --exclude='data/' \
            "$REPO_ROOT/experiments/distill-soft/" "ubuntu@$IP:~/distill-soft/" >/dev/null
        ssh "${SSH_OPTS[@]}" "ubuntu@$IP" 'cd ~/distill-soft && ~/.local/bin/uv sync' >/dev/null 2>&1
        # nproc - 2 workers, fall back to 1 if tiny
        NPROC=$(ssh "${SSH_OPTS[@]}" "ubuntu@$IP" 'nproc')
        WORKERS=$((NPROC - 2)); [ $WORKERS -lt 1 ] && WORKERS=1
        # Run in tmux so we can survive ssh hiccups; tail the log via ssh later.
        ssh "${SSH_OPTS[@]}" "ubuntu@$IP" "tmux new-session -d -s gen \"
            cd ~/distill-soft && ~/.local/bin/uv run python scripts/generate_data.py \\
                --n-games $GAMES --workers $WORKERS \\
                --depth $DEPTH --multipv $MULTIPV --temperature-pawns $T \\
                --seed $SEED --chunk-size 50 \\
                --library-root /home/ubuntu/library 2>&1 | tee /tmp/gen.log
        \""
        echo "[start] gen launched on $IID ($IP) seed=$SEED games=$GAMES workers=$WORKERS"
    ) &
    gen_pids+=($!)
done < "$RUN_DIR/ips.txt"
for p in "${gen_pids[@]}"; do wait "$p" || log "  (launch on one VM failed)"; done

# ---------- poll until all VMs finish (tmux session 'gen' exits) ----------
log "polling for gen completion on all VMs (loop checks every 5 min) ..."
log "to peek at progress on any VM: ssh ubuntu@<ip> 'tail -30 /tmp/gen.log'"

while true; do
    n_done=0
    n_total=0
    while IFS=' ' read -r IID IP SEED GAMES; do
        n_total=$((n_total + 1))
        if ! ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "ubuntu@$IP" 'tmux has-session -t gen 2>/dev/null' 2>/dev/null; then
            n_done=$((n_done + 1))
        fi
    done < "$RUN_DIR/ips.txt"
    log "[poll] $n_done / $n_total VMs done"
    if [ "$n_done" -eq "$n_total" ]; then break; fi
    sleep 300
done
log "all VMs finished generation"

# ---------- rsync results back ----------
log "rsync library shards back to laptop ..."
mkdir -p "$LIB_DIR/games"
while IFS=' ' read -r IID IP SEED GAMES; do
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$IP:~/library/games/" "$LIB_DIR/games/" || true
    ssh "${SSH_OPTS[@]}" "ubuntu@$IP" 'cat /tmp/gen.log' > "$RUN_DIR/gen_${IID}.log" 2>/dev/null || true
done < "$RUN_DIR/ips.txt"

# ---------- terminate ----------
log "terminating all instances"
INSTANCE_IDS=$(awk '{print $1}' "$RUN_DIR/instances.txt")
aws ec2 terminate-instances --region "$REGION" --instance-ids $INSTANCE_IDS > /dev/null
log "termination requested. Removing the trap so we don't double-terminate."
trap - EXIT INT TERM
rm -f "$RUN_DIR/instances.txt"

# ---------- merge shards into one canonical dataset ----------
MERGED_DIR="$LIB_DIR/games/$CFG_PATH/g${TOTAL_GAMES}-merged-${RUN_TS}"
log "merging shards into $MERGED_DIR"
mkdir -p "$MERGED_DIR"
uv run --project "$REPO_ROOT/wm_chess" python "$THIS_DIR/merge_shards.py" \
    --shards-glob "$LIB_DIR/games/$CFG_PATH/g${GAMES_PER_VM}-seed*" \
    --output "$MERGED_DIR" \
    --multipv "$MULTIPV" --temperature "$T" || log "merge failed; shards still in $LIB_DIR/games/$CFG_PATH/"

# ---------- catalog refresh ----------
log "regenerating CATALOG.md"
uv run --project "$REPO_ROOT/wm_chess" python "$REPO_ROOT/wm_chess/scripts/catalog.py" --quiet

log "DONE. Verify no instances left: aws ec2 describe-instances --region $REGION --filters Name=tag:Project,Values=wm-chess-parallel Name=instance-state-name,Values=running,pending --query 'Reservations[].Instances[].InstanceId' --output text"
log "logs + per-VM gen logs in $RUN_DIR/"
