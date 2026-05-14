#!/usr/bin/env bash
# Polls S3 for new checkpoints across all training runs. For each one
# that lacks a sibling eval_results.txt, launches a one-shot g6.4xlarge
# EC2 that:
#   1. Pulls the wm-chess-gpu image from ECR (always us-east-1, cross-region
#      pull works when the EC2 lives elsewhere)
#   2. Downloads the .pt from S3 (always us-east-1 bucket, cross-region read)
#   3. Runs eval.py vs Stockfish 1350 and depth=1 sub-tests
#   4. Uploads eval_results-<ckpt>.txt next to the ckpt
#   5. shutdown -h now -> instance terminates (initiated-shutdown=terminate)
#
# Multi-region: us-east-1 is the primary launch region; eu-central-1 is the
# fallback when us-east-1 hits the 32-vCPU G/VT quota. Each region has its
# own independent vCPU budget, so this doubles concurrent-eval capacity.
#
# Idempotent: each ckpt with an eval_results-<ckpt>.txt sibling is skipped.
# A claimed.<ckpt> marker prevents two pollers from double-launching; the
# marker is removed automatically on RunInstances failure.

set -u
LOG=/tmp/wm-autoeval-daemon.log
INTERVAL_SEC=${INTERVAL_SEC:-300}     # 5 min
ACCOUNT_ID=594561963943

# Where the ECR image + S3 buckets live. EC2s in any LAUNCH_REGION
# cross-region-pull the image and cross-region-read the buckets.
IMAGE_REGION=us-east-1
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

INSTANCE_TYPE=g6.4xlarge   # 16 vCPU, 8 workers × 13 games ≈ 3x faster
INSTANCE_PROFILE=wm-chess-merge-instance-profile   # IAM is global

# Per-region settings — using a case in `region_config` instead of
# associative arrays so this works on macOS bash 3.2 (no `declare -A`).
# Try us-east-1 first; if it fails (e.g. VcpuLimitExceeded), retry in
# eu-central-1 before giving up.
REGION_ORDER=(us-east-1 eu-central-1)

region_config() {
    # echoes "<subnet-id> <ami-id>" for the given region
    case "$1" in
        us-east-1)
            echo "subnet-042fb3c497e2631a7 ami-027c3ae8019fc0d3a" ;;
        eu-central-1)
            echo "subnet-0a4bed60 ami-01e9d13d4c5e54237" ;;
        *)
            echo "" ;;
    esac
}

# Buckets we watch (always us-east-1 — cross-region reads are fine).
BUCKETS=(
    "wm-chess-library-594561963943"
)

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Try launching an eval EC2 in $region. On success: echo instance id.
# On failure: log + leave caller to decide whether to retry elsewhere.
try_launch_in_region() {
    local region="$1"
    local ckpt_s3="$2"
    local result_key="$3"
    local ckpt_basename="$4"
    local cfg
    cfg=$(region_config "$region")
    local subnet="${cfg%% *}"
    local ami="${cfg##* }"

    local user_data
    user_data=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$ckpt_s3"
RESULT_KEY="$result_key"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    # Log + result both land in us-east-1 (single source of truth).
    aws s3 cp /var/log/eval.log "\${RESULT_KEY%.txt}.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
# ECR is in us-east-1 regardless of where this EC2 is running.
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e CKPT_S3=\$CKPT_S3 -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cd /work/experiments/distill-soft
        CKPT_LOCAL=/work-tmp/\$(basename \$CKPT_S3)
        aws s3 cp \$CKPT_S3 \$CKPT_LOCAL --no-progress
        OUT=/work-tmp/eval_results.txt
        echo "=== eval vs Stockfish UCI=1350 ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1350 --agent-device cuda 2>&1 | tee -a \$OUT
        echo "" | tee -a \$OUT
        echo "=== eval vs Stockfish UCI=-1 (top skill, anchor=none) ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo -1 --stockfish-depth 1 --agent-device cuda 2>&1 | tee -a \$OUT || echo "(deep stockfish failed; skipping)" >> \$OUT
        aws s3 cp \$OUT \$RESULT_KEY --no-progress
    '
EOF
)

    local launch_out
    launch_out=$(aws ec2 run-instances --region "$region" \
        --image-id "$ami" \
        --instance-type $INSTANCE_TYPE \
        --subnet-id "$subnet" \
        --iam-instance-profile Name=$INSTANCE_PROFILE \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
        --instance-initiated-shutdown-behavior terminate \
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-$ckpt_basename},{Key=role,Value=wm-chess-eval}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1)
    local launch_rc=$?
    if [ $launch_rc -eq 0 ] && ! echo "$launch_out" | grep -q 'An error occurred'; then
        echo "$launch_out"
        return 0
    fi
    log "  $region launch failed: $launch_out"
    return 1
}

launch_eval_instance() {
    local ckpt_s3="$1"
    local ckpt_basename
    ckpt_basename=$(basename "$ckpt_s3")
    local ckpt_dir_s3="${ckpt_s3%/*}"
    local result_key="$ckpt_dir_s3/eval_results-${ckpt_basename%.pt}.txt"
    local claim_key="$ckpt_dir_s3/.claimed-eval-${ckpt_basename%.pt}"

    # Idempotency: skip if result OR a claim exists.
    if aws s3 ls "$result_key" >/dev/null 2>&1; then
        log "SKIP $ckpt_s3 — eval_results already exists"
        return
    fi
    if aws s3 ls "$claim_key" >/dev/null 2>&1; then
        log "SKIP $ckpt_s3 — another runner has claimed it"
        return
    fi
    echo "claimed-by=$(hostname) at=$(date -u +%FT%TZ)" | \
        aws s3 cp - "$claim_key" 2>&1 | tail -1

    log "LAUNCH eval EC2 for $ckpt_s3"

    # Walk the region list; first success wins.
    local instance_id=""
    for region in "${REGION_ORDER[@]}"; do
        if instance_id=$(try_launch_in_region "$region" "$ckpt_s3" "$result_key" "$ckpt_basename"); then
            log "  launched $instance_id in $region"
            return 0
        fi
    done

    # All regions failed — release the claim so the next poll retries.
    log "  all regions failed, removing claim marker for retry"
    aws s3 rm "$claim_key" 2>&1 | tail -1 | tee -a "$LOG"
}

log "=== started; poll every ${INTERVAL_SEC}s ==="

while true; do
    for bucket in "${BUCKETS[@]}"; do
        ckpts=$(aws s3 ls "s3://$bucket/" --recursive 2>/dev/null \
                | awk '{print $NF}' \
                | grep -E 'checkpoints/net-[^/]+/[^/]+/distilled_epoch[0-9]+\.pt$' \
                | sort -u)
        for key in $ckpts; do
            launch_eval_instance "s3://$bucket/$key"
        done
    done
    sleep "$INTERVAL_SEC"
done
