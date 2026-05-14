#!/usr/bin/env bash
# Polls S3 for new checkpoints across all training runs. For each one
# that lacks a sibling eval_results.txt, launches a one-shot g6.xlarge
# EC2 in us-east-1 that:
#   1. Pulls the wm-chess-gpu image from ECR
#   2. Downloads the .pt from S3
#   3. Runs eval.py vs Stockfish 1320 (and optionally vs random)
#   4. Uploads eval_results-<ckpt>.txt next to the ckpt
#   5. shutdown -h now -> instance terminates (initiated-shutdown=terminate)
#
# Idempotent: each ckpt that already has an eval_results-<ckpt>.txt is skipped.
# A claimed.<ckpt> marker prevents two parallel pollers from double-launching.

set -u
LOG=/tmp/wm-autoeval-daemon.log
INTERVAL_SEC=${INTERVAL_SEC:-300}     # 5 min
REGION=us-east-1
ACCOUNT_ID=594561963943
ECR=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
INSTANCE_TYPE=g6.4xlarge   # 16 vCPU lets us run 8-12 parallel game-workers
                            # instead of 4 — roughly 3x faster per ckpt
INSTANCE_PROFILE=wm-chess-merge-instance-profile   # reused from merge step
# Deep Learning Base OSS Nvidia Driver GPU AMI (AL2023) — has docker +
# nvidia-container-toolkit preinstalled, so `docker run --gpus all` works.
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7

# Buckets we watch (extend here when adding new datasets).
BUCKETS=(
    "wm-chess-library-594561963943"
)

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

launch_eval_instance() {
    local ckpt_s3="$1"   # full s3://bucket/prefix/.../distilled_epoch004.pt
    local ckpt_basename
    ckpt_basename=$(basename "$ckpt_s3")
    local ckpt_dir_s3="${ckpt_s3%/*}"          # strip filename
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

    local user_data
    user_data=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$ckpt_s3"
RESULT_KEY="$result_key"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    aws s3 cp /var/log/eval.log "\${RESULT_KEY%.txt}.log" --region $REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e CKPT_S3=\$CKPT_S3 -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$REGION \\
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

    aws ec2 run-instances --region $REGION \
        --image-id $AMI \
        --instance-type $INSTANCE_TYPE \
        --subnet-id $SUBNET \
        --iam-instance-profile Name=$INSTANCE_PROFILE \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
        --instance-initiated-shutdown-behavior terminate \
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-$ckpt_basename},{Key=role,Value=wm-chess-eval}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1 | tee -a "$LOG"
}

log "=== started; poll every ${INTERVAL_SEC}s ==="

while true; do
    for bucket in "${BUCKETS[@]}"; do
        # Find all checkpoints across all run prefixes.
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
