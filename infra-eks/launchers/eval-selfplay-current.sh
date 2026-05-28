#!/usr/bin/env bash
# One-shot eval against the live selfplay loop's current.pt — gets an
# Elo signal without waiting for the loop's built-in eval cadence.
#
# Usage: bash infra-eks/launchers/eval-selfplay-current.sh
# Override CKPT_S3 / RUN_ID via env for other selfplay runs.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-00be06b9c01d8b036          # us-east-1c (capacity confirmed by Go evals)
INSTANCE_TYPE=g6.xlarge        # g6.4xlarge needs 16 vCPU spot; only 12 free
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

# Default: current selfplay run's current.pt
SELFPLAY_RUN_ID=${SELFPLAY_RUN_ID:-20260525T0453Z-selfplay-from-d10ep15}
CKPT_S3=${CKPT_S3:-s3://wm-chess-library-594561963943/d10-mpv8-T1-g250000-20260513T0615Z/selfplay/net-20x256/$SELFPLAY_RUN_ID/current.pt}
LABEL=${LABEL:-iter0}
# Overridable so runs living under a different prefix (e.g. d15-mpv8 R2v2)
# write their results next to their own checkpoints, not the d10 default.
RESULT_KEY=${RESULT_KEY:-s3://wm-chess-library-594561963943/d10-mpv8-T1-g250000-20260513T0615Z/selfplay/net-20x256/$SELFPLAY_RUN_ID/eval_results-${LABEL}-sims800.txt}

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$CKPT_S3"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    aws s3 cp /var/log/eval.log "\${RESULT_KEY%.txt}.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
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
        OUT=/work-tmp/eval_result.txt
        echo "=== selfplay eval: net-20x256, ckpt=\$CKPT_S3 ===" | tee \$OUT
        echo "=== sims=800 vs Stockfish UCI=1350 ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 4 --games-per-worker 25 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1350 --agent-device cuda 2>&1 | tee -a \$OUT
        echo "" | tee -a \$OUT
        echo "=== sims=800 vs Stockfish UCI=1800 ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 4 --games-per-worker 25 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1800 --agent-device cuda 2>&1 | tee -a \$OUT
        aws s3 cp \$OUT \$RESULT_KEY --no-progress
    '
EOF
)

aws ec2 run-instances --region $LAUNCH_REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-selfplay-eval-${LABEL}},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
