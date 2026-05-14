#!/usr/bin/env bash
# Experiment E: re-eval d15 ep19 with sims=4000 (5x deeper MCTS than the
# routine 800-sim evals). Tests whether the model is actually stronger
# than our daily evals show — if 4000-sim Elo is meaningfully higher than
# 800-sim, the model is "under-read" by routine eval depth.
#
# Uses the same daemon launch pattern but with --sims 4000.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6.4xlarge   # 16 vCPU, 8 workers
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

CKPT_S3=s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/checkpoints/net-20x256/20260514T1332Z/distilled_epoch019.pt
# Write to a distinct filename so the daemon's idempotency check (looking
# for eval_results-distilled_epoch019.txt) still skips the slow path.
RESULT_KEY=s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/checkpoints/net-20x256/20260514T1332Z/eval_results-distilled_epoch019-sims4000.txt

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
        OUT=/work-tmp/eval_results-sims4000.txt
        echo "=== eval vs Stockfish UCI=1350 (sims=4000) ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 4000 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1350 --agent-device cuda 2>&1 | tee -a \$OUT
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
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-sims4000-d15ep19},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
