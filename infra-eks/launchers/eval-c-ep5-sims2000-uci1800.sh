#!/usr/bin/env bash
# Sims-curve fill for C ep 5: 2000 sims @ UCI=1800.
#
# We have endpoints already:
#   800 sims  → 2,004 Elo at UCI=1800
#   4000 sims → 2,084 Elo at UCI=1800
# The 2000-sim point lets us draw the Elo-vs-log(sims) curve from one
# end to the other for the d10-30M-ep 5 checkpoint
# (file `distilled_epoch004.pt`, 0-indexed in code, 5th epoch by count).

set -euo pipefail

ACCOUNT_ID=594561963943
LAUNCH_REGION=eu-central-1
IMAGE_REGION=us-east-1
AMI=ami-064d2d57b247fd300
SUBNET=subnet-0a4bed60
INSTANCE_TYPE=g6.4xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

CKPT_S3=s3://wm-chess-library-594561963943/d10-mpv8-T1-g250000-20260513T0615Z/checkpoints/net-20x256/20260514T2332Z-full30M/distilled_epoch004.pt
RESULT_KEY=s3://wm-chess-library-594561963943/d10-mpv8-T1-g250000-20260513T0615Z/checkpoints/net-20x256/20260514T2332Z-full30M/eval_results-distilled_epoch004-sims2000-uci1800.txt

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
        OUT=/work-tmp/eval_results-sims2000-uci1800.txt
        echo "=== C ep 5 vs Stockfish UCI=1800 (sims=2000) ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 2000 --n-blocks 20 --n-filters 256 \\
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
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-c-ep5-sims2000-uci1800},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
