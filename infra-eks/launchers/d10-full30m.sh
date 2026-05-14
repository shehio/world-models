#!/usr/bin/env bash
# Experiment C: train d10 on the FULL 30M positions (vs the 5M subsample
# we've been using). Tests whether the 1750-1800 Elo plateau is the
# dataset-size ceiling. Needs the full ~150GB of states in memory (or
# OS page cache) → uses g6e.8xlarge (256 GB RAM + 1 L40S).
#
# Waits to be launched until d10 (1606Z) finishes — eu-central-1 G/VT
# quota is 32 vCPU and g6e.8xlarge is 32 vCPU all by itself.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6e.8xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d10-mpv8-T1-g250000-20260513T0615Z
RUN_ID=$(date -u +%Y%m%dT%H%MZ)-full30M

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/train.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/train.log "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/net-20x256/$RUN_ID/host-launch.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull $ECR/wm-chess-gpu:latest

mkdir -p /mnt/work
# No MAX_POSITIONS -> the extractor writes the FULL 30M positions.
# Set IN_RAM=1 so we materialise into the 256 GB of system RAM.
# Bigger batch (2048) to keep iteration count reasonable (30M / 2048 ≈ 14.6k batches).
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    -e S3_BUCKET=$S3_BUCKET \\
    -e S3_PREFIX=$S3_PREFIX \\
    -e RUN_ID=$RUN_ID \\
    -e EPOCHS=20 \\
    -e BATCH_SIZE=2048 \\
    -e SAVE_EVERY=5 \\
    -e N_BLOCKS=20 \\
    -e N_FILTERS=256 \\
    -e IN_RAM=1 \\
    -e AMP=1 \\
    -e COMPILE=0 \\
    --entrypoint /entrypoint-train.sh \\
    $ECR/wm-chess-gpu:latest
EOF
)

aws ec2 run-instances --region $LAUNCH_REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=300,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-d10-full30M-eu-central},{Key=role,Value=wm-chess-train}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
