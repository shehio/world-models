#!/usr/bin/env bash
# Train the 20x256 ResNet on the FULL d15 250K dataset (no MAX_POSITIONS cap).
# Equivalent to d10-full30m.sh but with the d15 teacher (~2500 Elo) instead
# of d10 (~2200 Elo). Tests whether scaling the teacher quality on top of
# full dataset breaks past the d10 30M-trained 2,171-Elo result.
#
# Needs the full ~30M positions in RAM (or OS page cache) → g6e.8xlarge
# (256 GB RAM + 1 L40S). Same launch shape as d10-full30m.sh.
#
# Prereq: the d15 250K datagen (cluster wm-chess-gen-d15-250k) has finished
# AND the merge Job has produced merged/data.npz under the S3 prefix below.
# Check with:
#   aws s3 ls s3://wm-chess-library-594561963943/d15-mpv8-T1-g250000-20260519T0412Z/merged/

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a               # DL Base GPU AL2023 us-east-1
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6e.8xlarge                # 32 vCPU, 256 GB RAM, 1× L40S
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d15-mpv8-T1-g250000-20260519T0412Z
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
# No MAX_POSITIONS -> the extractor writes the FULL ~30M positions.
# IN_RAM=1 so we materialise into the 256 GB of system RAM (page cache
# alone would work but explicit IN_RAM avoids the first-epoch I/O penalty).
# BATCH_SIZE=2048 keeps iteration count reasonable (30M / 2048 ≈ 14.6k batches).
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-d15-full30M-us-east},{Key=role,Value=wm-chess-train}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
