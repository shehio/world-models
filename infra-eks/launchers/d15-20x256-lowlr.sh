#!/usr/bin/env bash
# Second d15 training experiment. Hypothesis: the d15 40x256/LR=1e-3 run
# plateaued (peak Elo 1892 at ep1, then declining) because both the capacity
# AND the LR are too aggressive for the d15 soft-target signal.
#
# Differences vs d15-full30m (the 40x256 run still in flight):
#   - 20x256 net (was 40x256) — matches the d10 baseline shape that hit 2,064
#   - LR 5e-4 (was 1e-3) — half the rate, less likely to bounce out of optima
#   - weight_decay 1e-3 (was 1e-4) — 10x more L2 regularization
#   - BATCH 2048 (was 1024) — matches d10 baseline, fits comfortably for 20-block
#   - EPOCHS 30 (was 40) — 20-block converges faster, no need for 40
#   - SAVE_EVERY 1 (same) — per-epoch eval curve for direct comparison
#   - ON-DEMAND eu-central-1 (was us-east-1 spot) — us-east-1 spot quota
#     held by the parallel 40x256 run, and us-east-1 OD G/VT held by the
#     two concurrent eval EC2s (ep5 sims=800 + ep1 sims=4000). eu-central-1
#     OD G/VT is a separate quota pool with 32 vCPU free. Cross-region S3
#     pull adds ~3 min on the 3.7 GB dataset download.
#
# Cost: 20-block is ~half the compute/epoch of 40-block. 30 epochs × ~1h
# ≈ 30h training + 1h boot/data = ~31h total wallclock, ~$70 on-demand.
# Runs in parallel with the 40x256 run; ckpts go to a different S3 subdir
# (net-20x256/) so no collision.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=eu-central-1
AMI=ami-01e9d13d4c5e54237               # DL Base GPU AL2023 eu-central-1
SUBNET=subnet-0a4bed60                   # eu-central-1a default
INSTANCE_TYPE=g6e.8xlarge                # 32 vCPU, 256 GB RAM, 1× L40S
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com  # ECR stays us-east-1, cross-region pull

S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d15-mpv8-T1-g250000-20260519T0412Z
RUN_ID=$(date -u +%Y%m%dT%H%MZ)-full46M-20x256-lowlr

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
# No MAX_POSITIONS -> the extractor writes the FULL 45.9M positions.
# IN_RAM=1 materialises into the 256 GB system RAM (matches the 228 GB
# footprint observed on the 40x256 run).
# LR=5e-4 + WEIGHT_DECAY=1e-3 are the regularization-leaning tweaks.
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    -e S3_BUCKET=$S3_BUCKET \\
    -e S3_PREFIX=$S3_PREFIX \\
    -e RUN_ID=$RUN_ID \\
    -e EPOCHS=30 \\
    -e BATCH_SIZE=2048 \\
    -e LR=5e-4 \\
    -e WEIGHT_DECAY=1e-3 \\
    -e SAVE_EVERY=1 \\
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-d15-full46M-20x256-lowlr-od},{Key=role,Value=wm-chess-train}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
