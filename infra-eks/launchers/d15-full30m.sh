#!/usr/bin/env bash
# Train a 40x256 ResNet (~50M params) on the FULL d15 dataset (45.9M positions,
# 427K games). Pairs the strongest teacher (d15 ~2500 Elo) with both more data
# than d10 (45.9M vs 30M) and more capacity than the 20x256 baseline. Goal:
# break past d10 30M-trained 2,171 Elo.
#
# Differences vs d10-full30m baseline:
#   - 40x256 net (was 20x256) — 2x compute/sample, ~50M params
#   - 40 epochs (was 20) — more passes for the deeper net + harder signal
#   - BATCH_SIZE=1024 (was 2048) — safer for 40-block on L40S 48GB VRAM
#   - SAVE_EVERY=1 (was 5) — per-epoch Elo curve + minimal loss on spot evict
#   - SPOT (was on-demand) — ~25% cheaper, resume-from-S3 covers evictions
#
# Needs the full 45.9M positions in RAM → g6e.8xlarge (256 GB RAM + 1 L40S).
#
# Prereq: the d15 250K datagen has produced merged/data.npz. Confirmed
# 2026-05-22 16:45 UTC, 45.9M positions, 427K games.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1                   # ECR + S3 stay in us-east-1
LAUNCH_REGION=us-east-1                  # spot quota free again; eu-central-1c sold out
AMI=ami-027c3ae8019fc0d3a                # DL Base GPU AL2023 us-east-1
SUBNET=subnet-042fb3c497e2631a7          # us-east-1b (use1-az4) — us-east-1d sold out by 22:51 UTC 5/24
INSTANCE_TYPE=g6e.8xlarge                # 32 vCPU, 256 GB RAM, 1× L40S
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d15-mpv8-T1-g250000-20260519T0412Z
# Pinned RUN_ID for the cosine-LR run. The original constant-LR run
# (20260522T2229Z-full46M-40x256) plateaued by epoch 7. This is the
# fresh-restart variant with linear warmup + cosine decay testing
# whether LR scheduling escapes that plateau. Pin so the entrypoint
# auto-resumes from the latest S3 ckpt on eviction.
RUN_ID=20260524T0955Z-full46M-40x256-cosine

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/train.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/train.log "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/net-40x256/$RUN_ID/host-launch.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull $ECR/wm-chess-gpu:latest

mkdir -p /mnt/work
# No MAX_POSITIONS -> the extractor writes the FULL 45.9M positions.
# IN_RAM=1 so we materialise into the 256 GB of system RAM (page cache
# alone would work but explicit IN_RAM avoids the first-epoch I/O penalty).
# BATCH_SIZE=1024 keeps activations within L40S VRAM at 40 blocks
# (45.9M / 1024 ≈ 44.8k batches/epoch).
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    -e S3_BUCKET=$S3_BUCKET \\
    -e S3_PREFIX=$S3_PREFIX \\
    -e RUN_ID=$RUN_ID \\
    -e EPOCHS=40 \\
    -e BATCH_SIZE=1024 \\
    -e SAVE_EVERY=1 \\
    -e N_BLOCKS=40 \\
    -e N_FILTERS=256 \\
    -e IN_RAM=1 \\
    -e AMP=1 \\
    -e COMPILE=0 \\
    -e LR_SCHEDULER=cosine \\
    -e WARMUP_EPOCHS=3 \\
    -e LR_MIN=1e-5 \\
    --entrypoint bash \\
    $ECR/wm-chess-gpu:latest -lc '
        # The baked image was built before cosine-LR support landed
        # (commit on 2026-05-24). Pull the latest entrypoint + train.py
        # from main so this run gets the new schedule without waiting
        # on a CodeBuild rebuild. Drop this once the image is rebuilt.
        curl -sSL https://raw.githubusercontent.com/shehio/world-models/main/infra-eks/entrypoint-train.sh -o /entrypoint-train.sh
        curl -sSL https://raw.githubusercontent.com/shehio/world-models/main/experiments/distill-soft/scripts/train.py -o /work/experiments/distill-soft/scripts/train.py
        chmod +x /entrypoint-train.sh
        exec /entrypoint-train.sh
    '
EOF
)

aws ec2 run-instances --region $LAUNCH_REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=300,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-d15-full46M-40x256-spot},{Key=role,Value=wm-chess-train}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
