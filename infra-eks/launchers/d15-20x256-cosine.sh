#!/usr/bin/env bash
# R2 cosine variant — same as d15-20x256-lowlr.sh (20×256 + LR=5e-4 +
# wd=1e-3 + batch=2048 + 30 epochs) but adds the cosine LR schedule
# that train.py learned on 2026-05-24 (commit a71f5b9).
#
# Why: R2 (constant LR=5e-4) tied R1 (constant LR=1e-3) at sims=4,000
# (R2 ep 6 = 2,123 vs R1 ep 7 = 2,146, CIs overlap). Both fell ~50 Elo
# short of d10's 2,189 peak. The R1 v2 cosine retry tests whether
# cosine fixes the 40×256 plateau; *this* run is the symmetric test
# for the 20×256 net. If R2 v2 also beats R2 v1, constant-LR was
# bottlenecking distillation at every architecture, not just 40×256.
#
# Schedule:
#   - 3-epoch linear warmup: LR ramps 1e-5 → 5e-4
#   - 27-epoch cosine decay: 5e-4 → 1e-5
# Total 30 epochs, same as R2 v1.
#
# Region: us-east-1a (use1-az2) spot — best L40S spot score (3/10) when
# launched. eu-central-1 OD held by R1 v2 cosine; us-east-1 OD half-held
# by a sims=4000 autoeval. Spot is the only available pool right now.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=ap-northeast-1
AMI=ami-0c11e8f0937c0bf36              # ap-northeast-1 DLAMI
SUBNET=subnet-597e8511             # ap-northeast-1a
INSTANCE_TYPE=g6e.8xlarge                # 32 vCPU, 256 GB RAM, 1× L40S
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d15-mpv8-T1-g250000-20260519T0412Z
RUN_ID=$(date -u +%Y%m%dT%H%MZ)-full46M-20x256-cosine

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
# Curl trampoline because the baked image was built before cosine-LR
# support landed in train.py + entrypoint-train.sh.
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
    -e LR_SCHEDULER=cosine \\
    -e WARMUP_EPOCHS=3 \\
    -e LR_MIN=1e-5 \\
    --entrypoint bash \\
    $ECR/wm-chess-gpu:latest -lc '
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-d15-full46M-20x256-cosine-spot},{Key=role,Value=wm-chess-train}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
