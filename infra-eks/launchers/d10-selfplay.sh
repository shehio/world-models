#!/usr/bin/env bash
# AlphaZero-style self-play RL on top of the strongest distilled prior.
# This is the "the only lever that can plausibly hit 2,500" experiment
# from /next/. Bootstrapping from d10 ep 15 (sims=4,000 peak = 2,189).
#
# Why d10 ep 15 and not d15 R1 ep 7:
#   - d10 ep 15 (2,189) edges d15 R1 ep 7 (2,146) at sims=4,000 with the
#     CI overlap from the H2H — but they H2H'd 0/104/0, so a tie.
#   - d10 ep 15 is 20×256 which matches the selfplay entrypoint defaults.
#     d15 R1 was 40×256, would need different selfplay hparams + has
#     higher per-iter trainer cost.
#   - Either way, self-play will diverge from the prior over iters;
#     the prior just sets the warm start.
#
# Why this is the high-value experiment:
#   - The d15 46M experiment tied d10 30M. Pure distillation has
#     saturated near 2,150-2,200 Elo.
#   - To exceed the teacher's strength, the student needs to find
#     moves the teacher missed — only self-play search can do that.
#   - Even a modest +50-100 Elo from a 24h selfplay test validates
#     that the loop is non-regressing. If +200+ in 24h, scale up.
#
# What's been fixed since the first attempt (which regressed):
#   - entrypoint-selfplay.sh LR default 1e-3 → 1e-5 (commit on
#     2026-05-25). The original 1e-3 nuked the prior in a few iters.
#   - selfplay_loop_mp.py default LR has always been 1e-5; only the
#     entrypoint was overriding it wrong.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a                # DL Base GPU AL2023 us-east-1
SUBNET=subnet-00be06b9c01d8b036         # us-east-1c
INSTANCE_TYPE=g6e.4xlarge                # 16 vCPU, 128 GB RAM, 1× L40S — fits in remaining us-east-1 OD quota
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

# Prior + output paths
S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d10-mpv8-T1-g250000-20260513T0615Z
INIT_FROM_S3=${INIT_FROM_S3:-s3://$S3_BUCKET/$S3_PREFIX/checkpoints/net-20x256/20260514T2332Z-full30M/distilled_epoch014.pt}
RUN_ID=$(date -u +%Y%m%dT%H%MZ)-selfplay-from-d10ep15

# Selfplay hparams (most are entrypoint-selfplay.sh defaults; called out
# the ones that differ from default for clarity).
N_BLOCKS=20
N_FILTERS=256
SIMS=800
WORKERS=16                  # match g6e.4xlarge vCPU count (half of g6e.8xlarge)
GAMES_PER_WORKER=2    # 2x to keep games-per-iter unchanged
TRAIN_STEPS=200
LR=1e-5                     # fine-tune-grade; 100× lower than distillation
TIME_BUDGET=86400            # 24h initial test
EVAL_EVERY=1
EVAL_GAMES=20
EVAL_SIMS=200

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/selfplay.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/selfplay.log "s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-${N_BLOCKS}x${N_FILTERS}/$RUN_ID/host-launch.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull $ECR/wm-chess-gpu:latest

mkdir -p /mnt/work
# Curl trampoline: the baked image entrypoint-selfplay.sh has the old
# LR=1e-3 default that nuked the first attempt. Pull the fix from main.
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    -e S3_BUCKET=$S3_BUCKET \\
    -e S3_PREFIX=$S3_PREFIX \\
    -e RUN_ID=$RUN_ID \\
    -e INIT_FROM_S3=$INIT_FROM_S3 \\
    -e N_BLOCKS=$N_BLOCKS \\
    -e N_FILTERS=$N_FILTERS \\
    -e SIMS=$SIMS \\
    -e WORKERS=$WORKERS \\
    -e GAMES_PER_WORKER=$GAMES_PER_WORKER \\
    -e TRAIN_STEPS=$TRAIN_STEPS \\
    -e LR=$LR \\
    -e TIME_BUDGET=$TIME_BUDGET \\
    -e EVAL_EVERY=$EVAL_EVERY \\
    -e EVAL_GAMES=$EVAL_GAMES \\
    -e EVAL_SIMS=$EVAL_SIMS \\
    --entrypoint bash \\
    $ECR/wm-chess-gpu:latest -lc '
        curl -sSL https://raw.githubusercontent.com/shehio/world-models/main/infra-eks/entrypoint-selfplay.sh -o /entrypoint-selfplay.sh
        chmod +x /entrypoint-selfplay.sh
        exec /entrypoint-selfplay.sh
    '
EOF
)

aws ec2 run-instances --region $LAUNCH_REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-selfplay-from-d10ep15-od},{Key=role,Value=wm-chess-selfplay}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
