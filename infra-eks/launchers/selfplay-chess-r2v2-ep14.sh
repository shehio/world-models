#!/usr/bin/env bash
# AlphaZero-style self-play RL on top of R2v2 ep 14 — the project's
# strongest measurement (2,301 Elo [2,190, 2,601] at sims=4,000 vs
# Stockfish UCI=1,800, the first chess measurement with a 95% lower
# CI bound above 2,300).
#
# This is the "the only lever that can plausibly hit 2,500" experiment
# from /next/, run on OD this time. The d10-selfplay.sh variant ran six
# times on spot and was evicted before iter 1 finished every single
# time. Iters take ~90 min on g6e.8xlarge with 32 workers; spot
# eviction windows in us-east-1 are shorter than that. OD removes the
# bottleneck completely.
#
# Cost: g6e.8xlarge OD in eu-central-1 ≈ $2.69/h × 24h ≈ $65. Worth
# the 3× premium over spot — six failed attempts is the proof.
#
# Why eu-central-1: R1 v2 ran here on g6e.8xlarge OD for 36h without
# issue. Quota is empty as of launch time (32 vCPU available).
# Cross-region: ECR (wm-chess-gpu image) lives in us-east-1; bucket
# is us-east-1; only the EC2 host is in eu-central-1. Image pull is
# the only cross-region operation and is cached after the first iter.
#
# Why R2v2 ep 14 prior: the highest-CI-bound chess ckpt we have. Per
# /experiments/#d10-vs-d15 the project's distillation peak. Self-play
# on top of it tests whether RL can find moves the d15 teacher missed
# — exactly the Lc0 recipe.

set -euo pipefail

ACCOUNT_ID=594561963943
ECR_REGION=us-east-1               # wm-chess-gpu lives here
S3_REGION=us-east-1                # bucket here
LAUNCH_REGION=eu-central-1         # OD G quota fully free here
AMI=${AMI:-ami-01e9d13d4c5e54237}  # DL Base GPU AL2023 eu-central-1 (same as d15-full30m)
SUBNET=${SUBNET:-subnet-0a4bed60}  # eu-central-1a default (the proven-working one)
INSTANCE_TYPE=${INSTANCE_TYPE:-g6e.8xlarge}
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$ECR_REGION.amazonaws.com

# Prior + output paths
S3_BUCKET=wm-chess-library-594561963943
# R2v2 ep 14 — the d15 46M 20×256 cosine ckpt that landed 2,301 at sims=4000.
INIT_FROM_S3=${INIT_FROM_S3:-s3://$S3_BUCKET/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-20x256/20260525T0935Z-full46M-20x256-cosine/distilled_epoch014.pt}
S3_PREFIX=d15-mpv8-T1-g250000-20260519T0412Z
RUN_ID=$(date -u +%Y%m%dT%H%MZ)-selfplay-from-r2v2-ep14-OD

# Selfplay hparams. Most are entrypoint-selfplay.sh defaults; called out
# the ones that differ for clarity.
N_BLOCKS=20
N_FILTERS=256
SIMS=800
# g6e.8xlarge has 32 vCPU. One trainer process uses the GPU; the rest
# run MCTS workers. 30 workers × 4 games/worker = 120 games/iter
# (vs 32 games/iter on the previous g6.4xlarge attempts).
WORKERS=30
GAMES_PER_WORKER=4
TRAIN_STEPS=200
# LR=1e-5 — the value that preserved the prior on the prior spot
# attempts (sims=800 vs UCI=1800 went from 2034 seed → 1996/2034/1885
# at iter 0 across various retries, no catastrophic forgetting).
# LR=1e-4 caused catastrophic forgetting on the v3 attempt (sims=800
# dropped from 2034 → 1643 in one iter).
LR=1e-5
TIME_BUDGET=86400         # 24h
EVAL_EVERY=1
EVAL_GAMES=20
EVAL_SIMS=200

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/selfplay.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/selfplay.log "s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-${N_BLOCKS}x${N_FILTERS}/$RUN_ID/host-launch.log" --region $S3_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
for i in \$(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null

aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR
docker pull $ECR/wm-chess-gpu:latest

mkdir -p /mnt/work
# Curl trampoline: the baked image entrypoint-selfplay.sh history has had
# wrong LR defaults that nuked priors. Pull the latest from main so the
# fix doesn't depend on an image rebuild.
docker run --rm \\
    --gpus all \\
    --shm-size=4g \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$S3_REGION \\
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
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=300,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-selfplay-r2v2-ep14-OD-${RUN_ID}},{Key=role,Value=wm-chess-selfplay}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "run id:        $RUN_ID"
echo "ckpts at:      s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-${N_BLOCKS}x${N_FILTERS}/$RUN_ID/"
echo "host log at:   s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-${N_BLOCKS}x${N_FILTERS}/$RUN_ID/host-launch.log"
