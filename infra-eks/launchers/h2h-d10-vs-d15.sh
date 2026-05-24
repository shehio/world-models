#!/usr/bin/env bash
# Head-to-head: best d10 ckpt vs best d15 ckpt, both using MCTS at the
# same sim count. Uses experiments/selfplay/scripts/h2h_mp.py (extended
# to support per-agent --n-blocks/--n-filters).
#
# Why: the vs-Stockfish numbers say d10 ep15 (2189 sims=4000) is slightly
# ahead of d15 ep7 (2146 sims=4000), but the CIs overlap heavily. Putting
# the two networks directly against each other removes the Stockfish
# anchor noise and answers the question with a single score (out of N
# games) — much more informative per dollar than two more Stockfish evals.
#
# Defaults:
#   A = d10 30M 20×256 ep14 (file)   = ep15 display = sims=4000 peak 2189
#   B = d15 46M 40×256 ep06 (file)   = ep07 display = sims=4000 best 2146
# Override via CKPT_A_S3, CKPT_B_S3, NB_A, NF_A, NB_B, NF_B env vars to
# re-run later against d15's final peak when training completes.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a               # DL Base GPU AL2023 us-east-1
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6.4xlarge                # 16 vCPU, 1× L4 — enough for sims=4000 h2h
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

# Defaults are the current best ckpts; override via env when d15 finishes.
S3_BUCKET=wm-chess-library-594561963943
CKPT_A_S3=${CKPT_A_S3:-s3://$S3_BUCKET/d10-mpv8-T1-g250000-20260513T0615Z/checkpoints/net-20x256/20260514T2332Z-full30M/distilled_epoch014.pt}
CKPT_B_S3=${CKPT_B_S3:-s3://$S3_BUCKET/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-40x256/20260522T2229Z-full46M-40x256/distilled_epoch006.pt}
NB_A=${NB_A:-20}    # d10 net = 20 blocks
NF_A=${NF_A:-256}   #         × 256 channels
NB_B=${NB_B:-40}    # d15 R1 net = 40 blocks
NF_B=${NF_B:-256}   #             × 256 channels

SIMS=${SIMS:-4000}              # same depth as both peaks were measured at
WORKERS=${WORKERS:-8}            # 8 procs, each batch K=8 → 64 in-flight requests
GAMES_PER_WORKER=${GAMES_PER_WORKER:-13}   # 8 × 13 = 104 games total
MAX_PLIES=${MAX_PLIES:-200}

RUN_ID=$(date -u +%Y%m%dT%H%MZ)-h2h-d10ep15-vs-d15ep07-sims${SIMS}
RESULT_KEY=s3://$S3_BUCKET/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-40x256/20260522T2229Z-full46M-40x256/h2h-${RUN_ID}.txt

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/h2h.log) 2>&1
set -x

CKPT_A_S3="$CKPT_A_S3"
CKPT_B_S3="$CKPT_B_S3"
NB_A="$NB_A"; NF_A="$NF_A"
NB_B="$NB_B"; NF_B="$NF_B"
SIMS="$SIMS"
WORKERS="$WORKERS"
GAMES_PER_WORKER="$GAMES_PER_WORKER"
MAX_PLIES="$MAX_PLIES"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    aws s3 cp /var/log/h2h.log "\${RESULT_KEY%.txt}.log" --region $IMAGE_REGION 2>&1 || true
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
    -e CKPT_A_S3=\$CKPT_A_S3 -e CKPT_B_S3=\$CKPT_B_S3 \\
    -e NB_A=\$NB_A -e NF_A=\$NF_A -e NB_B=\$NB_B -e NF_B=\$NF_B \\
    -e SIMS=\$SIMS -e WORKERS=\$WORKERS -e GAMES_PER_WORKER=\$GAMES_PER_WORKER \\
    -e MAX_PLIES=\$MAX_PLIES -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        # Pull the latest h2h_mp.py from main — the baked image was built
        # before per-agent --n-blocks-a/-b were added (commit 1bfaf67).
        curl -sSL https://raw.githubusercontent.com/shehio/world-models/main/experiments/selfplay/scripts/h2h_mp.py \
             -o /work/experiments/selfplay/scripts/h2h_mp.py
        CKPT_A_LOCAL=/work-tmp/\$(basename \$CKPT_A_S3 .pt)-A.pt
        CKPT_B_LOCAL=/work-tmp/\$(basename \$CKPT_B_S3 .pt)-B.pt
        aws s3 cp \$CKPT_A_S3 \$CKPT_A_LOCAL --no-progress
        aws s3 cp \$CKPT_B_S3 \$CKPT_B_LOCAL --no-progress
        OUT=/work-tmp/h2h_result.txt
        cd /work/experiments/selfplay
        echo "=== head-to-head: A (d10) vs B (d15) sims=$SIMS ===" | tee \$OUT
        echo "A: \$CKPT_A_S3 (\${NB_A}×\${NF_A})" | tee -a \$OUT
        echo "B: \$CKPT_B_S3 (\${NB_B}×\${NF_B})" | tee -a \$OUT
        echo "" | tee -a \$OUT
        python scripts/h2h_mp.py \\
            --ckpt-a \$CKPT_A_LOCAL --ckpt-b \$CKPT_B_LOCAL \\
            --workers \$WORKERS --games-per-worker \$GAMES_PER_WORKER \\
            --sims \$SIMS --batch-size 8 \\
            --n-blocks-a \$NB_A --n-filters-a \$NF_A \\
            --n-blocks-b \$NB_B --n-filters-b \$NF_B \\
            --max-plies \$MAX_PLIES \\
            --device cuda 2>&1 | tee -a \$OUT
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-h2h-d10-vs-d15-sims${SIMS}},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
