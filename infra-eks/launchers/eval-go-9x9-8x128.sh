#!/usr/bin/env bash
# Elo eval for the trained Go 8×128 ResNet at /9x9-8x128-20260522T0625Z/
# vs KataGo (calibrated opponent). The training finished 2026-05-22 but
# eval was never run — only the earlier spike (4×64 net, 5K positions)
# was Elo-measured at −147.
#
# Default: ep 19 (the last saved ckpt). Override CKPT_EP for others.
# Default opponent: KataGo at visits=50 (kyu-level, see project memory).
# Default scale: 30 games at student-sims=100. ~20 min wallclock.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-2                   # wm-go image is in us-east-2 ECR
LAUNCH_REGION=us-east-1                  # us-east-1 has more spot pools to try
AMI=ami-027c3ae8019fc0d3a                # DL Base GPU AL2023 us-east-1
SUBNET=subnet-00be06b9c01d8b036          # us-east-1c (selfplay launched there fine)
INSTANCE_TYPE=g6.xlarge                  # 4 vCPU + 24 GB + L4 — plenty for a 30-game eval
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=go-9x9-mpv8-gpu-v400-20260521T1230Z
RUN_ID=9x9-8x128-20260522T0625Z
CKPT_EP=${CKPT_EP:-019}                  # default to last ckpt; override for others
CKPT_S3=s3://$S3_BUCKET/$S3_PREFIX/checkpoints/$RUN_ID/distilled_epoch${CKPT_EP}.pt

GAMES=${GAMES:-30}
STUDENT_SIMS=${STUDENT_SIMS:-100}
KATAGO_VISITS=${KATAGO_VISITS:-50}
RESULT_KEY=s3://$S3_BUCKET/$S3_PREFIX/checkpoints/$RUN_ID/eval_results-distilled_epoch${CKPT_EP}-vs-katago-v${KATAGO_VISITS}.txt

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$CKPT_S3"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-go-gpu:latest"

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
    -e CKPT_S3=\$CKPT_S3 \\
    -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cd /work/experiments/distill-go
        CKPT_LOCAL=/work-tmp/\$(basename \$CKPT_S3)
        aws s3 cp \$CKPT_S3 \$CKPT_LOCAL --no-progress
        OUT=/work-tmp/go_eval_result.txt
        echo "=== Go 9x9 eval: 8x128 ResNet vs KataGo (visits=$KATAGO_VISITS) ===" | tee \$OUT
        echo "ckpt: \$CKPT_S3" | tee -a \$OUT
        echo "student-sims: $STUDENT_SIMS  katago-visits: $KATAGO_VISITS  games: $GAMES" | tee -a \$OUT
        echo "" | tee -a \$OUT
        uv run python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL \\
            --board-size 9 --n-input-planes 4 \\
            --n-blocks 8 --n-filters 128 \\
            --student-sims $STUDENT_SIMS \\
            --katago-visits $KATAGO_VISITS \\
            --games $GAMES \\
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
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=80,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-go-eval-ep${CKPT_EP}-katago-v${KATAGO_VISITS}-spot},{Key=role,Value=wm-go-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
