#!/usr/bin/env bash
# Diagnostic: eval the FIRST hourly self-play checkpoint (net_hour_000.pt).
#
# The final.pt eval showed a ~700 Elo regression from the distilled prior
# (1807 → 1116/1262). Question: did this regression hit immediately
# (iter 0's 200 SGD steps already nuked the prior), or accumulate over
# 10 iters?
#
# net_hour_000.pt = network state right after iter 0's training step.
# Eval at sims=800 against UCI=1800 ONLY (skip the easy UCI=1350 leg)
# to keep the test under 30 min on a g6.4xlarge.

set -euo pipefail

ACCOUNT_ID=594561963943
REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6.4xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

SELFPLAY_S3=s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/selfplay/net-20x256/20260515T0157Z
CKPT_S3=$SELFPLAY_S3/net_hour_000.pt
RESULT_KEY=$SELFPLAY_S3/eval_results-hour000.txt

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$CKPT_S3"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    aws s3 cp /var/log/eval.log "\${RESULT_KEY%.txt}.log" --region $REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e CKPT_S3=\$CKPT_S3 -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cd /work/experiments/distill-soft
        CKPT_LOCAL=/work-tmp/\$(basename \$CKPT_S3)
        aws s3 cp \$CKPT_S3 \$CKPT_LOCAL --no-progress
        OUT=/work-tmp/eval_results-hour000.txt
        echo "=== eval net_hour_000.pt vs Stockfish UCI=1800 (sims=800) ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1800 --agent-device cuda 2>&1 | tee -a \$OUT
        aws s3 cp \$OUT \$RESULT_KEY --no-progress
    '
EOF
)

aws ec2 run-instances --region $REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-selfplay-hour000},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
