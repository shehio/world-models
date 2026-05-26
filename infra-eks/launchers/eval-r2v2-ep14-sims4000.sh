#!/usr/bin/env bash
# Deep-sims eval on R2 v2 ep 14 — the highest sims=800 score of the whole
# R2 v2 run (2,055 [1979, 2159] vs UCI=1,800). The deep-eval daemon
# never picked this ckpt up because it went offline (operator's laptop
# closed); the sims=4,000 number was never measured.
#
# Hypothesis: ep 14 follows R2 v2's sims=800 → sims=4,000 pattern of
# +150-200 Elo at deep sims, which puts it at 2,200-2,250+. If true, ep
# 14 might be a *tighter-CI* second candidate alongside ep 4's 2,285
# (which had a 95% CI ~±200 wide).
#
# Goal: a second checkpoint for the >2,300 question with less eval noise
# than ep 4. Cost ~$5 on a g6.4xlarge OD; ETA ~2.2h (sims=4,000 + same
# anchor as the existing ep 4 number = direct comparison).

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=${SUBNET:-subnet-042fb3c497e2631a7}
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.2xlarge}
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

CKPT_BASE=s3://wm-chess-library-594561963943/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-20x256/20260525T0935Z-full46M-20x256-cosine
CKPT_S3=$CKPT_BASE/distilled_epoch014.pt
RESULT_KEY=$CKPT_BASE/eval_results-distilled_epoch014-sims4000.txt

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$CKPT_S3"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    aws s3 cp /var/log/eval.log "\${RESULT_KEY%.txt}.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
for i in \$(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null

aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e CKPT_S3=\$CKPT_S3 -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cd /work/experiments/distill-soft
        CKPT_LOCAL=/work-tmp/\$(basename \$CKPT_S3)
        aws s3 cp \$CKPT_S3 \$CKPT_LOCAL --no-progress
        OUT=/work-tmp/eval_results.txt
        echo "=== eval vs Stockfish UCI=1350 (sims=4000) ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 4000 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1350 --agent-device cuda 2>&1 | tee -a \$OUT
        echo "" | tee -a \$OUT
        echo "=== eval vs Stockfish UCI=1800 (sims=4000) ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 4000 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1800 --agent-device cuda 2>&1 | tee -a \$OUT
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-r2v2-ep14-sims4000},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text

echo ""
echo "result: $RESULT_KEY"
echo "log:    ${RESULT_KEY%.txt}.log"
