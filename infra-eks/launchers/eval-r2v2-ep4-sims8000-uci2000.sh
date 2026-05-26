#!/usr/bin/env bash
# Deep-sims eval on the d15-46M R2 v2 (20×256 cosine) best ckpt — ep 4.
# At sims=4000 / UCI=1800 it scored 2,285 [2,177, 2,554] (104 games), the
# project's top point estimate but with a 95% CI ~±200 wide because the
# score (0.942) is near saturation.
#
# Going to sims=8000 + bumping the anchor to UCI=2000:
#   - sims=8000 tests the "+100–200 Elo from deeper search" hypothesis on
#     top of the already-deep sims=4000 number.
#   - UCI=2000 puts the score closer to 0.5 → tighter Elo CI than UCI=1800
#     where the model was already winning 94% of games.
#
# Goal: a confident yes/no on the ≥2,300 Elo question. Cost ~$5 on a
# g6.4xlarge OD; ETA ~3–4h (sims=8000 ≈ 2× sims=4000 wallclock per game,
# the existing sims=4000 UCI=1800 eval was 130 min).

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6.4xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

CKPT_BASE=s3://wm-chess-library-594561963943/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-20x256/20260525T0935Z-full46M-20x256-cosine
CKPT_S3=$CKPT_BASE/distilled_epoch004.pt
RESULT_KEY=$CKPT_BASE/eval_results-distilled_epoch004-sims8000-uci2000.txt

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
        echo "=== eval vs Stockfish UCI=2000 (sims=8000) ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 8000 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 2000 --agent-device cuda 2>&1 | tee -a \$OUT
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-r2v2-ep4-sims8000-uci2000},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text

echo ""
echo "result: $RESULT_KEY"
echo "log:    ${RESULT_KEY%.txt}.log"
