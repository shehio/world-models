#!/usr/bin/env bash
# Calibration match: KataGo@v200 vs Pachi on 9x9. Anchors KataGo@v200's
# absolute Elo to Pachi (~KGS 1d ≈ 2100 Elo at 5000 playouts).
#
# GNU Go was too weak (KataGo@v200 swept 100/0). Pachi is the right
# strength tier to give a measurable head-to-head.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-2
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=${SUBNET:-subnet-042fb3c497e2631a7}  # us-east-1b default
INSTANCE_TYPE=g6.xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
GAMES=${GAMES:-100}
KATAGO_VISITS=${KATAGO_VISITS:-200}
PACHI_PLAYOUTS=${PACHI_PLAYOUTS:-5000}
PACHI_ANCHOR_ELO=${PACHI_ANCHOR_ELO:-2100}

STAMP=$(date -u +%Y%m%dT%H%MZ)
RESULT_KEY=s3://$S3_BUCKET/go-calibration/pachi-katago-v${KATAGO_VISITS}-vs-pachi-${PACHI_PLAYOUTS}p-${STAMP}.txt
LOG_KEY=s3://$S3_BUCKET/go-calibration/pachi-katago-v${KATAGO_VISITS}-vs-pachi-${PACHI_PLAYOUTS}p-${STAMP}.log
CALIB_RAW=https://raw.githubusercontent.com/shehio/world-models/main/experiments/distill-go/scripts/calibrate.py

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/calibrate.log) 2>&1
set -x

ECR_IMAGE="$ECR/wm-go-gpu:latest"

cleanup() {
    aws s3 cp /var/log/calibrate.log "$LOG_KEY" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
curl -fsSL $CALIB_RAW -o /mnt/work/calibrate.py

docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        export DEBIAN_FRONTEND=noninteractive
        apt-get update && apt-get install -y --no-install-recommends pachi
        export PACHI_BIN=/usr/games/pachi
        ls -l \$PACHI_BIN
        \$PACHI_BIN --version 2>&1 | head -3 || true

        cp /work-tmp/calibrate.py /work/experiments/distill-go/scripts/calibrate.py
        cd /work/experiments/distill-go

        OUT=/work-tmp/result.txt
        echo "=== KataGo@v${KATAGO_VISITS} vs Pachi ${PACHI_PLAYOUTS} playouts (9x9) ===" | tee \$OUT
        echo "anchor: Pachi = ${PACHI_ANCHOR_ELO} Elo (KGS 1d-ish)" | tee -a \$OUT
        uv run python scripts/calibrate.py \\
            --board-size 9 --komi 7.5 \\
            --katago-visits $KATAGO_VISITS \\
            --opponent pachi --pachi-playouts $PACHI_PLAYOUTS \\
            --anchor-elo $PACHI_ANCHOR_ELO \\
            --games $GAMES 2>&1 | tee -a \$OUT
        aws s3 cp \$OUT $RESULT_KEY --no-progress
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
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-go-calibrate-pachi-${STAMP}-od},{Key=role,Value=wm-go-calibration}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "result: $RESULT_KEY"
echo "log:    $LOG_KEY"
