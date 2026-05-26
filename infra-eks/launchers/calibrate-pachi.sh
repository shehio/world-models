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
# First attempt at 100/5000p hung silently for 3h 49m — Pachi 1.0 (Ubuntu
# 22.04 apt) deadlocks in non-TTY containers despite `-d 0 + TERM=dumb`.
# Defaults are now smaller for a smoke-test pass; ramp up once we confirm
# Pachi+KataGo actually exchange GTP packets at all.
GAMES=${GAMES:-10}
KATAGO_VISITS=${KATAGO_VISITS:-200}
PACHI_PLAYOUTS=${PACHI_PLAYOUTS:-1000}
PACHI_ANCHOR_ELO=${PACHI_ANCHOR_ELO:-1900}  # 1000 playouts is weaker than 5000

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
        # coreutils gives us stdbuf to force line-buffered Pachi stdout
        apt-get update && apt-get install -y --no-install-recommends pachi coreutils
        export PACHI_BIN=/usr/games/pachi
        # Pachi pulls in ncurses; without a TTY it dies with
        # "Error opening terminal: unknown". TERM=dumb sidesteps that.
        export TERM=dumb
        ls -l \$PACHI_BIN
        echo "=== pachi smoke (echo name | pachi -d 0) ==="
        echo -e "name\nquit" | timeout 10 \$PACHI_BIN -d 0 2>&1 || echo "(pachi smoke timeout/error)"

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
        # Capture Pachi stderr so we can post-mortem any hang
        [ -f /tmp/gtp_opp.stderr.log ] && cp /tmp/gtp_opp.stderr.log /work-tmp/pachi.stderr.log || true
        aws s3 cp \$OUT $RESULT_KEY --no-progress
        [ -f /work-tmp/pachi.stderr.log ] && aws s3 cp /work-tmp/pachi.stderr.log ${RESULT_KEY%.txt}.pachi-stderr.log --no-progress || true
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-go-calibrate-pachi-${STAMP}-od},{Key=role,Value=wm-go-calibration}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "result: $RESULT_KEY"
echo "log:    $LOG_KEY"
