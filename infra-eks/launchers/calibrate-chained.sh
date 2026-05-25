#!/usr/bin/env bash
# Chained Go calibration: derive KataGo@v200's absolute Elo via a two-match
# chain. Needed because KataGo@v200 swept GNU Go 100/0 in the direct match
# (Elo gap unbounded above), so we anchor through a measurable intermediate.
#
#   Match A: KataGo@v1 (pure policy) vs GNU Go L10 — anchor v1 to 1800.
#   Match B: KataGo@v200 vs KataGo@v1 — anchor v200 to v1's measured value.
#
# Output: two S3 result files. Combine the two Elo diffs locally to get v200's
# absolute Elo.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-2
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-00be06b9c01d8b036
INSTANCE_TYPE=g6.xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
GAMES=${GAMES:-100}

STAMP=$(date -u +%Y%m%dT%H%MZ)
KEY_A=s3://$S3_BUCKET/go-calibration/chain-${STAMP}-A-katago-v1-vs-gnugo.txt
KEY_B=s3://$S3_BUCKET/go-calibration/chain-${STAMP}-B-katago-v200-vs-v1.txt
LOG_KEY=s3://$S3_BUCKET/go-calibration/chain-${STAMP}.log
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
        apt-get update && apt-get install -y --no-install-recommends gnugo
        export GNUGO_BIN=/usr/games/gnugo
        ls -l \$GNUGO_BIN

        cp /work-tmp/calibrate.py /work/experiments/distill-go/scripts/calibrate.py
        cd /work/experiments/distill-go

        # --- Match A: KataGo@v1 vs GNU Go L10 ---
        OUT_A=/work-tmp/match_A.txt
        echo "=== Match A: KataGo@v1 vs GNU Go L10 (anchor 1800) ===" | tee \$OUT_A
        uv run python scripts/calibrate.py \\
            --board-size 9 --komi 7.5 \\
            --katago-visits 1 \\
            --opponent gnugo --gnugo-level 10 \\
            --anchor-elo 1800 \\
            --games $GAMES 2>&1 | tee -a \$OUT_A
        aws s3 cp \$OUT_A $KEY_A --no-progress

        # Parse Match A: "Implied KataGo@1v absolute Elo ≈ NNNN"
        V1_ABS=\$(grep -oE "Implied KataGo@1v absolute Elo .* [0-9]+" \$OUT_A | tail -1 | awk "{print \$NF}")
        echo "parsed KataGo@v1 absolute Elo: \$V1_ABS"

        # --- Match B: KataGo@v200 vs KataGo@v1 (anchor = parsed v1) ---
        OUT_B=/work-tmp/match_B.txt
        echo "=== Match B: KataGo@v200 vs KataGo@v1 (anchor \$V1_ABS) ===" | tee \$OUT_B
        uv run python scripts/calibrate.py \\
            --board-size 9 --komi 7.5 \\
            --katago-visits 200 \\
            --opponent katago --opponent-visits 1 \\
            --anchor-elo \$V1_ABS \\
            --games $GAMES 2>&1 | tee -a \$OUT_B
        aws s3 cp \$OUT_B $KEY_B --no-progress
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-go-calibrate-chain-${STAMP}-spot},{Key=role,Value=wm-go-calibration}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "Match A (v1 vs gnugo):   $KEY_A"
echo "Match B (v200 vs v1):    $KEY_B"
echo "log:                     $LOG_KEY"
