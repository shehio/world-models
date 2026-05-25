#!/usr/bin/env bash
# Calibration match: KataGo@v200 (the same engine used as our 9x9 Go eval anchor)
# vs GNU Go on 9x9. Derives an absolute Elo for KataGo so we can convert all our
# "student − KataGo" deltas into absolute Go Elos.
#
# Defaults: 100 games, GNU Go level 10, KataGo at v200, GNU Go anchor = 1800 Elo
# (AlphaGo paper, Silver et al. 2016). ~1h wallclock on g6.xlarge.
#
# Uses the curl-trampoline pattern: pulls calibrate.py from main at container
# startup, since the baked image doesn't yet include it (see [[feedback-curl-trampoline]]).
#
# Override knobs via env: GAMES, KATAGO_VISITS, GNUGO_LEVEL, GNUGO_ANCHOR_ELO.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-2
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-00be06b9c01d8b036          # us-east-1c (consistent with eval launcher)
INSTANCE_TYPE=g6.xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943

GAMES=${GAMES:-100}
KATAGO_VISITS=${KATAGO_VISITS:-200}
GNUGO_LEVEL=${GNUGO_LEVEL:-10}
GNUGO_ANCHOR_ELO=${GNUGO_ANCHOR_ELO:-1800}

STAMP=$(date -u +%Y%m%dT%H%MZ)
RESULT_KEY=s3://$S3_BUCKET/go-calibration/katago-v${KATAGO_VISITS}-vs-gnugo-l${GNUGO_LEVEL}-${STAMP}.txt
CALIB_RAW=https://raw.githubusercontent.com/shehio/world-models/main/experiments/distill-go/scripts/calibrate.py

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/calibrate.log) 2>&1
set -x

RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-go-gpu:latest"

cleanup() {
    aws s3 cp /var/log/calibrate.log "\${RESULT_KEY%.txt}.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work

# Curl-trampoline: grab the latest calibrate.py from main, mount into container.
curl -fsSL $CALIB_RAW -o /mnt/work/calibrate.py

docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        export DEBIAN_FRONTEND=noninteractive
        apt-get update && apt-get install -y --no-install-recommends gnugo
        # gnugo lives in /usr/games on Ubuntu; not on PATH by default.
        export GNUGO_BIN=/usr/games/gnugo
        ls -l \$GNUGO_BIN

        cp /work-tmp/calibrate.py /work/experiments/distill-go/scripts/calibrate.py

        cd /work/experiments/distill-go
        OUT=/work-tmp/calibrate_result.txt
        echo "=== Calibration: KataGo@v${KATAGO_VISITS} vs GNU Go L${GNUGO_LEVEL} (9x9) ===" | tee \$OUT
        echo "anchor: GNU Go = ${GNUGO_ANCHOR_ELO} Elo (AlphaGo paper scale)" | tee -a \$OUT
        echo "games: $GAMES" | tee -a \$OUT
        echo "" | tee -a \$OUT
        uv run python scripts/calibrate.py \\
            --board-size 9 --komi 7.5 \\
            --katago-visits $KATAGO_VISITS \\
            --gnugo-level $GNUGO_LEVEL \\
            --gnugo-anchor-elo $GNUGO_ANCHOR_ELO \\
            --games $GAMES 2>&1 | tee -a \$OUT
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-go-calibrate-katago-v${KATAGO_VISITS}-vs-gnugo-l${GNUGO_LEVEL}-spot},{Key=role,Value=wm-go-calibration}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "result will land at: $RESULT_KEY"
echo "log at: ${RESULT_KEY%.txt}.log"
