#!/usr/bin/env bash
# Head-to-head: Go self-play latest ckpt vs the distilled prior.
#
# The vs-random eval per-iter has been saturated 10/0/0 since iter 0,
# so it gives zero signal on whether self-play actually improved over
# the starting checkpoint. This match is the real "did we gain
# anything?" eval — both sides play with the same MCTS, alternating
# colors, on 9x9.
#
# Defaults: 40 games (5 workers × 8), sims=100 on both sides, ~7-10
# minutes wallclock on g6.xlarge OD. Override via env:
#   SELFPLAY_S3, PRIOR_S3, GAMES, SIMS, WORKERS.

set -euo pipefail

ACCOUNT_ID=594561963943
ECR_REGION=us-east-2          # wm-go-gpu image lives here
S3_REGION=us-east-1           # bucket + EC2 here
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=${SUBNET:-subnet-00be06b9c01d8b036}   # us-east-1c default
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.xlarge}
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$ECR_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943

# Default to the latest known self-play iter; can override with SELFPLAY_S3.
SELFPLAY_S3=${SELFPLAY_S3:-s3://$S3_BUCKET/go-9x9-selfplay/20260526T1947Z-selfplay-9x9-8x128/current.pt}
PRIOR_S3=${PRIOR_S3:-s3://$S3_BUCKET/go-9x9-mpv8-gpu-v400-20260521T1230Z/checkpoints/9x9-8x128-20260522T0625Z/distilled_epoch015.pt}

WORKERS=${WORKERS:-5}
GAMES_PER_WORKER=${GAMES_PER_WORKER:-8}
SIMS=${SIMS:-100}

N_BLOCKS=${N_BLOCKS:-8}
N_FILTERS=${N_FILTERS:-128}
N_INPUT_PLANES=${N_INPUT_PLANES:-4}

STAMP=$(date -u +%Y%m%dT%H%MZ)
RESULT_KEY=s3://$S3_BUCKET/go-9x9-selfplay/h2h-${STAMP}.json
LOG_KEY=s3://$S3_BUCKET/go-9x9-selfplay/h2h-${STAMP}.log
H2H_RAW=https://raw.githubusercontent.com/shehio/world-models/main/experiments/distill-go/scripts/h2h.py

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/h2h.log) 2>&1
set -x

SELFPLAY_S3="$SELFPLAY_S3"
PRIOR_S3="$PRIOR_S3"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-go-gpu:latest"

cleanup() {
    aws s3 cp /var/log/h2h.log "$LOG_KEY" --region $S3_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
for i in \$(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null

aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
# Pull the latest h2h.py from main (curl-trampoline — image doesn't have it).
curl -fsSL $H2H_RAW -o /mnt/work/h2h.py

docker run --rm \\
    --shm-size=2g \\
    -v /mnt/work:/work-tmp \\
    -e SELFPLAY_S3="\$SELFPLAY_S3" -e PRIOR_S3="\$PRIOR_S3" \\
    -e AWS_DEFAULT_REGION=$S3_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cp /work-tmp/h2h.py /work/experiments/distill-go/scripts/h2h.py
        cd /work/experiments/distill-go

        OUT=/work-tmp/h2h_result.json
        uv run python scripts/h2h.py \\
            --ckpt-a "\$SELFPLAY_S3" \\
            --ckpt-b "\$PRIOR_S3" \\
            --board-size 9 --komi 7.5 \\
            --n-blocks $N_BLOCKS --n-filters $N_FILTERS --n-input-planes $N_INPUT_PLANES \\
            --workers $WORKERS --games-per-worker $GAMES_PER_WORKER \\
            --sims $SIMS --max-moves 200 \\
            --output \$OUT
        aws s3 cp \$OUT "$RESULT_KEY" --no-progress
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-go-h2h-$STAMP},{Key=role,Value=wm-go-h2h}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "A (self-play): $SELFPLAY_S3"
echo "B (prior):     $PRIOR_S3"
echo "result:        $RESULT_KEY"
echo "log:           $LOG_KEY"
