#!/usr/bin/env bash
# Periodic head-to-head: chess selfplay-trained ckpt vs R2v2 ep 14 prior.
#
# Fire this whenever a new selfplay iter ckpt lands during the 24h run
# to get early Elo signal — much faster than waiting for the whole run
# to finish. Each H2H: ~$3, ~1-1.5h wallclock at sims=400.
#
# Override CKPT_A_S3 to point at any selfplay iter; PRIOR_S3 to change
# the opponent baseline.

set -euo pipefail

ACCOUNT_ID=594561963943
ECR_REGION=us-east-1
S3_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a            # DL Base GPU AL2023 us-east-1
SUBNET=${SUBNET:-subnet-00be06b9c01d8b036}
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.xlarge}    # 4 vCPU + L4 — small box, MCTS is fast at sims=400
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$ECR_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943

# Must override CKPT_A_S3 to point at a real selfplay iter ckpt.
CKPT_A_S3=${CKPT_A_S3:?must set CKPT_A_S3 to a chess selfplay iter ckpt (e.g. s3://.../net_iter003.pt)}
PRIOR_S3=${PRIOR_S3:-s3://$S3_BUCKET/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-20x256/20260525T0935Z-full46M-20x256-cosine/distilled_epoch014.pt}

WORKERS=${WORKERS:-4}                         # 4 vCPU on g6.xlarge
GAMES_PER_WORKER=${GAMES_PER_WORKER:-8}        # 32 games total — tight enough for periodic checks
SIMS=${SIMS:-400}                             # cheap; 800 is the production eval setting
N_BLOCKS=${N_BLOCKS:-20}
N_FILTERS=${N_FILTERS:-256}

STAMP=$(date -u +%Y%m%dT%H%MZ)
CKPT_A_NAME=$(basename "$CKPT_A_S3" .pt)
RESULT_KEY=s3://$S3_BUCKET/d15-mpv8-T1-g250000-20260519T0412Z/selfplay/net-20x256/h2h-${CKPT_A_NAME}-vs-r2v2-ep14-${STAMP}.txt
LOG_KEY=${RESULT_KEY%.txt}.log

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/h2h.log) 2>&1
set -x

CKPT_A_S3="$CKPT_A_S3"
PRIOR_S3="$PRIOR_S3"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

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
aws s3 cp \$CKPT_A_S3 /mnt/work/ckpt_a.pt --no-progress
aws s3 cp \$PRIOR_S3 /mnt/work/ckpt_b.pt --no-progress

docker run --rm \\
    --gpus all \\
    --shm-size=2g \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$S3_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cd /work/experiments/selfplay
        OUT=/work-tmp/h2h_result.txt
        echo "=== chess h2h: $(basename $CKPT_A_S3) vs R2v2 ep 14 prior ===" | tee \$OUT
        echo "A: $CKPT_A_S3" | tee -a \$OUT
        echo "B: $PRIOR_S3" | tee -a \$OUT
        echo "sims: $SIMS  workers: $WORKERS  games: \$(($WORKERS * $GAMES_PER_WORKER))" | tee -a \$OUT
        python scripts/h2h_mp.py \\
            --ckpt-a /work-tmp/ckpt_a.pt \\
            --ckpt-b /work-tmp/ckpt_b.pt \\
            --workers $WORKERS --games-per-worker $GAMES_PER_WORKER --sims $SIMS \\
            --n-blocks $N_BLOCKS --n-filters $N_FILTERS \\
            --device cuda 2>&1 | tee -a \$OUT
        aws s3 cp \$OUT "$RESULT_KEY" --no-progress
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-chess-h2h-${CKPT_A_NAME}-${STAMP}},{Key=role,Value=wm-chess-h2h}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text

echo ""
echo "A:      $CKPT_A_S3"
echo "B:      $PRIOR_S3"
echo "result: $RESULT_KEY"
echo "log:    $LOG_KEY"
