#!/usr/bin/env bash
# AlphaZero-style self-play on top of the 8x128 distilled Go prior.
#
# Starts from the d15-equivalent 9x9 prior (8×128 ep 14 = parity with KataGo @v=200,
# absolute Elo ≥ 2,366) and runs distill-then-RL for a 24h budget. The chess
# six-attempts-evicted-on-spot postmortem says: USE OD, not spot. 9x9 iters
# are short (~5 min vs chess's 90 min) but per-attempt setup is the same;
# OD cost is ~$20 for the full 24h budget on g6.xlarge — not worth the
# spot-eviction risk.
#
# Why these defaults:
#  - LR=1e-5 — same gentle LR that didn't catastrophic-forget on chess
#    after we diagnosed the LR=1e-3 / LR=1e-4 regressions.
#  - PCR full=200, reduced=50, p=0.25 — KataGo paper-default ratios for 9x9.
#  - 4 workers — saturates the 4 vCPU on g6.xlarge.
#  - --train-device cuda — trainer uses the L4 GPU; workers use CPU since
#    8×128 forward passes are cheap and dispatching every MCTS leaf to the
#    GPU would serialize them.
#  - Curl-trampoline pulls the latest selfplay_loop.py from main on every
#    container start, so script-only fixes don't require an image rebuild.

set -euo pipefail

ACCOUNT_ID=594561963943
# wm-go-gpu image lives in us-east-2 ECR; the bucket + EC2 are us-east-1.
# Two distinct regions, so two distinct AWS-CLI --region flags below.
ECR_REGION=us-east-2
S3_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=${SUBNET:-subnet-042fb3c497e2631a7}      # us-east-1b default; override on capacity errors
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.xlarge}
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$ECR_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943

PRIOR_S3=${PRIOR_S3:-s3://$S3_BUCKET/go-9x9-mpv8-gpu-v400-20260521T1230Z/checkpoints/9x9-8x128-20260522T0625Z/distilled_epoch015.pt}

# Tunables (override via env). Defaults target 24h on a single g6.xlarge.
WORKERS=${WORKERS:-4}
GAMES_PER_WORKER=${GAMES_PER_WORKER:-2}
PCR_FULL=${PCR_FULL:-200}
PCR_REDUCED=${PCR_REDUCED:-50}
PCR_P_FULL=${PCR_P_FULL:-0.25}
TRAIN_STEPS=${TRAIN_STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-5}
TIME_BUDGET=${TIME_BUDGET:-86400}     # 24h
EVAL_EVERY=${EVAL_EVERY:-4}
EVAL_GAMES=${EVAL_GAMES:-10}

N_BLOCKS=${N_BLOCKS:-8}
N_FILTERS=${N_FILTERS:-128}
N_INPUT_PLANES=${N_INPUT_PLANES:-4}

STAMP=$(date -u +%Y%m%dT%H%MZ)
RUN_ID=${STAMP}-selfplay-9x9-8x128
S3_CKPT_BASE=s3://$S3_BUCKET/go-9x9-selfplay/$RUN_ID

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/selfplay.log) 2>&1
set -x

PRIOR_S3="$PRIOR_S3"
S3_CKPT_BASE="$S3_CKPT_BASE"
ECR_IMAGE="$ECR/wm-go-gpu:latest"

cleanup() {
    aws s3 cp /var/log/selfplay.log "\$S3_CKPT_BASE/host-launch.log" --region $S3_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
for i in \$(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null

aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
# Pull the latest selfplay loop from main (curl-trampoline — no image rebuild needed).
curl -fsSL https://raw.githubusercontent.com/shehio/world-models/main/experiments/distill-go/scripts/selfplay_loop.py \\
    -o /mnt/work/selfplay_loop.py

docker run --rm \\
    --gpus all \\
    --shm-size=2g \\
    -v /mnt/work:/work-tmp \\
    -e PRIOR_S3="\$PRIOR_S3" -e S3_CKPT_BASE="\$S3_CKPT_BASE" \\
    -e AWS_DEFAULT_REGION=$S3_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cp /work-tmp/selfplay_loop.py /work/experiments/distill-go/scripts/selfplay_loop.py
        cd /work/experiments/distill-go
        mkdir -p /work/checkpoints_selfplay

        uv run python scripts/selfplay_loop.py \\
            --board-size 9 --komi 7.5 \\
            --n-blocks $N_BLOCKS --n-filters $N_FILTERS --n-input-planes $N_INPUT_PLANES \\
            --workers $WORKERS --games-per-worker $GAMES_PER_WORKER --worker-device cpu \\
            --pcr-sims-full $PCR_FULL --pcr-sims-reduced $PCR_REDUCED --pcr-p-full $PCR_P_FULL \\
            --train-steps $TRAIN_STEPS --batch-size $BATCH_SIZE --lr $LR \\
            --train-device cuda \\
            --replay-shards 20 --temp-moves 30 --max-moves 200 \\
            --time-budget $TIME_BUDGET --max-iters 10000 \\
            --eval-every $EVAL_EVERY --eval-games $EVAL_GAMES --eval-sims 50 \\
            --resume \$PRIOR_S3 \\
            --ckpt-dir /work/checkpoints_selfplay \\
            --s3-ckpt-base \$S3_CKPT_BASE 2>&1 | tee -a /work/checkpoints_selfplay/selfplay_log.txt

        echo "=== final sync ==="
        aws s3 cp /work/checkpoints_selfplay/selfplay_log.txt "\$S3_CKPT_BASE/selfplay_log.txt" --no-progress || true
    '
EOF
)

aws ec2 run-instances --region $LAUNCH_REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-go-selfplay-9x9-$STAMP},{Key=role,Value=wm-go-selfplay}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "run id:   $RUN_ID"
echo "ckpts:    $S3_CKPT_BASE/"
echo "tail log: aws s3 cp $S3_CKPT_BASE/selfplay_log.txt - | tail -40"
