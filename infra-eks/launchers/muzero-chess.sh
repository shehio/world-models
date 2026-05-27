#!/usr/bin/env bash
# MuZero chess training launcher — bare EC2, one instance, OD by default.
#
# Why OD (not spot):
#   The chess self-play attempts 1-6 were ALL evicted before iter 1 on spot.
#   Attempt 7 moved to OD and made it past the loop start. MuZero is even
#   less mature (literally never run before this commit) — burning two
#   hours on a spot eviction is worth more than the OD premium for the
#   first real run. Override via INSTANCE_MARKET=spot if you want spot.
#
# Why g6.xlarge:
#   MuZero net is ~M params (64-ch latent, 5/3/2 ResNets) — far smaller
#   than the 20×256 chess ResNet selfplay uses. MCTS at 400 sims is the
#   wall-clock bottleneck, not GPU compute. 1× L4 is plenty.
#
# Usage:
#   bash infra-eks/launchers/muzero-chess.sh
# Overridable env:
#   RUN_ID, SIMS, EVAL_EVERY, EVAL_GAMES, EVAL_STOCKFISH_ELO,
#   ITERATIONS, TIME_BUDGET, INSTANCE_TYPE, INSTANCE_MARKET, REGION, ...

set -euo pipefail

ACCOUNT_ID=594561963943
REGION=${REGION:-us-east-1}
AMI=${AMI:-ami-027c3ae8019fc0d3a}             # DL Base GPU AL2023 us-east-1
SUBNET=${SUBNET:-subnet-00be06b9c01d8b036}   # us-east-1c (capacity confirmed)
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.xlarge}
INSTANCE_PROFILE=${INSTANCE_PROFILE:-wm-chess-merge-instance-profile}
INSTANCE_MARKET=${INSTANCE_MARKET:-od}       # "od" or "spot"
ECR=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

S3_BUCKET=${S3_BUCKET:-wm-chess-library-$ACCOUNT_ID}
S3_PREFIX=${S3_PREFIX:-muzero-chess}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)-muzero-chess}

# Defaults that the user explicitly asked for (cloud stage).
SIMS=${SIMS:-400}
EVAL_EVERY=${EVAL_EVERY:-5}
EVAL_GAMES=${EVAL_GAMES:-20}
EVAL_STOCKFISH_ELO=${EVAL_STOCKFISH_ELO:-1320}
EVAL_SIMS=${EVAL_SIMS:-0}                     # 0 = same as training SIMS

ITERATIONS=${ITERATIONS:-200}
TRAIN_STEPS=${TRAIN_STEPS:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-3}
MAX_PLIES=${MAX_PLIES:-200}
TEMP_MOVES=${TEMP_MOVES:-30}
UNROLL=${UNROLL:-5}
WARMUP_GAMES=${WARMUP_GAMES:-4}
BUFFER_CAPACITY=${BUFFER_CAPACITY:-200}
TIME_BUDGET=${TIME_BUDGET:-43200}            # 12h initial run

# Network shape — paper-ish for first cloud run, easy to scale up later.
LATENT_CHANNELS=${LATENT_CHANNELS:-64}
REPR_BLOCKS=${REPR_BLOCKS:-5}
REPR_FILTERS=${REPR_FILTERS:-64}
DYN_BLOCKS=${DYN_BLOCKS:-3}
DYN_FILTERS=${DYN_FILTERS:-64}
PRED_BLOCKS=${PRED_BLOCKS:-2}
PRED_FILTERS=${PRED_FILTERS:-64}

CKPT_S3_BASE="s3://$S3_BUCKET/$S3_PREFIX/$RUN_ID"

echo "[launcher] region=$REGION instance=$INSTANCE_TYPE market=$INSTANCE_MARKET"
echo "[launcher] s3 base: $CKPT_S3_BASE"
echo "[launcher] sims=$SIMS  eval_every=$EVAL_EVERY  eval_games=$EVAL_GAMES  eval_uci=$EVAL_STOCKFISH_ELO"
echo "[launcher] iterations=$ITERATIONS  time_budget=${TIME_BUDGET}s"

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/muzero-chess.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/muzero-chess.log "$CKPT_S3_BASE/host-launch.log" --region $REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR
docker pull $ECR/wm-chess-gpu:latest

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$REGION \\
    -e S3_BUCKET=$S3_BUCKET \\
    -e S3_PREFIX=$S3_PREFIX \\
    -e RUN_ID=$RUN_ID \\
    -e SIMS=$SIMS \\
    -e EVAL_EVERY=$EVAL_EVERY \\
    -e EVAL_GAMES=$EVAL_GAMES \\
    -e EVAL_STOCKFISH_ELO=$EVAL_STOCKFISH_ELO \\
    -e EVAL_SIMS=$EVAL_SIMS \\
    -e ITERATIONS=$ITERATIONS \\
    -e TRAIN_STEPS=$TRAIN_STEPS \\
    -e BATCH_SIZE=$BATCH_SIZE \\
    -e LR=$LR \\
    -e MAX_PLIES=$MAX_PLIES \\
    -e TEMP_MOVES=$TEMP_MOVES \\
    -e UNROLL=$UNROLL \\
    -e WARMUP_GAMES=$WARMUP_GAMES \\
    -e BUFFER_CAPACITY=$BUFFER_CAPACITY \\
    -e TIME_BUDGET=$TIME_BUDGET \\
    -e LATENT_CHANNELS=$LATENT_CHANNELS \\
    -e REPR_BLOCKS=$REPR_BLOCKS -e REPR_FILTERS=$REPR_FILTERS \\
    -e DYN_BLOCKS=$DYN_BLOCKS -e DYN_FILTERS=$DYN_FILTERS \\
    -e PRED_BLOCKS=$PRED_BLOCKS -e PRED_FILTERS=$PRED_FILTERS \\
    --entrypoint bash \\
    $ECR/wm-chess-gpu:latest -lc '
        # Trampoline: pull the muzero-chess entrypoint from main so we can
        # iterate on the cloud loop without rebuilding the image.
        curl -sSL https://raw.githubusercontent.com/shehio/world-models/main/infra-eks/entrypoint-muzero-chess.sh -o /entrypoint-muzero-chess.sh
        chmod +x /entrypoint-muzero-chess.sh
        exec /entrypoint-muzero-chess.sh
    '
EOF
)

MARKET_OPTS=""
if [ "$INSTANCE_MARKET" = "spot" ]; then
    MARKET_OPTS='--instance-market-options MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'
fi

aws ec2 run-instances --region $REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    $MARKET_OPTS \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-muzero-chess-${RUN_ID}},{Key=role,Value=wm-chess-muzero}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
