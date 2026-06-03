#!/usr/bin/env bash
# Distill-init MuZero launcher — train only the dynamics g on top of the
# frozen distilled teacher (R2 v2 ep 14, 2,301 Elo). Bare EC2, OD by default
# (consistent with the muzero-chess launcher; spot is the wrong bet for a
# first run). g6.xlarge L4 is plenty — the teacher is 24M params, g is small.
#
# Usage: bash infra-eks/launchers/muzero-distill-init.sh
# Override TEACHER_CKPT_S3, SIMS, N_GAMES, ROUNDS, INSTANCE_MARKET, ... via env.

set -euo pipefail

ACCOUNT_ID=594561963943
REGION=${REGION:-us-east-1}
AMI=${AMI:-ami-027c3ae8019fc0d3a}
SUBNET=${SUBNET:-subnet-00be06b9c01d8b036}
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.xlarge}
INSTANCE_PROFILE=${INSTANCE_PROFILE:-wm-chess-merge-instance-profile}
INSTANCE_MARKET=${INSTANCE_MARKET:-od}
ECR=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

S3_BUCKET=${S3_BUCKET:-wm-chess-library-$ACCOUNT_ID}
S3_PREFIX=${S3_PREFIX:-muzero-distill-init}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)-distill-init}

# The distilled teacher: R2 v2 ep 14 (2,301 Elo @ sims=4000 vs UCI=1800, 20x256).
# Verified path — the cosine run, epoch014.pt (display "ep 14" == file epoch014).
TEACHER_CKPT_S3=${TEACHER_CKPT_S3:-s3://wm-chess-library-$ACCOUNT_ID/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-20x256/20260525T0935Z-full46M-20x256-cosine/distilled_epoch014.pt}
TEACHER_N_BLOCKS=${TEACHER_N_BLOCKS:-20}
TEACHER_N_FILTERS=${TEACHER_N_FILTERS:-256}

SIMS=${SIMS:-800}                       # match the teacher's sims=800 eval anchor
MCTS_BATCH_SIZE=${MCTS_BATCH_SIZE:-8}
MCTS_TOP_K=${MCTS_TOP_K:-32}
UNROLL=${UNROLL:-5}
N_GAMES=${N_GAMES:-2000}                # teacher self-play games for transitions
ROUNDS=${ROUNDS:-10}
TRAIN_STEPS_PER_ROUND=${TRAIN_STEPS_PER_ROUND:-500}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-3}
EPSILON_RANDOM=${EPSILON_RANDOM:-0.1}
LATENT_LOSS_WEIGHT=${LATENT_LOSS_WEIGHT:-1.0}   # 0 = drop latent-match → paper value-equivalence only
DYNAMICS_GRAD_SCALE=${DYNAMICS_GRAD_SCALE:-1.0} # 0.5 = MuZero App. G dynamics gradient scaling
G_OUTPUT_RELU=${G_OUTPUT_RELU:-0}               # 1 = ReLU g output → match teacher's post-ReLU latent manifold
EVAL_GAMES=${EVAL_GAMES:-30}
EVAL_ELOS=${EVAL_ELOS:-1350,1800}       # 1350 = our SF floor (engine min is 1320 on SF16+; 1350 safe on SF <=15.1)
TIME_BUDGET=${TIME_BUDGET:-21600}       # 6h cap

CKPT_S3_BASE="s3://$S3_BUCKET/$S3_PREFIX/$RUN_ID"
echo "[launcher] region=$REGION instance=$INSTANCE_TYPE market=$INSTANCE_MARKET"
echo "[launcher] s3 base: $CKPT_S3_BASE"
echo "[launcher] teacher: $TEACHER_CKPT_S3"
echo "[launcher] sims=$SIMS n_games=$N_GAMES rounds=$ROUNDS eval_elos=$EVAL_ELOS"
echo "[launcher] levers: latent_loss_weight=$LATENT_LOSS_WEIGHT epsilon_random=$EPSILON_RANDOM dynamics_grad_scale=$DYNAMICS_GRAD_SCALE g_output_relu=$G_OUTPUT_RELU"

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/distill-init.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/distill-init.log "$CKPT_S3_BASE/host-launch.log" --region $REGION 2>&1 || true
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
    -e S3_BUCKET=$S3_BUCKET -e S3_PREFIX=$S3_PREFIX -e RUN_ID=$RUN_ID \\
    -e TEACHER_CKPT_S3=$TEACHER_CKPT_S3 \\
    -e TEACHER_N_BLOCKS=$TEACHER_N_BLOCKS -e TEACHER_N_FILTERS=$TEACHER_N_FILTERS \\
    -e SIMS=$SIMS -e MCTS_BATCH_SIZE=$MCTS_BATCH_SIZE -e MCTS_TOP_K=$MCTS_TOP_K \\
    -e UNROLL=$UNROLL -e N_GAMES=$N_GAMES -e ROUNDS=$ROUNDS \\
    -e TRAIN_STEPS_PER_ROUND=$TRAIN_STEPS_PER_ROUND \\
    -e BATCH_SIZE=$BATCH_SIZE -e LR=$LR -e EPSILON_RANDOM=$EPSILON_RANDOM \\
    -e LATENT_LOSS_WEIGHT=$LATENT_LOSS_WEIGHT -e DYNAMICS_GRAD_SCALE=$DYNAMICS_GRAD_SCALE \\
    -e G_OUTPUT_RELU=$G_OUTPUT_RELU \\
    -e EVAL_GAMES=$EVAL_GAMES -e EVAL_ELOS=$EVAL_ELOS \\
    --entrypoint bash \\
    $ECR/wm-chess-gpu:latest -lc '
        curl -sSL https://raw.githubusercontent.com/shehio/world-models/main/infra-eks/entrypoint-muzero-distill-init.sh -o /entrypoint.sh
        chmod +x /entrypoint.sh
        exec /entrypoint.sh
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-distill-init-${RUN_ID}},{Key=role,Value=wm-chess-muzero}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text
