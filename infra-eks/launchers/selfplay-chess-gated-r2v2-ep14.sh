#!/usr/bin/env bash
# GATED self-play RL proof-of-mechanism, on top of R2v2 ep 14 (the project's
# strongest distilled ckpt). This is the MVP "prove self-play works" run:
#
#   1. Arena gating (AlphaGo Zero evaluator): workers self-play from a frozen
#      CHAMPION; the trainer improves a CANDIDATE; every GATE_EVERY iters the
#      candidate must beat the champion >= GATE_THRESHOLD to be promoted.
#      Improvement is monotonic by construction — a weaker net can't be promoted.
#   2. KL-anchor to the teacher (KL_BETA): trust region so warm-started RL sharpens
#      where search is confident without forgetting the broad supervised distribution.
#   3. Bigger replay window (REPLAY_SHARDS=50): more data per train step, free.
#   4. In-loop Stockfish-panel yardstick on every promotion: the honest ruler the
#      vs-random eval never gave us. (Definitive Elo still comes from a full
#      distill-soft eval.py run on the saved champions.)
#
# Gating/KL are opt-in in entrypoint-selfplay.sh (GATE_EVERY=0 / KL_BETA=0 off by
# default); this launcher turns them on. The standard selfplay-chess-r2v2-ep14.sh
# run is unaffected.
#
# SUCCESS CRITERION (lock once the teacher baseline eval lands): champion Elo on
# the Stockfish panel (same harness) clears the teacher baseline by >= 50 with
# non-overlapping CIs. One verified promotion + a rising SF tick = directional proof.
#
# The hparams below are STARTING POINTS — tune after the local CPU smoke and the
# teacher baseline. In particular GATE_EVERY/KL_BETA and how many generations fit
# in TIME_BUDGET depend on measured per-iter cost.
#
# Cost: g6.4xlarge OD us-east-1 ~ $1.3/h; gating adds a match every GATE_EVERY
# iters + an SF panel on promotion. Budget ~24h ≈ $25-40.

set -euo pipefail

ACCOUNT_ID=594561963943
ECR_REGION=us-east-1
S3_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=${AMI:-ami-027c3ae8019fc0d3a}
SUBNET=${SUBNET:-subnet-00be06b9c01d8b036}
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.4xlarge}
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$ECR_REGION.amazonaws.com

# Prior + output paths
S3_BUCKET=wm-chess-library-594561963943
INIT_FROM_S3=${INIT_FROM_S3:-s3://$S3_BUCKET/d15-mpv8-T1-g250000-20260519T0412Z/checkpoints/net-20x256/20260525T0935Z-full46M-20x256-cosine/distilled_epoch014.pt}
S3_PREFIX=d15-mpv8-T1-g250000-20260519T0412Z
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)-selfplay-from-r2v2-ep14-gated}

# Base selfplay hparams (match the non-gated run). All env-overridable so a short
# cloud smoke can reuse this script (e.g. TIME_BUDGET=2400 GATE_EVERY=1 ...).
N_BLOCKS=${N_BLOCKS:-20}
N_FILTERS=${N_FILTERS:-256}
SIMS=${SIMS:-800}
WORKERS=${WORKERS:-14}
GAMES_PER_WORKER=${GAMES_PER_WORKER:-4}
TRAIN_STEPS=${TRAIN_STEPS:-200}
LR=${LR:-1e-5}
TIME_BUDGET=${TIME_BUDGET:-86400}     # 24h
EVAL_EVERY=${EVAL_EVERY:-1}
EVAL_GAMES=${EVAL_GAMES:-20}
EVAL_SIMS=${EVAL_SIMS:-200}
REPLAY_SHARDS=${REPLAY_SHARDS:-50}    # #3: bigger window (was 20)

# Gating + KL + Stockfish yardstick (the MVP additions). STARTING POINTS — tune.
GATE_EVERY=${GATE_EVERY:-2}           # gate every N iters of champion self-play + candidate training
GATE_GAMES=${GATE_GAMES:-60}          # candidate-vs-champion match size
GATE_SIMS=${GATE_SIMS:-200}
GATE_THRESHOLD=${GATE_THRESHOLD:-0.55}
KL_BETA=${KL_BETA:-0.5}               # trust-region weight on KL(policy||teacher); tune 0.1-1.0
SF_EVAL_ELOS=${SF_EVAL_ELOS:-1350,1800}
SF_EVAL_GAMES=${SF_EVAL_GAMES:-20}
SF_EVAL_SIMS=${SF_EVAL_SIMS:-400}

# The gating/KL code isn't baked into the image — pull it from main at startup.
REFRESH_SRC_FROM_MAIN=${REFRESH_SRC_FROM_MAIN:-1}

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/selfplay.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/selfplay.log "s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-${N_BLOCKS}x${N_FILTERS}/$RUN_ID/host-launch.log" --region $S3_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
for i in \$(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null

aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin $ECR
docker pull $ECR/wm-chess-gpu:latest

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    --shm-size=4g \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$S3_REGION \\
    -e S3_BUCKET=$S3_BUCKET \\
    -e S3_PREFIX=$S3_PREFIX \\
    -e RUN_ID=$RUN_ID \\
    -e INIT_FROM_S3=$INIT_FROM_S3 \\
    -e N_BLOCKS=$N_BLOCKS \\
    -e N_FILTERS=$N_FILTERS \\
    -e SIMS=$SIMS \\
    -e WORKERS=$WORKERS \\
    -e GAMES_PER_WORKER=$GAMES_PER_WORKER \\
    -e TRAIN_STEPS=$TRAIN_STEPS \\
    -e LR=$LR \\
    -e TIME_BUDGET=$TIME_BUDGET \\
    -e EVAL_EVERY=$EVAL_EVERY \\
    -e EVAL_GAMES=$EVAL_GAMES \\
    -e EVAL_SIMS=$EVAL_SIMS \\
    -e REPLAY_SHARDS=$REPLAY_SHARDS \\
    -e GATE_EVERY=$GATE_EVERY \\
    -e GATE_GAMES=$GATE_GAMES \\
    -e GATE_SIMS=$GATE_SIMS \\
    -e GATE_THRESHOLD=$GATE_THRESHOLD \\
    -e KL_BETA=$KL_BETA \\
    -e SF_EVAL_ELOS=$SF_EVAL_ELOS \\
    -e SF_EVAL_GAMES=$SF_EVAL_GAMES \\
    -e SF_EVAL_SIMS=$SF_EVAL_SIMS \\
    -e REFRESH_SRC_FROM_MAIN=$REFRESH_SRC_FROM_MAIN \\
    --entrypoint bash \\
    $ECR/wm-chess-gpu:latest -lc '
        curl -sSL https://raw.githubusercontent.com/shehio/world-models/main/infra-eks/entrypoint-selfplay.sh -o /entrypoint-selfplay.sh
        chmod +x /entrypoint-selfplay.sh
        exec /entrypoint-selfplay.sh
    '
EOF
)

aws ec2 run-instances --region $LAUNCH_REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=300,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-selfplay-gated-${RUN_ID}},{Key=role,Value=wm-chess-selfplay}]" \
    --query 'Instances[0].[InstanceId,State.Name,InstanceLifecycle]' --output text

echo ""
echo "run id:        $RUN_ID"
echo "ckpts at:      s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-${N_BLOCKS}x${N_FILTERS}/$RUN_ID/"
echo "host log at:   s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-${N_BLOCKS}x${N_FILTERS}/$RUN_ID/host-launch.log"
