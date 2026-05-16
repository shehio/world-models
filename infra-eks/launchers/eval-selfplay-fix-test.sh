#!/usr/bin/env bash
# Verification of the self-play LR fix (1e-3 → 1e-5).
#
# What this does on one g6.4xlarge:
#   1. Pull the distilled-epoch019 prior (the 1807-Elo starting point)
#   2. Run ONE short self-play iter: 8 games, sims=200, 50 train steps,
#      lr=1e-5 (the new default after the fix)
#   3. Eval the result at sims=800 vs UCI=1800 with 80 games
#   4. Upload eval + log, terminate
#
# Expected if the LR fix works: Elo ≈ 1800 (i.e., the prior held up;
# one iter of gentle SGD didn't nuke it).
# Expected if regression persists: Elo << 1800.
#
# Total wallclock target: ~30 min on a g6.4xlarge.

set -euo pipefail

ACCOUNT_ID=594561963943
REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6.4xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

PRIOR_S3=s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/checkpoints/net-20x256/20260514T1332Z/distilled_epoch019.pt
RESULT_PREFIX=s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/selfplay/net-20x256/fix-test-$(date -u +%Y%m%dT%H%MZ)

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/fix-test.log) 2>&1
set -x

PRIOR_S3="$PRIOR_S3"
RESULT_PREFIX="$RESULT_PREFIX"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    aws s3 cp /var/log/fix-test.log "\$RESULT_PREFIX/fix-test.log" --region $REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e PRIOR_S3=\$PRIOR_S3 -e RESULT_PREFIX=\$RESULT_PREFIX \\
    -e AWS_DEFAULT_REGION=$REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        # ---- 1. Pull prior ----
        mkdir -p /work-tmp/cp
        aws s3 cp \$PRIOR_S3 /work-tmp/cp/prior.pt --no-progress

        # ---- 2. Short self-play iter ----
        cd /work/experiments/selfplay
        OUT=/work-tmp/fix-test-out.txt
        echo "=== 1 iter self-play with fix (lr=1e-5, train_steps=50, sims=200, games=8) ===" | tee \$OUT
        python scripts/selfplay_loop_mp.py \\
            --resume /work-tmp/cp/prior.pt \\
            --workers 4 --games-per-worker 2 --sims 200 --batch-size 8 \\
            --n-blocks 20 --n-filters 256 \\
            --train-steps 50 --lr 1e-5 \\
            --time-budget 600 --max-iters 1 \\
            --eval-every 999 \\
            --ckpt-dir /work-tmp/cp 2>&1 | tee -a \$OUT

        # ---- 3. Eval the POST-iter weights vs UCI=1800 ----
        # NOTE: current.pt is the PRE-iter weights (written at the start
        # of each iter for workers). The post-train state lives in
        # final.pt (wrapped {net, opt, iter, hour}). distill-soft eval.py
        # unwraps that via distill_soft.ckpt.unwrap_state_dict.
        cd /work/experiments/distill-soft
        echo "" | tee -a \$OUT
        echo "=== eval final.pt (post-iter-0 trained weights) vs Stockfish UCI=1800 (sims=800, 80 games) ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt /work-tmp/cp/final.pt --workers 8 --games-per-worker 10 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1800 --agent-device cuda 2>&1 | tee -a \$OUT

        aws s3 cp \$OUT \$RESULT_PREFIX/result.txt --no-progress
        aws s3 cp /work-tmp/cp/final.pt \$RESULT_PREFIX/final.pt --no-progress
    '
EOF
)

aws ec2 run-instances --region $REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-selfplay-fix-test},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text

echo "Result will land in: $RESULT_PREFIX/"
