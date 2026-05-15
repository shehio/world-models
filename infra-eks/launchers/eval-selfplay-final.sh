#!/usr/bin/env bash
# Evaluate the self-play loop's final checkpoint.
#
# The auto-eval daemon's filename filter is `distilled_epoch[0-9]+\.pt`
# which doesn't match self-play outputs (net_iter*.pt, final.pt). This
# launcher is a one-off — runs eval.py against the self-play final.pt
# at the same anchors (UCI=1350, UCI=1800) and sims=800.
#
# eval.py was patched to accept both bare state_dict and the wrapped
# {"net": ..., "opt": ...} format the self-play loop writes, so no
# unwrap step needed here.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6.4xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

# The self-play run that initialized from d15 ep19 distilled prior.
# Eval the final.pt (post-10-iter time-budget) which represents the
# strongest weights the loop produced.
SELFPLAY_S3=s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/selfplay/net-20x256/20260515T0157Z
CKPT_S3=$SELFPLAY_S3/final.pt
RESULT_KEY=$SELFPLAY_S3/eval_results-final.txt

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$CKPT_S3"
RESULT_KEY="$RESULT_KEY"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    aws s3 cp /var/log/eval.log "\${RESULT_KEY%.txt}.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull \$ECR_IMAGE

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e CKPT_S3=\$CKPT_S3 -e RESULT_KEY=\$RESULT_KEY \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    --entrypoint bash \\
    \$ECR_IMAGE -lc '
        set -ex
        cd /work/experiments/distill-soft
        CKPT_LOCAL=/work-tmp/\$(basename \$CKPT_S3)
        aws s3 cp \$CKPT_S3 \$CKPT_LOCAL --no-progress
        OUT=/work-tmp/eval_results-selfplay-final.txt
        echo "=== eval vs Stockfish UCI=1350 (sims=800) ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1350 --agent-device cuda 2>&1 | tee -a \$OUT
        echo "" | tee -a \$OUT
        echo "=== eval vs Stockfish UCI=1800 (sims=800) ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 800 --n-blocks 20 --n-filters 256 \\
            --stockfish-elo 1800 --agent-device cuda 2>&1 | tee -a \$OUT || echo "(uci1800 eval failed)" >> \$OUT
        aws s3 cp \$OUT \$RESULT_KEY --no-progress
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-selfplay-final},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
