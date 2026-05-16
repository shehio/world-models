#!/usr/bin/env bash
# Run elo-bisect across every (run × epoch) checkpoint we have:
#   d15 5M baseline (20×256): ep5, 10, 15, 20
#   d15 5M capacity (40×256): ep5, 10, 15, 20
#   d10 5M subset (20×256):   ep5, 10, 15, 20
#   d10 30M full (20×256):    ep5, 10, 15, 20
# Total: 16 checkpoints. Each bisect is ~3-4 probes of 40 games at
# sims=800 → ~15-20 min per checkpoint on a g6.4xlarge. Wallclock ~5h,
# cost ~$2.
#
# Results land in S3 alongside each checkpoint as bisect_results-*.json.

set -euo pipefail

ACCOUNT_ID=594561963943
REGION=us-east-1
AMI=ami-027c3ae8019fc0d3a
SUBNET=subnet-042fb3c497e2631a7
INSTANCE_TYPE=g6.4xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
RUN_TS=$(date -u +%Y%m%dT%H%MZ)

# The inner script (what runs inside the docker container). Single-quoted
# heredoc → no expansion at this layer; all $VAR refs are literal until
# the container bash sees them.
INNER_SCRIPT=$(cat <<'INNER'
set -eux
cd /work/experiments/distill-soft

# Each entry: <s3_prefix>;<n_blocks>;<label>
CKPTS=(
  "s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/checkpoints/net-20x256/20260514T1332Z;20;d15-baseline-20x256"
  "s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/checkpoints/net-40x256/20260514T1748Z;40;d15-capacity-40x256"
  "s3://wm-chess-library-594561963943/d10-mpv8-T1-g250000-20260513T0615Z/checkpoints/net-20x256/20260514T1606Z;20;d10-5M-subset"
  "s3://wm-chess-library-594561963943/d10-mpv8-T1-g250000-20260513T0615Z/checkpoints/net-20x256/20260514T2332Z-full30M;20;d10-30M-full"
)

for entry in "${CKPTS[@]}"; do
  IFS=";" read -r prefix nblocks label <<< "$entry"
  for ep in 004 009 014 019; do
    ckpt_s3="$prefix/distilled_epoch$ep.pt"
    out_s3="$prefix/bisect_results-distilled_epoch$ep.json"
    local_ckpt="/work-tmp/$label-ep$ep.pt"
    local_out="/work-tmp/$label-ep$ep.json"
    echo "=== bisect $label ep$ep ==="
    aws s3 cp "$ckpt_s3" "$local_ckpt" --no-progress || { echo "FAIL: $ckpt_s3 not found"; continue; }
    python scripts/elo_bisect.py \
        --ckpt "$local_ckpt" \
        --n-blocks "$nblocks" --n-filters 256 \
        --workers 8 --games-per-probe 40 --sims 800 \
        --out "$local_out" 2>&1 | tail -30
    aws s3 cp "$local_out" "$out_s3" --no-progress
    rm -f "$local_ckpt"
  done
done
echo "ALL BISECTS DONE"
INNER
)

# Outer user_data — the only $-expansion happening at heredoc-write time
# is for the launcher-level vars ($ECR, $REGION, $ACCOUNT_ID, $RUN_TS,
# $INNER_SCRIPT). Everything inside $INNER_SCRIPT is already literal.
USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/bisect.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/bisect.log s3://wm-chess-library-${ACCOUNT_ID}/bisect-runs/${RUN_TS}/bisect.log --region ${REGION} 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR}
docker pull ${ECR}/wm-chess-gpu:latest

mkdir -p /mnt/work
cat > /mnt/work/inner.sh <<'INNERSH'
${INNER_SCRIPT}
INNERSH
chmod +x /mnt/work/inner.sh

docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=${REGION} \\
    --entrypoint bash \\
    ${ECR}/wm-chess-gpu:latest /work-tmp/inner.sh
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-bisect-all-ckpts},{Key=role,Value=wm-chess-eval}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text

echo "Results will land alongside each checkpoint as bisect_results-distilled_epochNNN.json"
echo "Master log: s3://wm-chess-library-${ACCOUNT_ID}/bisect-runs/${RUN_TS}/bisect.log"
