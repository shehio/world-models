#!/usr/bin/env bash
# Experiment A: train a deeper 40-block ResNet (same 256 channels, ~50M
# params vs the 23.7M of the 20x256 we used so far) on the d15 dataset.
# If this plateaus at ~1800 Elo too, we're data-limited; if it climbs,
# we were capacity-limited.

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=eu-central-1
AMI=ami-01e9d13d4c5e54237             # DL Base GPU AL2023 eu-central-1
SUBNET=subnet-0a4bed60                 # eu-central-1a default
INSTANCE_TYPE=g6e.xlarge               # L40S, 4 vCPU, 32 GB RAM
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d15-mpv8-T1-g100000-20260512T1844Z
RUN_ID=$(date -u +%Y%m%dT%H%MZ)

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/train.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/train.log "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/net-40x256/$RUN_ID/host-launch.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
aws ecr get-login-password --region $IMAGE_REGION | docker login --username AWS --password-stdin $ECR
docker pull $ECR/wm-chess-gpu:latest

mkdir -p /mnt/work
docker run --rm \\
    --gpus all \\
    -v /mnt/work:/work-tmp \\
    -e AWS_DEFAULT_REGION=$IMAGE_REGION \\
    -e S3_BUCKET=$S3_BUCKET \\
    -e S3_PREFIX=$S3_PREFIX \\
    -e RUN_ID=$RUN_ID \\
    -e EPOCHS=20 \\
    -e BATCH_SIZE=512 \\
    -e SAVE_EVERY=5 \\
    -e N_BLOCKS=40 \\
    -e N_FILTERS=256 \\
    -e MAX_POSITIONS=5000000 \\
    -e IN_RAM=1 \\
    -e AMP=1 \\
    -e COMPILE=0 \\
    --entrypoint /entrypoint-train.sh \\
    $ECR/wm-chess-gpu:latest
EOF
)

aws ec2 run-instances --region $LAUNCH_REGION \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET \
    --iam-instance-profile Name=$INSTANCE_PROFILE \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-d15-40x256-eu-central},{Key=role,Value=wm-chess-train}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
