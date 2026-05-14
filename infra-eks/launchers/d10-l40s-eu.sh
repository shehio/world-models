#!/usr/bin/env bash
# One-shot launcher: bring up a g6e.xlarge (L40S) in eu-central-1
# running the d10 training entrypoint. Used when the k8s cluster
# can't get L40S capacity in us-east-1 and we need to bypass EKS.
#
# Cross-region wiring (same as the eval daemon):
#   - EC2 lives in eu-central-1
#   - Image cross-region-pulled from us-east-1 ECR
#   - S3 reads/writes target us-east-1 buckets (npz + ckpts)

set -euo pipefail

ACCOUNT_ID=594561963943
IMAGE_REGION=us-east-1
LAUNCH_REGION=eu-central-1
AMI=ami-01e9d13d4c5e54237             # DL Base GPU AL2023, eu-central-1
SUBNET=subnet-0a4bed60                 # eu-central-1a default
INSTANCE_TYPE=g6e.xlarge               # L40S, 4 vCPU, 32 GB RAM
INSTANCE_PROFILE=wm-chess-merge-instance-profile
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

# Training config (mirrors the run-train.sh defaults + d10 overrides).
S3_BUCKET=wm-chess-library-594561963943
S3_PREFIX=d10-mpv8-T1-g250000-20260513T0615Z
INIT_FROM_S3=s3://$S3_BUCKET/$S3_PREFIX/checkpoints/net-20x256/20260514T1030Z/distilled_epoch004.pt
START_EPOCH=5
RUN_ID=$(date -u +%Y%m%dT%H%MZ)

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/train.log) 2>&1
set -x

cleanup() {
    aws s3 cp /var/log/train.log "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/net-20x256/$RUN_ID/host-launch.log" --region $IMAGE_REGION 2>&1 || true
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
    -e BATCH_SIZE=1024 \\
    -e SAVE_EVERY=5 \\
    -e N_BLOCKS=20 \\
    -e N_FILTERS=256 \\
    -e MAX_POSITIONS=5000000 \\
    -e IN_RAM=1 \\
    -e AMP=1 \\
    -e COMPILE=0 \\
    -e INIT_FROM_S3=$INIT_FROM_S3 \\
    -e START_EPOCH=$START_EPOCH \\
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-d10-l40s-eu-central},{Key=role,Value=wm-chess-train}]" \
    --query 'Instances[0].[InstanceId,State.Name]' --output text
