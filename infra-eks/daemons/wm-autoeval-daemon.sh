#!/usr/bin/env bash
# Polls S3 for new checkpoints across all training runs. For each one
# that lacks a sibling eval_results.txt, launches a one-shot g6.4xlarge
# EC2 that:
#   1. Pulls the wm-chess-gpu image from ECR (always us-east-1, cross-region
#      pull works when the EC2 lives elsewhere)
#   2. Downloads the .pt from S3 (always us-east-1 bucket, cross-region read)
#   3. Runs eval.py vs Stockfish 1350 and depth=1 sub-tests
#   4. Uploads eval_results-<ckpt>.txt next to the ckpt
#   5. shutdown -h now -> instance terminates (initiated-shutdown=terminate)
#
# Multi-region: us-east-1 is the primary launch region; eu-central-1 is the
# fallback when us-east-1 hits the 32-vCPU G/VT quota. Each region has its
# own independent vCPU budget, so this doubles concurrent-eval capacity.
#
# Idempotent: each ckpt with an eval_results-<ckpt>.txt sibling is skipped.
# A claimed.<ckpt> marker prevents two pollers from double-launching; the
# marker is removed automatically on RunInstances failure.

set -u
LOG=/tmp/wm-autoeval-daemon.log
INTERVAL_SEC=${INTERVAL_SEC:-300}     # 5 min
ACCOUNT_ID=594561963943

# Where the ECR image + S3 buckets live. EC2s in any LAUNCH_REGION
# cross-region-pull the image and cross-region-read the buckets.
IMAGE_REGION=us-east-1
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com

INSTANCE_TYPE=g6.4xlarge   # 16 vCPU, 8 workers × 13 games ≈ 3x faster
INSTANCE_PROFILE=wm-chess-merge-instance-profile   # IAM is global

# Per-region/AZ settings — using a case in `region_config` instead of
# associative arrays so this works on macOS bash 3.2 (no `declare -A`).
#
# Each entry is "<region>:<az>:<market>". The chain is tried in order;
# first launch that succeeds wins. The pattern:
#   1. OD in the cheapest/stable region first (when G/VT quota has room)
#   2. Spot across multiple AZs to dodge g6.4xlarge capacity famines —
#      the famine is usually localized to one or two AZs at a time.
#
# Empirical: us-east-1 OD G quota is 32 vCPU; one g6.4xlarge eval = 16,
# plus running training boxes typically eat the rest. So OD is best-effort
# and spot is the workhorse. Adding many spot AZs is cheap (each failed
# launch is a fast API call) and gives us many shots at capacity.
REGION_ORDER=(
    # OD tier — first try, may fail on VcpuLimitExceeded.
    us-east-1:1b:on-demand
    eu-central-1:1a:on-demand
    # Spot, eu-central-1 — capacity error message suggested 1b/1c.
    eu-central-1:1b:spot
    eu-central-1:1c:spot
    eu-central-1:1a:spot
    # Spot, us-east-1 — all 6 AZs.
    us-east-1:1b:spot
    us-east-1:1a:spot
    us-east-1:1c:spot
    us-east-1:1d:spot
    us-east-1:1f:spot
)

region_config() {
    # Args: <region> <az>. Echoes "<subnet-id> <ami-id>" or empty on fail.
    local region="$1"
    local az="$2"
    local ami subnet
    case "$region" in
        us-east-1)
            ami=ami-027c3ae8019fc0d3a
            case "$az" in
                1a) subnet=subnet-05af0889ba2b8947a ;;
                1b) subnet=subnet-042fb3c497e2631a7 ;;
                1c) subnet=subnet-00be06b9c01d8b036 ;;
                1d) subnet=subnet-059b67fde3350d8b8 ;;
                1e) subnet=subnet-06273e78bc829cb29 ;;
                1f) subnet=subnet-04d064f932678aaa6 ;;
                *)  echo ""; return ;;
            esac ;;
        eu-central-1)
            ami=ami-01e9d13d4c5e54237
            case "$az" in
                1a) subnet=subnet-0a4bed60 ;;
                1b) subnet=subnet-fb7e6286 ;;
                1c) subnet=subnet-fbab48b7 ;;
                *)  echo ""; return ;;
            esac ;;
        *)
            echo ""; return ;;
    esac
    echo "$subnet $ami"
}

# Buckets we watch (always us-east-1 — cross-region reads are fine).
BUCKETS=(
    "wm-chess-library-594561963943"
)

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Try launching an eval EC2 in $region. On success: echo instance id.
# On failure: log + leave caller to decide whether to retry elsewhere.
try_launch_in_region() {
    local region_spec="$1"   # "<region>:<az>:<market>" e.g. eu-central-1:1b:spot
    local ckpt_s3="$2"
    local result_key="$3"
    local ckpt_basename="$4"
    # Parse "<region>:<az>:<market>" — read into three vars on colon.
    local region az market
    IFS=':' read -r region az market <<< "$region_spec"
    local cfg
    cfg=$(region_config "$region" "$az")
    if [ -z "$cfg" ]; then
        log "  $region_spec unknown region/az combo"
        return 1
    fi
    local subnet="${cfg%% *}"
    local ami="${cfg##* }"
    local market_args=()
    local tag_suffix=""
    if [ "$market" = "spot" ]; then
        market_args=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
        tag_suffix="-spot"
    fi

    # Parse the architecture out of the ckpt path:
    # s3://.../checkpoints/net-<blocks>x<filters>/<run-id>/distilled_epochNNN.pt
    # Falls back to 20x256 if the path doesn't match (oldest runs).
    local arch_part
    arch_part=$(echo "$ckpt_s3" | grep -oE 'net-[0-9]+x[0-9]+' | head -1)
    local n_blocks=20
    local n_filters=256
    if [ -n "$arch_part" ]; then
        n_blocks=$(echo "$arch_part" | sed -E 's/net-([0-9]+)x.*/\1/')
        n_filters=$(echo "$arch_part" | sed -E 's/net-[0-9]+x([0-9]+)/\1/')
    fi

    local user_data
    user_data=$(cat <<EOF
#!/bin/bash
exec > >(tee -a /var/log/eval.log) 2>&1
set -x

CKPT_S3="$ckpt_s3"
RESULT_KEY="$result_key"
ECR_IMAGE="$ECR/wm-chess-gpu:latest"

cleanup() {
    # Log + result both land in us-east-1 (single source of truth).
    aws s3 cp /var/log/eval.log "\${RESULT_KEY%.txt}.log" --region $IMAGE_REGION 2>&1 || true
    shutdown -h +1 || true
}
trap cleanup EXIT

systemctl enable --now docker
# ECR is in us-east-1 regardless of where this EC2 is running.
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
        OUT=/work-tmp/eval_results.txt
        echo "=== eval vs Stockfish UCI=1350 ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 800 --n-blocks $n_blocks --n-filters $n_filters \\
            --stockfish-elo 1350 --agent-device cuda 2>&1 | tee -a \$OUT
        echo "" | tee -a \$OUT
        echo "=== eval vs Stockfish UCI=1800 (calibrated, close to model strength) ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 800 --n-blocks $n_blocks --n-filters $n_filters \\
            --stockfish-elo 1800 --agent-device cuda 2>&1 | tee -a \$OUT || echo "(uci1800 eval failed; skipping)" >> \$OUT
        aws s3 cp \$OUT \$RESULT_KEY --no-progress
    '
EOF
)

    local launch_out
    launch_out=$(aws ec2 run-instances --region "$region" \
        --image-id "$ami" \
        --instance-type $INSTANCE_TYPE \
        --subnet-id "$subnet" \
        --iam-instance-profile Name=$INSTANCE_PROFILE \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
        --instance-initiated-shutdown-behavior terminate \
        "${market_args[@]}" \
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-eval-${ckpt_basename}${tag_suffix}},{Key=role,Value=wm-chess-eval}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1)
    local launch_rc=$?
    if [ $launch_rc -eq 0 ] && ! echo "$launch_out" | grep -q 'An error occurred'; then
        echo "$launch_out"
        return 0
    fi
    log "  $region_spec launch failed: $launch_out"
    return 1
}

launch_eval_instance() {
    local ckpt_s3="$1"
    local ckpt_basename
    ckpt_basename=$(basename "$ckpt_s3")
    local ckpt_dir_s3="${ckpt_s3%/*}"
    local result_key="$ckpt_dir_s3/eval_results-${ckpt_basename%.pt}.txt"
    local claim_key="$ckpt_dir_s3/.claimed-eval-${ckpt_basename%.pt}"

    # Idempotency: skip if result OR a claim exists.
    if aws s3 ls "$result_key" >/dev/null 2>&1; then
        log "SKIP $ckpt_s3 — eval_results already exists"
        return
    fi
    if aws s3 ls "$claim_key" >/dev/null 2>&1; then
        log "SKIP $ckpt_s3 — another runner has claimed it"
        return
    fi
    echo "claimed-by=$(hostname) at=$(date -u +%FT%TZ)" | \
        aws s3 cp - "$claim_key" 2>&1 | tail -1

    log "LAUNCH eval EC2 for $ckpt_s3"

    # Walk the region list; first success wins.
    local instance_id=""
    for region_spec in "${REGION_ORDER[@]}"; do
        if instance_id=$(try_launch_in_region "$region_spec" "$ckpt_s3" "$result_key" "$ckpt_basename"); then
            log "  launched $instance_id in $region_spec"
            return 0
        fi
    done

    # All regions failed — release the claim so the next poll retries.
    log "  all regions failed, removing claim marker for retry"
    aws s3 rm "$claim_key" 2>&1 | tail -1 | tee -a "$LOG"
}

# How often to eval selfplay iters. Distill epochs are evaluated every
# epoch; selfplay iters land much more frequently (~80 min vs ~2h) so we
# downsample. 2 = eval iter 0, 2, 4, ...; 4 = eval iter 0, 4, 8, ...
SELFPLAY_EVAL_EVERY=${SELFPLAY_EVAL_EVERY:-2}

log "=== started; poll every ${INTERVAL_SEC}s; selfplay eval cadence = every ${SELFPLAY_EVAL_EVERY} iters ==="

while true; do
    for bucket in "${BUCKETS[@]}"; do
        # One scan, two filters. distill = all epochs; selfplay = downsampled.
        all_pt=$(aws s3 ls "s3://$bucket/" --recursive 2>/dev/null | awk '{print $NF}')

        distill_ckpts=$(echo "$all_pt" \
                | grep -E 'checkpoints/net-[^/]+/[^/]+/distilled_epoch[0-9]+\.pt$' \
                | sort -u)

        selfplay_ckpts=$(echo "$all_pt" \
                | grep -E 'selfplay/net-[^/]+/[^/]+/net_iter[0-9]+\.pt$' \
                | awk -F'net_iter' -v every="$SELFPLAY_EVAL_EVERY" '{
                    iter_str = $2;
                    sub(/\.pt$/, "", iter_str);
                    if ((iter_str + 0) % every == 0) print $0;
                  }' \
                | sort -u)

        for key in $distill_ckpts $selfplay_ckpts; do
            launch_eval_instance "s3://$bucket/$key"
        done
    done
    sleep "$INTERVAL_SEC"
done
