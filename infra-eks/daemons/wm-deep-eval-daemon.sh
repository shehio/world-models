#!/usr/bin/env bash
# Polls S3 for sims=800 eval results (the autoeval daemon's output).
# When a ckpt's sims=800 Elo crosses $ELO_THRESHOLD, fires a sims=4000
# eval EC2 to get a sharper measurement.
#
# Why: sims=800 systematically under-reports strength vs sims=4000 (we
# measured +277 Elo on d15 ep 20 between the two). The autoeval daemon
# gives a cheap learning curve; this daemon converts only the *promising*
# ckpts into the headline sims=4000 numbers without bleeding budget on
# every epoch.
#
# Idempotent:
#   - eval_results-<ckpt>-sims4000.txt sibling → permanent skip
#   - .claimed-sims4000-<ckpt> marker → prevent double-launch
# Parses the UCI=1800 Elo line:
#   "Agent absolute Elo (anchor 1800): 1892 [1826, 1966]"

set -u
LOG=/tmp/wm-deep-eval-daemon.log
INTERVAL_SEC=${INTERVAL_SEC:-300}     # 5 min
ELO_THRESHOLD=${ELO_THRESHOLD:-1950}  # only fire sims=4000 above this
ACCOUNT_ID=594561963943

IMAGE_REGION=us-east-1
ECR=$ACCOUNT_ID.dkr.ecr.$IMAGE_REGION.amazonaws.com
INSTANCE_TYPE=g6.4xlarge
INSTANCE_PROFILE=wm-chess-merge-instance-profile

REGION_ORDER=(us-east-1 eu-central-1)
region_config() {
    case "$1" in
        us-east-1)    echo "subnet-042fb3c497e2631a7 ami-027c3ae8019fc0d3a" ;;
        eu-central-1) echo "subnet-0a4bed60 ami-01e9d13d4c5e54237" ;;
        *)            echo "" ;;
    esac
}

BUCKETS=(
    "wm-chess-library-594561963943"
)

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Parse the UCI=1800 sims=800 Elo point estimate from an eval_results txt.
# Returns "" if not found.
parse_elo_uci1800() {
    local result_text="$1"
    echo "$result_text" \
        | grep -oE 'Agent absolute Elo \(anchor 1800\): [0-9]+' \
        | head -1 \
        | grep -oE '[0-9]+$'
}

try_launch_in_region() {
    local region="$1"
    local ckpt_s3="$2"
    local result_key="$3"
    local ckpt_basename="$4"
    local cfg
    cfg=$(region_config "$region")
    local subnet="${cfg%% *}"
    local ami="${cfg##* }"

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
        OUT=/work-tmp/eval_results-sims4000.txt
        echo "=== eval vs Stockfish UCI=1350 (sims=4000) ===" | tee \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 4000 --n-blocks $n_blocks --n-filters $n_filters \\
            --stockfish-elo 1350 --agent-device cuda 2>&1 | tee -a \$OUT
        echo "" | tee -a \$OUT
        echo "=== eval vs Stockfish UCI=1800 (sims=4000) ===" | tee -a \$OUT
        python scripts/eval.py \\
            --ckpt \$CKPT_LOCAL --workers 8 --games-per-worker 13 \\
            --sims 4000 --n-blocks $n_blocks --n-filters $n_filters \\
            --stockfish-elo 1800 --agent-device cuda 2>&1 | tee -a \$OUT || echo "(uci1800 eval failed)" >> \$OUT
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
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-deep-eval-$ckpt_basename},{Key=role,Value=wm-chess-eval}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1)
    local launch_rc=$?
    if [ $launch_rc -eq 0 ] && ! echo "$launch_out" | grep -q 'An error occurred'; then
        echo "$launch_out"
        return 0
    fi
    log "  $region launch failed: $launch_out"
    return 1
}

# Given an eval_results-<ckpt>.txt S3 key, decide whether to fire sims=4000.
maybe_fire_sims4000() {
    local result_s3="$1"   # s3://.../eval_results-distilled_epochNNN.txt
    local ckpt_dir_s3="${result_s3%/*}"
    local result_basename
    result_basename=$(basename "$result_s3")
    # Strip leading "eval_results-" and trailing ".txt" -> "distilled_epochNNN"
    local ckpt_stem="${result_basename#eval_results-}"
    ckpt_stem="${ckpt_stem%.txt}"
    local ckpt_basename="${ckpt_stem}.pt"
    local ckpt_s3="$ckpt_dir_s3/$ckpt_basename"

    local sims4000_key="$ckpt_dir_s3/eval_results-${ckpt_stem}-sims4000.txt"
    local claim_key="$ckpt_dir_s3/.claimed-sims4000-${ckpt_stem}"

    # Idempotency: skip if final result OR claim exists.
    if aws s3 ls "$sims4000_key" >/dev/null 2>&1; then
        return
    fi
    if aws s3 ls "$claim_key" >/dev/null 2>&1; then
        return
    fi

    # Download + parse sims=800 Elo.
    local txt
    txt=$(aws s3 cp "$result_s3" - 2>/dev/null)
    local elo
    elo=$(parse_elo_uci1800 "$txt")
    if [ -z "$elo" ]; then
        log "SKIP $result_s3 — could not parse UCI=1800 Elo"
        return
    fi
    if [ "$elo" -lt "$ELO_THRESHOLD" ]; then
        return
    fi

    log "MATCH $result_s3 — Elo=$elo >= $ELO_THRESHOLD; will fire sims=4000"
    echo "claimed-by=$(hostname) elo=$elo at=$(date -u +%FT%TZ)" | \
        aws s3 cp - "$claim_key" 2>&1 | tail -1

    local instance_id=""
    for region in "${REGION_ORDER[@]}"; do
        if instance_id=$(try_launch_in_region "$region" "$ckpt_s3" "$sims4000_key" "$ckpt_basename"); then
            log "  launched $instance_id in $region"
            return 0
        fi
    done

    log "  all regions failed, removing claim marker for retry"
    aws s3 rm "$claim_key" 2>&1 | tail -1 | tee -a "$LOG"
}

log "=== started; threshold=$ELO_THRESHOLD (UCI=1800 sims=800); poll every ${INTERVAL_SEC}s ==="

while true; do
    for bucket in "${BUCKETS[@]}"; do
        results=$(aws s3 ls "s3://$bucket/" --recursive 2>/dev/null \
                  | awk '{print $NF}' \
                  | grep -E 'checkpoints/net-[^/]+/[^/]+/eval_results-distilled_epoch[0-9]+\.txt$' \
                  | sort -u)
        for key in $results; do
            maybe_fire_sims4000 "s3://$bucket/$key"
        done
    done
    sleep "$INTERVAL_SEC"
done
