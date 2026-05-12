#!/usr/bin/env bash
# Two-box split pipeline — the cheapest way to run 02c end-to-end on AWS.
#
# Why this exists
# ---------------
# run_overnight.sh runs all three phases on one box. ~60% of the bill in
# the 02c-30ep run went to GPU idle time during the CPU-bound phases
# (data gen, eval). This script provisions a CPU-only spot box for those
# and a single-GPU spot box for training, sequentially:
#
#     Phase 1 (data gen)  →  c7i.8xlarge spot (CPU, 32 vCPU)
#     Phase 2 (training)  →  g5.xlarge   spot (1 A10G GPU)
#     Phase 3 (eval x2)   →  c7i.8xlarge spot (CPU, 32 vCPU)
#
# Handoffs go through the laptop: each phase rsyncs the small artifact
# (dataset NPZ ~1.5GB; checkpoint .pt ~80MB) back home, then the next
# phase rsyncs it onto the new box. No S3 needed.
#
# Estimated cost (us-east-1 spot, 2026 rates):
#     Phase 1   ~28 min  c7i.8xlarge $0.34/h  → $0.16
#     Phase 2   ~75 min  g5.xlarge   $0.30/h  → $0.38
#     Phase 3   ~90 min  c7i.8xlarge $0.34/h  → $0.51
#     bootstrap 3 × ~3min lost                → $0.10
#     ────────────────────────────────────────────────
#     Total                                   → ~$1.15
#
# Compare to run_overnight.sh on g5.8xlarge: ~$9. ~7–8× cheaper.
#
# Usage
# -----
#   KEY_NAME=mykey bash scripts/run_split_pipeline.sh
#
# Resuming from a specific phase (artifacts must exist locally):
#   KEY_NAME=mykey START_PHASE=2 bash scripts/run_split_pipeline.sh
#
# Required env:
#   KEY_NAME                AWS EC2 key pair name (must exist in REGION)
#
# Optional env (with defaults):
#   REGION=us-east-1
#   CPU_INSTANCE=c7i.8xlarge
#   GPU_INSTANCE=g5.xlarge
#   USE_SPOT=true
#   N_GAMES=4000  DEPTH=10  MULTIPV=8  TEMP_PAWNS=1.0  CHUNK_SIZE=50
#   EPOCHS=30  N_BLOCKS=20  N_FILTERS=256  HARD_TARGETS=false
#   EVAL_GAMES_TOTAL=120  SIMS_BASELINE=800  SIMS_STRETCH=1600  SF_ELO=1350
#   START_PHASE=1     (1, 2, or 3 — useful for resuming)
#   STOP_AFTER_PHASE  (1, 2, or 3 — useful for partial runs)
#   LOCAL_DATA=data/sf_d10_mpv8_4000g.npz
#   LOCAL_CKPT=checkpoints/run01/distilled_epoch029.pt
#
# Prereqs
# -------
#   - terraform installed
#   - aws CLI configured with credentials
#   - SSH private key at ~/.ssh/${KEY_NAME}.pem (chmod 600)
#   - infra/ directory at repo root (it is)

set -uo pipefail

# ---------- resolve paths ----------
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
INFRA_DIR="$REPO_ROOT/infra"

cd "$PROJECT_ROOT"

# ---------- config defaults ----------
: "${KEY_NAME:?Set KEY_NAME=<your-aws-key-pair-name>}"
: "${REGION:=us-east-1}"
: "${CPU_INSTANCE:=c7i.8xlarge}"
: "${GPU_INSTANCE:=g5.xlarge}"
: "${USE_SPOT:=true}"

: "${N_GAMES:=4000}"
: "${DEPTH:=10}"
: "${MULTIPV:=8}"
: "${TEMP_PAWNS:=1.0}"
: "${CHUNK_SIZE:=50}"

: "${EPOCHS:=30}"
: "${N_BLOCKS:=20}"
: "${N_FILTERS:=256}"
: "${HARD_TARGETS:=false}"

: "${EVAL_GAMES_TOTAL:=120}"
: "${SIMS_BASELINE:=800}"
: "${SIMS_STRETCH:=1600}"
: "${SF_ELO:=1350}"

: "${START_PHASE:=1}"
: "${STOP_AFTER_PHASE:=3}"

: "${LOCAL_DATA:=data/sf_d${DEPTH}_mpv${MULTIPV}_g${N_GAMES}.npz}"
: "${LOCAL_CKPT_DIR:=checkpoints/split-run-$(date -u +%Y%m%dT%H%MZ)}"
LOCAL_CKPT_DIR_ABS="$PROJECT_ROOT/$LOCAL_CKPT_DIR"
LOCAL_CKPT="$LOCAL_CKPT_DIR/distilled_epoch$(printf '%03d' $((EPOCHS - 1))).pt"

KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ConnectTimeout=10)
RSYNC_EXCLUDES=(
    --exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache'
    --exclude='data/' --exclude='checkpoints/' --exclude='aws_results/'
    --exclude='.git/'
)

# ---------- helpers ----------
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

terraform_up() {
    local instance_type="$1"
    log "terraform apply (instance_type=$instance_type)"
    local my_cidr
    my_cidr="$(curl -s -4 ifconfig.me)/32"
    cd "$INFRA_DIR"
    terraform apply -auto-approve \
        -var "key_name=$KEY_NAME" \
        -var "region=$REGION" \
        -var "instance_type=$instance_type" \
        -var "use_spot=$USE_SPOT" \
        -var "allowed_ssh_cidr=$my_cidr" \
        > /tmp/tf_apply.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "terraform apply FAILED (rc=$rc). Tail:"
        tail -20 /tmp/tf_apply.log
        cd "$PROJECT_ROOT"
        return $rc
    fi
    PUBLIC_IP="$(terraform output -raw public_ip)"
    cd "$PROJECT_ROOT"
    log "instance up: $PUBLIC_IP"
}

terraform_down() {
    log "terraform destroy"
    cd "$INFRA_DIR"
    terraform destroy -auto-approve \
        -var "key_name=$KEY_NAME" \
        -var "region=$REGION" \
        > /tmp/tf_destroy.log 2>&1 || {
            log "terraform destroy FAILED — CHECK MANUALLY: $INFRA_DIR/terraform.tfstate"
            tail -20 /tmp/tf_destroy.log
        }
    cd "$PROJECT_ROOT"
}

wait_for_bootstrap() {
    local ip="$1"
    log "waiting for /home/ubuntu/BOOTSTRAP_COMPLETE on $ip ..."
    local deadline=$(( $(date +%s) + 600 ))
    while [ $(date +%s) -lt $deadline ]; do
        if ssh "${SSH_OPTS[@]}" "ubuntu@$ip" 'test -f /home/ubuntu/BOOTSTRAP_COMPLETE' 2>/dev/null; then
            log "bootstrap complete on $ip"
            return 0
        fi
        sleep 10
    done
    log "FATAL: bootstrap did not complete within 10 min on $ip"
    return 1
}

rsync_project_up() {
    local ip="$1"
    log "rsync project → $ip"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCLUDES[@]}" \
        "$PROJECT_ROOT/" "ubuntu@$ip:~/02c-distill-scaled/"
    log "uv sync on $ip"
    ssh "${SSH_OPTS[@]}" "ubuntu@$ip" \
        'cd ~/02c-distill-scaled && ~/.local/bin/uv sync' > /tmp/uv_sync.log 2>&1
}

rsync_artifact_up() {
    local ip="$1"; local local_path="$2"; local remote_path="$3"
    log "rsync artifact → $ip:$remote_path"
    ssh "${SSH_OPTS[@]}" "ubuntu@$ip" "mkdir -p ~/02c-distill-scaled/$(dirname "$remote_path")"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$local_path" "ubuntu@$ip:~/02c-distill-scaled/$remote_path"
}

rsync_artifact_down() {
    local ip="$1"; local remote_path="$2"; local local_path="$3"
    log "rsync artifact ← $ip:$remote_path"
    mkdir -p "$(dirname "$local_path")"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$ip:~/02c-distill-scaled/$remote_path" "$local_path"
}

# Run a command on the box, streaming output back here.
remote_run() {
    local ip="$1"; shift
    ssh "${SSH_OPTS[@]}" "ubuntu@$ip" "cd ~/02c-distill-scaled && $*"
}

# ---------- pipeline ----------
phase_1_datagen() {
    log "=== PHASE 1: data generation on $CPU_INSTANCE ==="
    terraform_up "$CPU_INSTANCE" || return 1
    wait_for_bootstrap "$PUBLIC_IP" || { terraform_down; return 1; }
    rsync_project_up "$PUBLIC_IP"

    log "running generate_data.py — N=$N_GAMES depth=$DEPTH multipv=$MULTIPV"
    local n_workers
    n_workers="$(ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" 'nproc')"
    # Leave 2 cores for system + orchestration overhead.
    n_workers=$((n_workers - 2))
    [ $n_workers -lt 1 ] && n_workers=1

    remote_run "$PUBLIC_IP" "~/.local/bin/uv run python scripts/generate_data.py \
        --n-games $N_GAMES --workers $n_workers \
        --depth $DEPTH --multipv $MULTIPV \
        --temperature-pawns $TEMP_PAWNS \
        --chunk-size $CHUNK_SIZE \
        --output $LOCAL_DATA \
        --progress-dir data/ 2>&1 | tee /tmp/phase1.log" || {
            log "Phase 1 FAILED — attempting to pull whatever chunks survived"
            rsync_artifact_down "$PUBLIC_IP" "data/" "$PROJECT_ROOT/data/" || true
            terraform_down
            return 1
        }

    rsync_artifact_down "$PUBLIC_IP" "$LOCAL_DATA" "$PROJECT_ROOT/$LOCAL_DATA"
    rsync_artifact_down "$PUBLIC_IP" "${LOCAL_DATA%.npz}.pgn" "$PROJECT_ROOT/${LOCAL_DATA%.npz}.pgn" || true
    rsync_artifact_down "$PUBLIC_IP" "/tmp/phase1.log" "$LOCAL_CKPT_DIR_ABS/phase1.log" || true

    terraform_down
    log "=== PHASE 1 DONE — dataset at $LOCAL_DATA ==="
}

phase_2_training() {
    log "=== PHASE 2: training on $GPU_INSTANCE ==="
    if [ ! -f "$PROJECT_ROOT/$LOCAL_DATA" ]; then
        log "FATAL: $LOCAL_DATA missing — run Phase 1 first or set START_PHASE=1"
        return 1
    fi

    terraform_up "$GPU_INSTANCE" || return 1
    wait_for_bootstrap "$PUBLIC_IP" || { terraform_down; return 1; }
    rsync_project_up "$PUBLIC_IP"
    rsync_artifact_up "$PUBLIC_IP" "$PROJECT_ROOT/$LOCAL_DATA" "$LOCAL_DATA"

    local hard_flag=""
    [ "$HARD_TARGETS" = "true" ] && hard_flag="--hard-targets"

    remote_run "$PUBLIC_IP" "~/.local/bin/uv run python scripts/train.py \
        --data $LOCAL_DATA \
        --epochs $EPOCHS \
        --device cuda \
        --n-blocks $N_BLOCKS --n-filters $N_FILTERS \
        --ckpt-dir $LOCAL_CKPT_DIR \
        --save-every 5 $hard_flag 2>&1 | tee /tmp/phase2.log" || {
            log "Phase 2 FAILED — pulling any partial checkpoints"
            rsync_artifact_down "$PUBLIC_IP" "$LOCAL_CKPT_DIR/" "$LOCAL_CKPT_DIR_ABS/" || true
            terraform_down
            return 1
        }

    rsync_artifact_down "$PUBLIC_IP" "$LOCAL_CKPT_DIR/" "$LOCAL_CKPT_DIR_ABS/"
    rsync_artifact_down "$PUBLIC_IP" "/tmp/phase2.log" "$LOCAL_CKPT_DIR_ABS/phase2.log" || true

    terraform_down
    log "=== PHASE 2 DONE — checkpoint at $LOCAL_CKPT ==="
}

phase_3_eval() {
    log "=== PHASE 3: eval (2 sim budgets) on $CPU_INSTANCE ==="
    if [ ! -f "$PROJECT_ROOT/$LOCAL_CKPT" ]; then
        log "FATAL: $LOCAL_CKPT missing — run Phase 2 first or set START_PHASE=2"
        return 1
    fi

    terraform_up "$CPU_INSTANCE" || return 1
    wait_for_bootstrap "$PUBLIC_IP" || { terraform_down; return 1; }
    rsync_project_up "$PUBLIC_IP"
    rsync_artifact_up "$PUBLIC_IP" "$PROJECT_ROOT/$LOCAL_CKPT" "$LOCAL_CKPT"

    local n_workers
    n_workers="$(ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" 'nproc')"
    n_workers=$((n_workers - 2))
    [ $n_workers -lt 1 ] && n_workers=1
    local games_per_worker=$(( (EVAL_GAMES_TOTAL + n_workers - 1) / n_workers ))

    for sims in "$SIMS_BASELINE" "$SIMS_STRETCH"; do
        log "Phase 3 eval @ $sims sims"
        remote_run "$PUBLIC_IP" "~/.local/bin/uv run python scripts/eval.py \
            --ckpt $LOCAL_CKPT \
            --workers $n_workers --games-per-worker $games_per_worker \
            --sims $sims \
            --n-blocks $N_BLOCKS --n-filters $N_FILTERS \
            --agent-device cpu \
            --stockfish-elo $SF_ELO 2>&1 | tee /tmp/phase3_${sims}sims.log" || {
                log "Phase 3 ($sims sims) FAILED — continuing"
            }
        rsync_artifact_down "$PUBLIC_IP" "/tmp/phase3_${sims}sims.log" \
            "$LOCAL_CKPT_DIR_ABS/phase3_${sims}sims.log" || true
    done

    terraform_down
    log "=== PHASE 3 DONE — logs in $LOCAL_CKPT_DIR_ABS/ ==="
}

# ---------- driver ----------
log "split pipeline starting | CPU=$CPU_INSTANCE  GPU=$GPU_INSTANCE  REGION=$REGION"
log "artifacts -> $LOCAL_CKPT_DIR_ABS"
mkdir -p "$LOCAL_CKPT_DIR_ABS"

# Make sure any stale terraform state is empty before we start.
if [ "$START_PHASE" = "1" ] && [ -f "$INFRA_DIR/terraform.tfstate" ]; then
    if (cd "$INFRA_DIR" && terraform state list 2>/dev/null | grep -q aws_instance); then
        log "WARNING: pre-existing instance in terraform state. Destroying first."
        terraform_down
    fi
fi

if [ "$START_PHASE" -le 1 ] && [ "$STOP_AFTER_PHASE" -ge 1 ]; then phase_1_datagen || exit 1; fi
if [ "$START_PHASE" -le 2 ] && [ "$STOP_AFTER_PHASE" -ge 2 ]; then phase_2_training || exit 1; fi
if [ "$START_PHASE" -le 3 ] && [ "$STOP_AFTER_PHASE" -ge 3 ]; then phase_3_eval     || exit 1; fi

log "split pipeline COMPLETE. Artifacts under: $LOCAL_CKPT_DIR_ABS"
log "Reminder: terraform destroy ran after each phase — no resources should be billing."
log "To verify: aws ec2 describe-instances --region $REGION --filters Name=instance-state-name,Values=running"
