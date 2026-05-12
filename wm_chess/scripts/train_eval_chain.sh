#!/usr/bin/env bash
# Train + eval pipeline for one dataset from library/games/.
#
# Picks up where overnight_datagen.sh left off: takes a library path (or
# resolves one from config) and runs:
#   Phase 2: g5.xlarge spot → train → ckpt
#   Phase 3: c7i.8xlarge spot → eval at 800 sims + 1600 sims
#
# Each phase ends with terraform destroy. Idempotent: if a final ckpt
# already exists locally for this run, training is skipped.
#
# Usage (single dataset):
#   KEY_NAME=mykey LIB_PATH=library/games/sf-18/d10-mpv8-T1/g4000-seed42 \
#     bash wm_chess/scripts/train_eval_chain.sh
#
# Usage (all datasets in library that don't yet have a trained ckpt):
#   KEY_NAME=mykey ALL=1 bash wm_chess/scripts/train_eval_chain.sh
#
# Env (all optional):
#   EPOCHS=30  N_BLOCKS=20  N_FILTERS=256
#   HARD_TARGETS=false   ← set true for Experiment A (ablation)
#   SIMS_BASELINE=800  SIMS_STRETCH=1600  SF_ELO=1350
#   EVAL_GAMES_TOTAL=120

set -uo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
INFRA_DIR="$REPO_ROOT/infra"
LIB_DIR="$REPO_ROOT/library"

: "${KEY_NAME:?Set KEY_NAME=<your-aws-key-pair-name>}"
: "${REGION:=us-east-1}"
: "${CPU_INSTANCE:=c7i.8xlarge}"
: "${GPU_INSTANCE:=g5.xlarge}"
: "${USE_SPOT:=true}"

: "${EPOCHS:=30}"
: "${N_BLOCKS:=20}"
: "${N_FILTERS:=256}"
: "${HARD_TARGETS:=false}"

: "${EVAL_GAMES_TOTAL:=120}"
: "${SIMS_BASELINE:=800}"
: "${SIMS_STRETCH:=1600}"
: "${SF_ELO:=1350}"
: "${ALL:=}"

KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ConnectTimeout=10)
RSYNC_EXCL=(--exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache' --exclude='.git/' --exclude='data/' --exclude='checkpoints/')

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

terraform_up() {
    local instance_type="$1"
    local my_cidr; my_cidr="$(curl -s -4 ifconfig.me)/32"
    cd "$INFRA_DIR"
    terraform apply -auto-approve \
        -var "key_name=$KEY_NAME" -var "region=$REGION" \
        -var "instance_type=$instance_type" -var "use_spot=$USE_SPOT" \
        -var "allowed_ssh_cidr=$my_cidr" > /tmp/tf_apply.log 2>&1 || {
            log "terraform apply FAILED ($instance_type)"; tail -15 /tmp/tf_apply.log; cd "$REPO_ROOT"; return 1; }
    PUBLIC_IP="$(terraform output -raw public_ip)"
    cd "$REPO_ROOT"; log "instance up ($instance_type): $PUBLIC_IP"
}

terraform_down() {
    cd "$INFRA_DIR"
    terraform destroy -auto-approve \
        -var "key_name=$KEY_NAME" -var "region=$REGION" \
        > /tmp/tf_destroy.log 2>&1 || {
            log "terraform destroy FAILED — CHECK MANUALLY"; tail -10 /tmp/tf_destroy.log; }
    cd "$REPO_ROOT"
}

wait_bootstrap() {
    local ip="$1"; log "waiting for BOOTSTRAP_COMPLETE on $ip"
    local deadline=$(( $(date +%s) + 600 ))
    while [ $(date +%s) -lt $deadline ]; do
        if ssh "${SSH_OPTS[@]}" "ubuntu@$ip" 'test -f /home/ubuntu/BOOTSTRAP_COMPLETE' 2>/dev/null; then
            log "bootstrap complete"; return 0; fi
        sleep 10
    done
    log "FATAL: bootstrap timeout"; return 1
}

rsync_project_up() {
    local ip="$1"
    log "rsync wm_chess + distill-soft → $ip"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCL[@]}" "$REPO_ROOT/wm_chess/" "ubuntu@$ip:~/wm_chess/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCL[@]}" "$REPO_ROOT/experiments/distill-soft/" "ubuntu@$ip:~/distill-soft/"
    ssh "${SSH_OPTS[@]}" "ubuntu@$ip" 'cd ~/distill-soft && ~/.local/bin/uv sync' > /tmp/uv_sync.log 2>&1
}

# Train one dataset. Inputs: lib_path (relative to REPO_ROOT), ckpt_dir (relative).
phase_train() {
    local lib_path="$1" ckpt_dir="$2"
    local data_local="$REPO_ROOT/$lib_path/data.npz"
    if [ ! -f "$data_local" ]; then log "FATAL: $data_local missing"; return 1; fi

    local final_ckpt="$REPO_ROOT/$ckpt_dir/distilled_epoch$(printf '%03d' $((EPOCHS-1))).pt"
    if [ -f "$final_ckpt" ]; then log "SKIP train (ckpt exists): $final_ckpt"; return 0; fi

    log "=== TRAIN on $GPU_INSTANCE for $lib_path → $ckpt_dir ==="
    terraform_up "$GPU_INSTANCE" || return 1
    wait_bootstrap "$PUBLIC_IP" || { terraform_down; return 1; }
    rsync_project_up "$PUBLIC_IP"

    ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" "mkdir -p ~/distill-soft/$(dirname "$lib_path") ~/distill-soft/$ckpt_dir"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$data_local" "ubuntu@$PUBLIC_IP:~/distill-soft/$lib_path/data.npz"

    local hard_flag=""
    [ "$HARD_TARGETS" = "true" ] && hard_flag="--hard-targets"

    ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" "
        cd ~/distill-soft && ~/.local/bin/uv run python scripts/train.py \
            --data $lib_path/data.npz \
            --epochs $EPOCHS --device cuda \
            --n-blocks $N_BLOCKS --n-filters $N_FILTERS \
            --ckpt-dir $ckpt_dir \
            --save-every 5 $hard_flag 2>&1 | tee /tmp/train.log
    " || log "training failed, pulling partial ckpts"

    mkdir -p "$REPO_ROOT/$ckpt_dir"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$PUBLIC_IP:~/distill-soft/$ckpt_dir/" "$REPO_ROOT/$ckpt_dir/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$PUBLIC_IP:/tmp/train.log" "$REPO_ROOT/$ckpt_dir/train.log" || true
    terraform_down
    log "=== TRAIN done — $final_ckpt ==="
}

phase_eval() {
    local ckpt_dir="$1"
    local final_ckpt="$REPO_ROOT/$ckpt_dir/distilled_epoch$(printf '%03d' $((EPOCHS-1))).pt"
    if [ ! -f "$final_ckpt" ]; then log "FATAL: missing $final_ckpt"; return 1; fi

    log "=== EVAL on $CPU_INSTANCE for $ckpt_dir ==="
    terraform_up "$CPU_INSTANCE" || return 1
    wait_bootstrap "$PUBLIC_IP" || { terraform_down; return 1; }
    rsync_project_up "$PUBLIC_IP"

    ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" "mkdir -p ~/distill-soft/$ckpt_dir"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$final_ckpt" "ubuntu@$PUBLIC_IP:~/distill-soft/$ckpt_dir/"

    local n_workers
    n_workers="$(ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" 'nproc')"
    n_workers=$((n_workers - 2)); [ $n_workers -lt 1 ] && n_workers=1
    local gpw=$(( (EVAL_GAMES_TOTAL + n_workers - 1) / n_workers ))

    for sims in "$SIMS_BASELINE" "$SIMS_STRETCH"; do
        log "Eval @ $sims sims"
        ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" "
            cd ~/distill-soft && ~/.local/bin/uv run python scripts/eval.py \
                --ckpt $ckpt_dir/distilled_epoch$(printf '%03d' $((EPOCHS-1))).pt \
                --workers $n_workers --games-per-worker $gpw \
                --sims $sims \
                --n-blocks $N_BLOCKS --n-filters $N_FILTERS \
                --agent-device cpu \
                --stockfish-elo $SF_ELO 2>&1 | tee /tmp/eval_${sims}.log
        " || log "Eval @ $sims FAILED — continuing"
        rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$PUBLIC_IP:/tmp/eval_${sims}.log" \
            "$REPO_ROOT/$ckpt_dir/eval_${sims}sims.log" || true
    done

    terraform_down
    log "=== EVAL done — logs in $ckpt_dir/ ==="
}

# Derive ckpt_dir from lib_path. Encodes target shape so soft vs hard runs don't collide.
ckpt_dir_for() {
    local lib_path="$1"
    # lib_path: library/games/sf-<v>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/
    local rel="${lib_path#library/games/}"
    local target_shape="soft"
    [ "$HARD_TARGETS" = "true" ] && target_shape="hard"
    local run_id="${target_shape}-$(date -u +%Y%m%dT%H%MZ)"
    echo "checkpoints/$rel/net-${N_BLOCKS}x${N_FILTERS}/$run_id"
}

process_one() {
    local lib_path="$1"
    local ckpt_dir
    ckpt_dir="$(ckpt_dir_for "$lib_path")"
    phase_train "$lib_path" "$ckpt_dir" || { log "abort: train failed for $lib_path"; return 1; }
    phase_eval  "$ckpt_dir"              || { log "abort: eval failed for $ckpt_dir";  return 1; }
}

# --- driver ---
if [ -n "$ALL" ]; then
    log "ALL mode: walking library/games/ for complete datasets"
    found=0
    while IFS= read -r leaf; do
        rel="${leaf#$REPO_ROOT/}"
        process_one "$rel" && found=$((found+1))
    done < <(find "$LIB_DIR/games" -type f -name data.npz 2>/dev/null | sed 's|/data.npz||')
    log "ALL: processed $found datasets"
else
    : "${LIB_PATH:?Set LIB_PATH=library/games/sf-<v>/... or use ALL=1}"
    process_one "$LIB_PATH"
fi

log "DONE. Verify nothing left billing: aws ec2 describe-instances --region $REGION --filters Name=instance-state-name,Values=running"
