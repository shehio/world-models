#!/usr/bin/env bash
# Drive N data-generation configs sequentially on AWS spot CPU instances,
# writing to the shared library/games/ tree. Designed to be left running
# overnight; idempotent — already-complete datasets are skipped.
#
# Inputs:
#   KEY_NAME=mykey                                  (required)
#   CONFIGS=wm_chess/scripts/overnight_configs.tsv  (optional; default this path)
#
# Config file format (TSV, # for comments):
#   # depth multipv  T    n_games  seed
#   10      8        1.0  4000     42
#   15      8        1.0  4000     42
#   10      8        0.3  4000     42
#
# Per config, this script does:
#   1. Resolve library path: library/games/sf-<v>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/
#   2. If data.npz already exists → skip (idempotent)
#   3. terraform apply c7i.8xlarge spot in us-east-1
#   4. rsync wm_chess + distill-soft up; uv sync on both
#   5. Run distill-soft generate_data.py with --library-root pointing at
#      library/games (resolves to the same canonical path)
#   6. rsync library/games/ back to the laptop
#   7. terraform destroy
#   8. Regenerate library/CATALOG.md
#
# A spot reclamation mid-run preserves all chunks written so far. The
# next overnight run picks up the same config and finalizes from chunks
# (workers detect existing chunks, though resume-from-chunk is a TODO —
# today re-runs duplicate work; safe but wasteful).

set -uo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
INFRA_DIR="$REPO_ROOT/infra"
LIB_DIR="$REPO_ROOT/library"

: "${KEY_NAME:?Set KEY_NAME=<your-aws-key-pair-name>}"
: "${REGION:=us-east-1}"
: "${CPU_INSTANCE:=c7i.8xlarge}"
: "${USE_SPOT:=true}"
: "${CONFIGS:=$THIS_DIR/overnight_configs.tsv}"
: "${SF_VERSION:=18}"  # for path-construction only; data-gen detects it for real

KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ConnectTimeout=10)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
run_log_dir="$LIB_DIR/runs/$(date -u +%Y%m%dT%H%MZ)"
mkdir -p "$run_log_dir"
RUNLOG="$run_log_dir/run.log"
exec > >(tee -a "$RUNLOG") 2>&1

terraform_up() {
    local my_cidr
    my_cidr="$(curl -s -4 ifconfig.me)/32"
    cd "$INFRA_DIR"
    terraform apply -auto-approve \
        -var "key_name=$KEY_NAME" -var "region=$REGION" \
        -var "instance_type=$CPU_INSTANCE" -var "use_spot=$USE_SPOT" \
        -var "allowed_ssh_cidr=$my_cidr" > /tmp/tf_apply.log 2>&1 || {
            log "terraform apply FAILED"; tail -15 /tmp/tf_apply.log; cd "$REPO_ROOT"; return 1; }
    PUBLIC_IP="$(terraform output -raw public_ip)"
    cd "$REPO_ROOT"; log "instance up: $PUBLIC_IP"
}

terraform_down() {
    cd "$INFRA_DIR"
    terraform destroy -auto-approve \
        -var "key_name=$KEY_NAME" -var "region=$REGION" \
        > /tmp/tf_destroy.log 2>&1 || {
            log "terraform destroy FAILED — CHECK MANUALLY"; tail -10 /tmp/tf_destroy.log; }
    cd "$REPO_ROOT"
}

wait_for_bootstrap() {
    local ip="$1"; log "waiting for BOOTSTRAP_COMPLETE on $ip"
    local deadline=$(( $(date +%s) + 600 ))
    while [ $(date +%s) -lt $deadline ]; do
        if ssh "${SSH_OPTS[@]}" "ubuntu@$ip" 'test -f /home/ubuntu/BOOTSTRAP_COMPLETE' 2>/dev/null; then
            log "bootstrap complete on $ip"; return 0; fi
        sleep 10
    done
    log "FATAL: bootstrap timeout on $ip"; return 1
}

rsync_project_up() {
    local ip="$1"
    log "rsync wm_chess + distill-soft → $ip"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
        --exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache' \
        "$REPO_ROOT/wm_chess/" "ubuntu@$ip:~/wm_chess/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
        --exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache' --exclude='data/' \
        "$REPO_ROOT/experiments/distill-soft/" "ubuntu@$ip:~/distill-soft/"
    ssh "${SSH_OPTS[@]}" "ubuntu@$ip" \
        'cd ~/distill-soft && ~/.local/bin/uv sync' > /tmp/uv_sync.log 2>&1
}

resolve_lib_path() {
    local depth="$1" mpv="$2" T="$3" games="$4" seed="$5"
    # Match the format that stockfish_data.library_dataset_path uses (T uses %g format).
    local T_fmt
    T_fmt="$(python3 -c "print(f'{float(\"$T\"):g}')")"
    echo "library/games/sf-${SF_VERSION}/d${depth}-mpv${mpv}-T${T_fmt}/g${games}-seed${seed}"
}

generate_one() {
    local depth="$1" mpv="$2" T="$3" games="$4" seed="$5"
    local lib_rel
    lib_rel="$(resolve_lib_path "$depth" "$mpv" "$T" "$games" "$seed")"
    local lib_abs="$REPO_ROOT/$lib_rel"

    if [ -f "$lib_abs/data.npz" ] && [ -f "$lib_abs/metadata.json" ]; then
        log "SKIP (already complete): $lib_rel"; return 0; fi

    log "GEN: $lib_rel  (depth=$depth mpv=$mpv T=$T games=$games seed=$seed)"
    terraform_up || return 1
    wait_for_bootstrap "$PUBLIC_IP" || { terraform_down; return 1; }
    rsync_project_up "$PUBLIC_IP"

    local n_workers
    n_workers="$(ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" 'nproc')"
    n_workers=$((n_workers - 2)); [ $n_workers -lt 1 ] && n_workers=1

    ssh "${SSH_OPTS[@]}" "ubuntu@$PUBLIC_IP" "
        cd ~/distill-soft && ~/.local/bin/uv run python scripts/generate_data.py \
            --n-games $games --workers $n_workers \
            --depth $depth --multipv $mpv --temperature-pawns $T --seed $seed \
            --chunk-size 50 \
            --library-root /home/ubuntu/library 2>&1 | tee /tmp/gen.log
    " || log "Phase failed; pulling chunks anyway"

    log "rsync library back"
    mkdir -p "$LIB_DIR/games"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$PUBLIC_IP:~/library/games/" "$LIB_DIR/games/" || true
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$PUBLIC_IP:/tmp/gen.log" "$run_log_dir/${lib_rel//\//_}.log" || true

    terraform_down
}

# ---------- driver ----------
log "starting overnight datagen | configs=$CONFIGS"
if [ ! -f "$CONFIGS" ]; then log "FATAL: configs file not found"; exit 1; fi

# Pre-flight: nothing should be currently provisioned.
if (cd "$INFRA_DIR" && terraform state list 2>/dev/null | grep -q aws_instance); then
    log "WARNING: pre-existing instance in tf state — destroying first"
    terraform_down
fi

while IFS=$'\t ' read -r depth mpv T games seed _rest; do
    # skip comments + blanks
    [[ "$depth" =~ ^#.*$ ]] && continue
    [ -z "${depth:-}" ] && continue
    generate_one "$depth" "$mpv" "$T" "$games" "$seed"
done < "$CONFIGS"

log "regenerating CATALOG.md"
(cd "$REPO_ROOT" && uv run --project wm_chess python wm_chess/scripts/catalog.py --quiet)

log "DONE. Logs in $run_log_dir"
log "Verify nothing left billing: aws ec2 describe-instances --region $REGION --filters Name=instance-state-name,Values=running"
