#!/usr/bin/env bash
# Orchestrator for AlphaZero-style self-play RL on EKS.
#
# Usage:
#   ./run-selfplay.sh up             # eksctl create cluster (~15 min)
#   ./run-selfplay.sh submit         # apply the self-play Job
#   ./run-selfplay.sh logs           # tail the pod logs
#   ./run-selfplay.sh status         # Job + pod state + S3 ckpts
#   ./run-selfplay.sh down           # eksctl delete cluster
#
# Loads config from .env at repo root. Required env (set via .env or
# explicit caller):
#   S3_BUCKET_US, ECR_URI_US (for the GPU image)
# And for `submit`:
#   INIT_FROM_S3   — the distilled prior to start from
#                    (e.g. s3://wm-chess-library-.../distilled_epoch019.pt)
#   S3_PREFIX      — under which $BUCKET prefix the selfplay output lives
#                    (defaults to the same prefix as the prior)

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
    while IFS='=' read -r key val; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [ -z "$key" ] && continue
        [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
        [ -n "${!key+x}" ] && continue
        export "$key=$val"
    done < "$REPO_ROOT/.env"
fi

CMD="${1:?usage: $0 up|image|submit|logs|status|down}"

: "${REGION:=${AWS_PRIMARY_REGION:-us-east-1}}"
: "${BUCKET:=${S3_BUCKET_US:-}}"
: "${ECR_URI:=${ECR_URI_US:+${ECR_URI_US}-gpu}}"
[ -n "$BUCKET" ]  || { echo "BUCKET unset (set via .env)"; exit 2; }
[ -n "$ECR_URI" ] || { echo "ECR_URI unset"; exit 2; }

CLUSTER_SPEC="$THIS_DIR/cluster-selfplay-us.yaml"
KUBECTL_CTX="wm-selfplay-us"

log() { echo "[$(date -u +%H:%M:%S) selfplay] $*"; }

case "$CMD" in
    up)
        log "eksctl create cluster (~15 min)"
        eksctl create cluster -f "$CLUSTER_SPEC"
        aws eks update-kubeconfig --region "$REGION" \
            --name wm-chess-selfplay --alias "$KUBECTL_CTX"
        ;;
    image)
        log "Same image as run-train.sh ($ECR_URI:latest). Use:"
        log "    ./run-train.sh image us"
        log "(Dockerfile.train now bundles both entrypoints.)"
        ;;
    submit)
        : "${INIT_FROM_S3:?required (e.g. s3://wm-chess-library-.../distilled_epoch019.pt)}"
        # Derive default S3_PREFIX from the prior's path if not set.
        if [ -z "${S3_PREFIX:-}" ]; then
            # s3://BUCKET/PREFIX/checkpoints/.../distilled_epochNNN.pt -> PREFIX
            S3_PREFIX=$(echo "$INIT_FROM_S3" | sed -E 's|^s3://[^/]+/([^/]+)/.*|\1|')
            log "S3_PREFIX derived from INIT_FROM_S3: $S3_PREFIX"
        fi

        # Defaults — 20x256 net, 800 sims, 12h time budget.
        : "${N_BLOCKS:=20}"
        : "${N_FILTERS:=256}"
        : "${SIMS:=800}"
        : "${WORKERS:=24}"
        : "${GAMES_PER_WORKER:=2}"
        : "${TRAIN_STEPS:=200}"
        : "${LR:=1e-3}"
        : "${TIME_BUDGET:=43200}"   # 12h
        : "${PCR:=1}"
        : "${N_HISTORY:=1}"
        : "${NODEGROUP_LABEL:=gpu-selfplay}"
        : "${RUN_ID:=$(date -u +%Y%m%dT%H%MZ)}"

        export S3_BUCKET="$BUCKET" S3_PREFIX="$S3_PREFIX" \
               AWS_DEFAULT_REGION="$REGION" ECR_URI_TRAIN="$ECR_URI" \
               JOB_PREFIX_LABEL="$(echo "$S3_PREFIX" | tr '[:upper:].' '[:lower:]-' | cut -c1-63)" \
               INIT_FROM_S3 RUN_ID N_BLOCKS N_FILTERS SIMS WORKERS \
               GAMES_PER_WORKER TRAIN_STEPS LR TIME_BUDGET PCR N_HISTORY \
               NODEGROUP_LABEL

        kubectl --context "$KUBECTL_CTX" delete job wm-chess-selfplay --ignore-not-found --wait=true
        envsubst < "$THIS_DIR/k8s/job-selfplay.yaml" \
            | kubectl --context "$KUBECTL_CTX" apply -f -
        log "submitted job=wm-chess-selfplay run_id=$RUN_ID"
        log "tail logs with: $0 logs"
        ;;
    logs)
        POD=$(kubectl --context "$KUBECTL_CTX" get pods -l app=wm-chess-selfplay -o jsonpath='{.items[0].metadata.name}')
        kubectl --context "$KUBECTL_CTX" logs -f "$POD"
        ;;
    status)
        kubectl --context "$KUBECTL_CTX" get jobs,pods -l app=wm-chess-selfplay
        log "iter checkpoints in s3://$BUCKET/${S3_PREFIX:-?}/selfplay/"
        if [ -n "${S3_PREFIX:-}" ]; then
            aws s3 ls "s3://$BUCKET/$S3_PREFIX/selfplay/" --recursive --region "$REGION" 2>/dev/null \
                | tail -20 || echo "  (none yet)"
        fi
        ;;
    down)
        log "eksctl delete cluster (frees \$0.10/h control plane + g6.8xlarge)"
        eksctl delete cluster -f "$CLUSTER_SPEC"
        ;;
    *)
        echo "unknown command: $CMD"; exit 2 ;;
esac
