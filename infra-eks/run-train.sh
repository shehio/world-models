#!/usr/bin/env bash
# Orchestrator for GPU training on EKS, per region.
#
# Usage:
#   ./run-train.sh up <us|eu>           # eksctl create cluster
#   ./run-train.sh image <us|eu>        # docker build + push the GPU image
#   ./run-train.sh submit <us|eu>       # apply the training Job to the cluster
#   ./run-train.sh logs <us|eu>         # tail the Job's pod logs
#   ./run-train.sh status <us|eu>       # Job + pod status + checkpoint listing
#   ./run-train.sh down <us|eu>         # eksctl delete cluster
#
# Loads config from .env at repo root.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"

# shellcheck disable=SC1091
[ -f "$REPO_ROOT/.env" ] && source "$REPO_ROOT/.env"

CMD="${1:?usage: $0 up|image|submit|logs|status|down  us|eu}"
REGION_ARG="${2:?usage: $0 $CMD  us|eu}"

case "$REGION_ARG" in
    us) REGION="${AWS_PRIMARY_REGION:-us-east-1}"
        BUCKET="${S3_BUCKET_US:?missing in .env}"
        PREFIX="${RUN_PREFIX_US:?missing in .env}"
        ECR_URI="${ECR_URI_US:?missing in .env}-gpu" ;;
    eu) REGION="${AWS_SECONDARY_REGION:-eu-west-1}"
        BUCKET="${S3_BUCKET_EU:?missing in .env}"
        PREFIX="${RUN_PREFIX_EU:?missing in .env}"
        ECR_URI="${ECR_URI_EU:?missing in .env}-gpu" ;;
    *)  echo "unknown region: $REGION_ARG (use us or eu)"; exit 2 ;;
esac

CLUSTER_SPEC="$THIS_DIR/cluster-train-${REGION_ARG}.yaml"
# Per-region kubeconfig context (created by `aws eks update-kubeconfig
# ... --alias`). Used as `kubectl --context "$KUBECTL_CTX"` EVERYWHERE
# below so we never depend on the operator's current-context.
KUBECTL_CTX="wm-train-${REGION_ARG}"

log() { echo "[$(date -u +%H:%M:%S) $REGION_ARG] $*"; }

case "$CMD" in
    up)
        log "eksctl create cluster (~15 min)"
        eksctl create cluster -f "$CLUSTER_SPEC"
        ;;
    image)
        log "ECR login + buildx push to $ECR_URI:latest"
        # Ensure the repo exists (gen/merge image used the non-suffixed repo)
        local_repo="$(echo "$ECR_URI" | awk -F/ '{print $NF}')"
        aws ecr describe-repositories --region "$REGION" --repository-names "$local_repo" >/dev/null 2>&1 \
            || aws ecr create-repository --region "$REGION" --repository-name "$local_repo"
        aws ecr get-login-password --region "$REGION" \
            | docker login --username AWS --password-stdin "${ECR_URI%/*}"
        docker buildx build \
            --platform linux/amd64 \
            --file "$THIS_DIR/Dockerfile.train" \
            --tag "$ECR_URI:latest" \
            --push \
            "$REPO_ROOT"
        ;;
    submit)
        log "rendering job-train.yaml and kubectl apply"
        # envsubst doesn't support ${VAR:-default}, so set every var
        # the YAML references with shell ${VAR:=default} before export.
        : "${EPOCHS:=20}"
        : "${BATCH_SIZE:=256}"
        : "${LR:=1e-3}"
        : "${N_BLOCKS:=20}"
        : "${N_FILTERS:=256}"
        : "${SAVE_EVERY:=5}"
        : "${MAX_POSITIONS:=}"
        : "${IN_RAM:=}"
        # Target nodegroup label (default: original `gpu-trainer`. Override
        # to `gpu-fast` when targeting the g6.4xlarge in-RAM-loading group).
        : "${NODEGROUP_LABEL:=gpu-trainer}"
        # Job name suffix lets you run two trainings concurrently on
        # one cluster (e.g. d15 on gpu-trainer, d10 on gpu-fast).
        : "${JOB_NAME:=wm-chess-train}"
        export S3_BUCKET="$BUCKET" S3_PREFIX="$PREFIX" \
               AWS_DEFAULT_REGION="$REGION" ECR_URI_TRAIN="$ECR_URI" \
               JOB_PREFIX_LABEL="$(echo "$PREFIX" | tr '[:upper:].' '[:lower:]-' | cut -c1-63)" \
               EPOCHS BATCH_SIZE LR N_BLOCKS N_FILTERS SAVE_EVERY \
               MAX_POSITIONS IN_RAM NODEGROUP_LABEL JOB_NAME
        # Always delete-then-apply: Job spec.template is immutable, so
        # if a prior Job with the same name exists, apply errors out.
        kubectl --context "$KUBECTL_CTX" delete job "$JOB_NAME" --ignore-not-found --wait=true
        # Render with envsubst, then sed in the Job name (kept simple
        # so the same YAML template serves two concurrent runs).
        envsubst < "$THIS_DIR/k8s/job-train.yaml" \
            | sed "s|name: wm-chess-train$|name: $JOB_NAME|" \
            | kubectl --context "$KUBECTL_CTX" apply -f -
        log "submitted job=$JOB_NAME nodegroup=$NODEGROUP_LABEL; tail logs with: $0 logs $REGION_ARG"
        ;;
    logs)
        log "tailing pod logs (Ctrl-C to stop)"
        POD=$(kubectl --context "$KUBECTL_CTX" get pods -l app=wm-chess-train -o jsonpath='{.items[0].metadata.name}')
        kubectl --context "$KUBECTL_CTX" logs -f "$POD"
        ;;
    status)
        log "Job + pod status"
        kubectl --context "$KUBECTL_CTX" get jobs,pods -l app=wm-chess-train
        log "checkpoints in s3://$BUCKET/$PREFIX/checkpoints/"
        aws s3 ls "s3://$BUCKET/$PREFIX/checkpoints/" --region "$REGION" 2>/dev/null \
            || echo "  (none yet)"
        ;;
    down)
        log "eksctl delete cluster (frees control plane $ + GPU nodes)"
        eksctl delete cluster -f "$CLUSTER_SPEC"
        ;;
    *)
        echo "unknown command: $CMD"; exit 2 ;;
esac
