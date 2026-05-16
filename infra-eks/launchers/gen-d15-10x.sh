#!/usr/bin/env bash
# Launch the 10x d15 datagen pipeline.
#
# Targets: 500,000 games at Stockfish depth 15, multipv=8, T=1 pawn.
# Approximate scale of the resulting dataset:
#   - Existing d15: 49,525 games / 5.07 M positions
#   - 10x target : ~500k games / ~50 M positions
#
# Wallclock and cost depend on actual spot capacity AWS allocates:
#   - At the current 256-vCPU spot quota (8 pods): ~4 days, ~$300
#   - At a raised 2,560-vCPU quota (80 pods)     : ~10-12 hours, ~$800
#   - Multi-region (us + eu-west)                : ~2 days, ~$300-400
#
# This launcher provisions us-east-1 only. The Job submits with
# parallelism=80; AWS spot will fulfill whatever capacity it can, and
# the rest stay Pending. If you get a quota increase later, scale the
# nodegroup up:
#   eksctl scale nodegroup --cluster wm-chess-gen-10x \
#       --name workers-spot-10x --nodes <N>
#
# Usage:
#   bash infra-eks/launchers/gen-d15-10x.sh up        # eksctl create cluster
#   bash infra-eks/launchers/gen-d15-10x.sh submit    # build image + apply Job
#   bash infra-eks/launchers/gen-d15-10x.sh status    # Job + pod state
#   bash infra-eks/launchers/gen-d15-10x.sh logs      # tail pod 0
#   bash infra-eks/launchers/gen-d15-10x.sh merge     # apply merge Job
#   bash infra-eks/launchers/gen-d15-10x.sh down      # eksctl delete cluster

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"

CMD="${1:?usage: $0 up|submit|status|logs|merge|down}"

REGION=us-east-1
CLUSTER=wm-chess-gen-10x
KUBECTL_CTX=wm-gen-10x

# Same image as the original datagen — Dockerfile (CPU + stockfish).
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=wm-chess
ECR_URI=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO

S3_BUCKET=wm-chess-library-$ACCOUNT_ID
RUN_TS=${RUN_TS:-$(date -u +%Y%m%dT%H%MZ)}
S3_PREFIX=${S3_PREFIX:-d15-mpv8-T1-g500000-${RUN_TS}}

log() { echo "[$(date -u +%H:%M:%S) gen-10x] $*"; }

case "$CMD" in
    up)
        log "eksctl create cluster wm-chess-gen-10x (~15-20 min)"
        eksctl create cluster -f "$REPO_ROOT/infra-eks/cluster-gen-10x.yaml"
        aws eks update-kubeconfig --region "$REGION" \
            --name "$CLUSTER" --alias "$KUBECTL_CTX"
        log "submit a Service Quotas increase for 'Running On-Demand Standard "
        log "(A,C,D,H,I,M,R,T,Z) instances' AND 'All Standard (A, C, D, H, I, "
        log "M, R, T, Z) Spot Instance Requests' to ~2560 vCPU for true 10x speed."
        log "Without the bump, the Job runs at ~8 pods (the rest stay Pending)."
        ;;
    image)
        log "triggering CodeBuild → $ECR_REPO:latest (no local Docker)"
        bash "$REPO_ROOT/infra-eks/scripts/rebuild-image.sh" "$ECR_REPO"
        ;;
    submit)
        log "rendering job-gen-10x.yaml with ECR=$ECR_URI prefix=$S3_PREFIX"
        export ECR_URI S3_BUCKET S3_PREFIX
        envsubst < "$REPO_ROOT/infra-eks/k8s/job-gen-10x.yaml" \
            | kubectl --context "$KUBECTL_CTX" apply -f -
        log "Job submitted. Output will land in s3://$S3_BUCKET/$S3_PREFIX/"
        log "Tail with: $0 logs"
        ;;
    status)
        kubectl --context "$KUBECTL_CTX" get jobs,pods -l app=wm-chess
        echo
        log "shards in S3 so far:"
        aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/shards_partial/" --region "$REGION" 2>/dev/null \
            | head -20 || echo "  (none yet)"
        ;;
    logs)
        POD=$(kubectl --context "$KUBECTL_CTX" get pods -l job-name=wm-chess-gen-10x \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        [ -n "$POD" ] || { echo "no pod found yet"; exit 1; }
        kubectl --context "$KUBECTL_CTX" logs -f "$POD"
        ;;
    merge)
        log "applying merge job — concatenates all shards into one data.npz"
        export ECR_URI S3_BUCKET S3_PREFIX
        envsubst < "$REPO_ROOT/infra-eks/k8s/job-merge.yaml" \
            | kubectl --context "$KUBECTL_CTX" apply -f -
        ;;
    down)
        log "eksctl delete cluster wm-chess-gen-10x"
        eksctl delete cluster -f "$REPO_ROOT/infra-eks/cluster-gen-10x.yaml"
        ;;
    *)
        echo "unknown command: $CMD"; exit 2 ;;
esac
