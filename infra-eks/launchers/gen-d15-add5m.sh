#!/usr/bin/env bash
# Generate ~5M more d15 positions (50K games) and stage them for merging
# against the existing 5M d15 baseline.
#
# Target:    50,000 games at Stockfish depth 15, multipv=8, T=1 pawn.
# Wallclock: ~24 h at the current 256-vCPU spot quota (8 pods × 32 vCPU).
# Cost:      ~$35 (c-class spot @ $0.30–0.40 per pod-hour × 8 × 24).
#
# Reuses the cluster-gen-10x.yaml spec (desiredCapacity=8 already matches
# the spot quota). Image is built via CodeBuild — no local Docker needed.
#
# Usage:
#   bash infra-eks/launchers/gen-d15-add5m.sh up        # eksctl create cluster
#   bash infra-eks/launchers/gen-d15-add5m.sh image     # CodeBuild → wm-chess:latest
#   bash infra-eks/launchers/gen-d15-add5m.sh submit    # apply the gen Job
#   bash infra-eks/launchers/gen-d15-add5m.sh status    # Job + pod state
#   bash infra-eks/launchers/gen-d15-add5m.sh logs      # tail pod 0
#   bash infra-eks/launchers/gen-d15-add5m.sh merge     # apply merge Job (this run only)
#   bash infra-eks/launchers/gen-d15-add5m.sh down      # eksctl delete cluster

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"

CMD="${1:?usage: $0 up|image|submit|status|logs|merge|down}"

REGION=us-east-1
CLUSTER=wm-chess-gen-10x        # reuse the 10x cluster spec (desiredCapacity=8)
KUBECTL_CTX=wm-gen-10x

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=wm-chess
ECR_URI=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO

S3_BUCKET=wm-chess-library-$ACCOUNT_ID
RUN_TS=${RUN_TS:-$(date -u +%Y%m%dT%H%MZ)}
S3_PREFIX=${S3_PREFIX:-d15-mpv8-T1-g50000-add5m-${RUN_TS}}

log() { echo "[$(date -u +%H:%M:%S) gen-add5m] $*"; }

case "$CMD" in
    up)
        log "eksctl create cluster $CLUSTER (~15-20 min)"
        eksctl create cluster -f "$REPO_ROOT/infra-eks/cluster-gen-10x.yaml"
        aws eks update-kubeconfig --region "$REGION" \
            --name "$CLUSTER" --alias "$KUBECTL_CTX"
        ;;
    image)
        log "triggering CodeBuild → $ECR_REPO:latest (no local Docker)"
        bash "$REPO_ROOT/infra-eks/scripts/rebuild-image.sh" "$ECR_REPO"
        ;;
    submit)
        log "rendering job-gen-add5m.yaml with ECR=$ECR_URI prefix=$S3_PREFIX"
        export ECR_URI S3_BUCKET S3_PREFIX
        envsubst < "$REPO_ROOT/infra-eks/k8s/job-gen-add5m.yaml" \
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
        POD=$(kubectl --context "$KUBECTL_CTX" get pods -l job-name=wm-chess-gen-add5m \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        [ -n "$POD" ] || { echo "no pod found yet"; exit 1; }
        kubectl --context "$KUBECTL_CTX" logs -f "$POD"
        ;;
    merge)
        log "applying merge job — concatenates the 8 add5m shards into data.npz"
        export ECR_URI S3_BUCKET S3_PREFIX
        envsubst < "$REPO_ROOT/infra-eks/k8s/job-merge.yaml" \
            | kubectl --context "$KUBECTL_CTX" apply -f -
        ;;
    down)
        log "eksctl delete cluster $CLUSTER"
        eksctl delete cluster -f "$REPO_ROOT/infra-eks/cluster-gen-10x.yaml"
        ;;
    *)
        echo "unknown command: $CMD"; exit 2 ;;
esac
