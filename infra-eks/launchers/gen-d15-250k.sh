#!/usr/bin/env bash
# Launch the d15 250K datagen pipeline — same scale as the d10 30M run
# that produced our 2,171-Elo headline, but with the d15 teacher
# (~2500 Elo) instead of d10 (~2200 Elo).
#
# Target: 250,000 games at Stockfish depth 15, multipv=8, T=1 pawn.
#         8 pods × 31,250 games each.
# Wallclock: ~24-36 h on 8 c-class spot pods (256 vCPU).
# Cost: ~$75-100 (8 pods × ~$0.30-0.40 per pod-hour × ~30h + EKS control plane).
#
# Recipe lineage:
#   - cluster spec from cluster-gen-10x.yaml (the resilient diversified-spot one)
#   - job spec backoffLimit=240 from job-gen-10x.yaml (the only d15-shape
#     gen job that ever survived reclaim churn; add5m at backoffLimit=24
#     died at 22%)
#   - BASE_SEED=30042 disjoint from previous d15 runs so games don't repeat
#     when datasets are merged
#
# Usage:
#   bash infra-eks/launchers/gen-d15-250k.sh up      # eksctl create cluster (~15-20 min)
#   bash infra-eks/launchers/gen-d15-250k.sh image   # CodeBuild → wm-chess:latest
#   bash infra-eks/launchers/gen-d15-250k.sh submit  # apply the gen Job
#   bash infra-eks/launchers/gen-d15-250k.sh status  # Job + pod state
#   bash infra-eks/launchers/gen-d15-250k.sh logs    # tail pod 0
#   bash infra-eks/launchers/gen-d15-250k.sh merge   # apply merge Job once gen completes
#   bash infra-eks/launchers/gen-d15-250k.sh down    # eksctl delete cluster

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"

CMD="${1:?usage: $0 up|image|submit|status|logs|merge|down}"

REGION=us-east-1
CLUSTER=wm-chess-gen-d15-250k
KUBECTL_CTX=wm-gen-d15-250k

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=wm-chess
ECR_URI=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO

S3_BUCKET=wm-chess-library-$ACCOUNT_ID
RUN_TS=${RUN_TS:-$(date -u +%Y%m%dT%H%MZ)}
S3_PREFIX=${S3_PREFIX:-d15-mpv8-T1-g250000-${RUN_TS}}

log() { echo "[$(date -u +%H:%M:%S) gen-d15-250k] $*"; }

case "$CMD" in
    up)
        log "eksctl create cluster $CLUSTER (~15-20 min)"
        eksctl create cluster -f "$REPO_ROOT/infra-eks/cluster-gen-d15-250k.yaml"
        aws eks update-kubeconfig --region "$REGION" \
            --name "$CLUSTER" --alias "$KUBECTL_CTX"
        log "cluster up. next: $0 image"
        ;;
    image)
        log "triggering CodeBuild → $ECR_REPO:latest"
        bash "$REPO_ROOT/infra-eks/scripts/rebuild-image.sh" "$ECR_REPO"
        ;;
    submit)
        log "rendering job-gen-d15-250k.yaml with ECR=$ECR_URI prefix=$S3_PREFIX"
        export ECR_URI S3_BUCKET S3_PREFIX
        envsubst < "$REPO_ROOT/infra-eks/k8s/job-gen-d15-250k.yaml" \
            | kubectl --context "$KUBECTL_CTX" apply -f -
        log "Job submitted. Output will land in s3://$S3_BUCKET/$S3_PREFIX/"
        log "Tail with: $0 logs"
        log "Resume info — when pods get reclaimed they restart on the same"
        log "index, read existing chunks from shards_partial/, and pick up"
        log "where they left off. The full game count is durable."
        ;;
    status)
        kubectl --context "$KUBECTL_CTX" get jobs,pods -l app=wm-chess
        echo
        log "shards in S3 so far:"
        aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/shards_partial/" --region "$REGION" 2>/dev/null \
            | head -20 || echo "  (none yet)"
        ;;
    logs)
        POD=$(kubectl --context "$KUBECTL_CTX" get pods -l job-name=wm-chess-gen-d15-250k \
              -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        [ -n "$POD" ] || { echo "no pod found yet"; exit 1; }
        kubectl --context "$KUBECTL_CTX" logs -f "$POD"
        ;;
    merge)
        log "applying merge job — concatenates the 8 shards into one data.npz"
        export ECR_URI S3_BUCKET S3_PREFIX
        envsubst < "$REPO_ROOT/infra-eks/k8s/job-merge.yaml" \
            | kubectl --context "$KUBECTL_CTX" apply -f -
        ;;
    down)
        log "eksctl delete cluster $CLUSTER"
        eksctl delete cluster -f "$REPO_ROOT/infra-eks/cluster-gen-d15-250k.yaml"
        ;;
    *)
        echo "unknown command: $CMD"; exit 2 ;;
esac
