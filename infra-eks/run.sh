#!/usr/bin/env bash
# End-to-end EKS launcher for the wm-chess datagen + merge pipeline.
#
# Assumes the cluster is already created (`eksctl create cluster -f cluster.yaml`).
# This script handles:
#   1. ECR repo (create if missing) + container build + push
#   2. S3 bucket (create if missing) for shard staging
#   3. IAM role + service account (so pods can read/write S3)
#   4. Render Job manifests with the right image / bucket / prefix
#   5. kubectl apply gen Job → wait for completion (auto-retries via Job
#      controller absorb spot reclamations cleanly)
#   6. kubectl apply merge Job → wait
#   7. Pull merged dataset from S3 to laptop
#   8. Optionally tear down cluster
#
# Usage:
#   bash infra-eks/run.sh
#
# Optional env (defaults shown):
#   REGION=us-east-1
#   CLUSTER=wm-chess
#   REPO=wm-chess
#   BUCKET=wm-chess-library-<account_id>     (auto, must be globally unique)
#   PREFIX=d15-mpv8-T1-g100000-<utc>
#   GAMES_PER_POD=12500   N_PODS=8   DEPTH=15   MULTIPV=8   T=1.0   BASE_SEED=42
#   TEARDOWN_AFTER=0       (set 1 to eksctl delete cluster at end)

set -uo pipefail

: "${REGION:=us-east-1}"
: "${CLUSTER:=wm-chess}"
: "${REPO:=wm-chess}"
: "${GAMES_PER_POD:=12500}"
: "${N_PODS:=8}"
: "${DEPTH:=15}"
: "${MULTIPV:=8}"
: "${T:=1.0}"
: "${BASE_SEED:=42}"
: "${TEARDOWN_AFTER:=0}"

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
: "${BUCKET:=wm-chess-library-${ACCOUNT_ID}}"
T_FMT="$(python3 -c "print(f'{float(\"$T\"):g}')")"
: "${PREFIX:=d${DEPTH}-mpv${MULTIPV}-T${T_FMT}-g$((GAMES_PER_POD * N_PODS))-$(date -u +%Y%m%dT%H%MZ)}"

IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"

log "cluster   : $CLUSTER ($REGION)"
log "image     : $IMAGE"
log "s3        : s3://$BUCKET/$PREFIX/"
log "config    : N_PODS=$N_PODS GAMES_PER_POD=$GAMES_PER_POD d$DEPTH mpv$MULTIPV T=$T"

# ---------- ECR + image ----------
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
    || aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null
log "ECR repo OK: $REPO"

aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null
log "docker logged in"

log "building image (~2 min for first build)..."
docker build --platform linux/amd64 -t "$IMAGE" -f infra-eks/Dockerfile . > /tmp/docker_build.log 2>&1 || {
    log "docker build failed; tail:"; tail -25 /tmp/docker_build.log; exit 1; }
log "pushing image..."
docker push "$IMAGE" > /tmp/docker_push.log 2>&1 || {
    log "docker push failed; tail:"; tail -10 /tmp/docker_push.log; exit 1; }
log "image pushed."

# ---------- S3 bucket ----------
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null \
    || aws s3 mb "s3://$BUCKET" --region "$REGION" >/dev/null
log "S3 bucket OK: $BUCKET"

# ---------- IAM service account for S3 access ----------
ROLE_NAME="wm-chess-runner-role"
SA_NAME="wm-chess-runner"

# Inline policy: allow read/write to the bucket.
POLICY_DOC=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::$BUCKET"},
    {"Effect": "Allow", "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject"], "Resource": "arn:aws:s3:::$BUCKET/*"}
  ]
}
EOF
)

POLICY_NAME="wm-chess-s3-rw"
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/$POLICY_NAME"
aws iam get-policy --policy-arn "$POLICY_ARN" >/dev/null 2>&1 || {
    aws iam create-policy --policy-name "$POLICY_NAME" --policy-document "$POLICY_DOC" >/dev/null
    log "created IAM policy: $POLICY_NAME"
}

# Create (or update) the IRSA-backed SA via eksctl.
eksctl create iamserviceaccount \
    --cluster "$CLUSTER" --region "$REGION" \
    --namespace default --name "$SA_NAME" \
    --attach-policy-arn "$POLICY_ARN" \
    --override-existing-serviceaccounts \
    --approve > /tmp/sa.log 2>&1 || {
        log "iamserviceaccount op failed:"; tail -10 /tmp/sa.log; exit 1; }
log "SA $SA_NAME bound to IAM role"

# ---------- render + apply Jobs ----------
render_and_apply() {
    local file="$1"
    sed \
        -e "s|PLACEHOLDER_IMAGE|$IMAGE|g" \
        -e "s|PLACEHOLDER_BUCKET|$BUCKET|g" \
        -e "s|PLACEHOLDER_PREFIX|$PREFIX|g" \
        "$file" | kubectl apply -f -
}

# Delete prior Jobs of the same names so re-runs are clean.
kubectl delete job wm-chess-gen wm-chess-merge --ignore-not-found=true >/dev/null 2>&1

render_and_apply infra-eks/k8s/job-gen.yaml
log "gen Job submitted. Watching pods (kubectl get pods -l job-name=wm-chess-gen -w in another terminal)..."

kubectl wait --for=condition=complete --timeout=24h job/wm-chess-gen || {
    log "gen Job did NOT complete in time. Check: kubectl describe job/wm-chess-gen"; exit 1; }
log "gen Job complete."

render_and_apply infra-eks/k8s/job-merge.yaml
kubectl wait --for=condition=complete --timeout=1h job/wm-chess-merge || {
    log "merge Job failed. Check: kubectl logs job/wm-chess-merge"; exit 1; }
log "merge Job complete."

# ---------- pull merged dataset to local library ----------
LOCAL_OUT="library/games/sf-18/d${DEPTH}-mpv${MULTIPV}-T${T_FMT}/g$((GAMES_PER_POD * N_PODS))-merged-${PREFIX##*-}"
mkdir -p "$LOCAL_OUT"
aws s3 sync "s3://$BUCKET/$PREFIX/merged/" "$LOCAL_OUT/" --no-progress
log "merged dataset pulled to $LOCAL_OUT"

uv run --project wm_chess python wm_chess/scripts/catalog.py --quiet || true

if [ "$TEARDOWN_AFTER" = "1" ]; then
    log "tearing down cluster (TEARDOWN_AFTER=1)..."
    eksctl delete cluster -f infra-eks/cluster.yaml
fi

log "DONE. S3 staging dir: s3://$BUCKET/$PREFIX/"
