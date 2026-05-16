#!/usr/bin/env bash
# Trigger the wm-chess-image CodeBuild project and wait for SUCCEEDED.
#
# CodeBuild's source is `shehio/world-models` on GitHub — it does its
# own checkout. So if you want the image to include local changes,
# push them to `main` FIRST, then run this.
#
# Usage:
#   bash infra-eks/scripts/rebuild-image.sh [ECR_REPO]
#       wm-chess      → CPU/datagen image (Dockerfile, buildspec.yml).
#       wm-chess-gpu  → GPU eval/training image (Dockerfile.train,
#                       buildspec-train.yml). Used by all the eval-*
#                       and d10/d15 training launchers.
#       (default: wm-chess)
#
# Why this exists: launchers used to call `docker buildx build --push`
# from the laptop, which (a) needs a running local Docker daemon, (b)
# couples image content to laptop state instead of git, and (c) lets
# source changes ship without rebuilding the image. The self-play
# eval-final regression (eval.py patched in source but image stale) is
# exactly that bug class — and it caught us once.

set -euo pipefail

REPO="${1:-wm-chess}"
REGION="${AWS_REGION:-us-east-1}"
PROJECT=wm-chess-image

# The CodeBuild project's default buildspec builds the CPU image. For the
# GPU image we override to buildspec-train.yml (different Dockerfile,
# different base image — pytorch CUDA — different install path).
case "$REPO" in
    wm-chess)        BUILDSPEC=infra-eks/buildspec.yml ;;
    wm-chess-gpu)    BUILDSPEC=infra-eks/buildspec-train.yml ;;
    *) echo "unknown repo: $REPO (expected wm-chess or wm-chess-gpu)"; exit 2 ;;
esac

log() { echo "[$(date -u +%H:%M:%S) rebuild-image] $*"; }

log "starting CodeBuild $PROJECT  ECR_REPO=$REPO  buildspec=$BUILDSPEC  region=$REGION"
BUILD_ID=$(aws codebuild start-build \
    --project-name "$PROJECT" --region "$REGION" \
    --buildspec-override "$BUILDSPEC" \
    --environment-variables-override "name=ECR_REPO,value=$REPO,type=PLAINTEXT" \
    --query 'build.id' --output text)
log "build id: $BUILD_ID"

while true; do
    STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
             --query 'builds[0].buildStatus' --output text)
    case "$STATUS" in
        SUCCEEDED) log "image built + pushed: $REPO:latest"; exit 0 ;;
        FAILED|FAULT|STOPPED|TIMED_OUT)
            log "build $STATUS — inspect:"
            log "  aws codebuild batch-get-builds --ids $BUILD_ID --region $REGION"
            exit 1 ;;
        *) sleep 20 ;;
    esac
done
