#!/usr/bin/env bash
# Trigger the wm-chess-image CodeBuild project and wait for SUCCEEDED.
#
# CodeBuild's source is `shehio/world-models` on GitHub — it does its
# own checkout. So if you want the image to include local changes,
# push them to `main` FIRST, then run this.
#
# Usage:
#   bash infra-eks/scripts/rebuild-image.sh [ECR_REPO]
#       ECR_REPO defaults to wm-chess (CPU/datagen). Pass wm-chess-gpu
#       to push the same Dockerfile to the GPU eval/training repo.
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

log() { echo "[$(date -u +%H:%M:%S) rebuild-image] $*"; }

log "starting CodeBuild $PROJECT  ECR_REPO=$REPO  region=$REGION"
BUILD_ID=$(aws codebuild start-build \
    --project-name "$PROJECT" --region "$REGION" \
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
