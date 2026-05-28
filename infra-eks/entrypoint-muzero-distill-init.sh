#!/usr/bin/env bash
# Container entrypoint for distill-init MuZero (train only the dynamics g on
# top of a frozen distilled teacher). Same trampoline pattern as
# entrypoint-muzero-chess.sh — pulls muzero-chess source from main at start
# (the wm-chess-gpu image only bakes wm_chess + distill-soft + selfplay).
#
# Env:
#   S3_BUCKET, S3_PREFIX, AWS_DEFAULT_REGION   — required
#   TEACHER_CKPT_S3                            — required (the distilled net)
#   RUN_ID, SIMS, N_GAMES, ROUNDS, TRAIN_STEPS_PER_ROUND, EVAL_GAMES,
#   EVAL_ELOS, BATCH_SIZE, LR, ...             — see run_distill_init.py

set -euo pipefail

: "${S3_BUCKET:?required}"
: "${S3_PREFIX:?required}"
: "${AWS_DEFAULT_REGION:?required}"
: "${TEACHER_CKPT_S3:?required (s3:// path to the distilled teacher checkpoint)}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)-distill-init}"
S3_BASE="s3://$S3_BUCKET/$S3_PREFIX/$RUN_ID"
WORK_REPO=/tmp/world-models
CKPT_DIR="${CKPT_DIR:-/work/checkpoints_distill_init}"
mkdir -p "$CKPT_DIR"

echo "=== $(date -u) distill-init entrypoint ==="
echo "RUN_ID=$RUN_ID  S3_BASE=$S3_BASE  TEACHER=$TEACHER_CKPT_S3"
nvidia-smi || echo "(no nvidia-smi)"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# curl+tar (no git in the pytorch runtime base image; see entrypoint-muzero-chess.sh).
if [ ! -d "$WORK_REPO" ]; then
    echo "[fetch] world-models main → $WORK_REPO"
    curl -fsSL https://github.com/shehio/world-models/archive/refs/heads/main.tar.gz -o /tmp/wm.tar.gz
    tar -xzf /tmp/wm.tar.gz -C /tmp/
    mv /tmp/world-models-main "$WORK_REPO"
    rm -f /tmp/wm.tar.gz
fi

export PYTHONPATH="$WORK_REPO/experiments/muzero-chess/src:${PYTHONPATH:-}"

exec python "$WORK_REPO/experiments/muzero-chess/scripts/run_distill_init.py" \
    --ckpt-dir "$CKPT_DIR" \
    --s3-base "$S3_BASE"
