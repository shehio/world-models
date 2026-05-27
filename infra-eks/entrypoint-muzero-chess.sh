#!/usr/bin/env bash
# Container entrypoint for the MuZero chess training run.
#
# The wm-chess-gpu image (Dockerfile.train) only bakes in wm_chess,
# distill-soft, and selfplay — NOT muzero-chess (the experiment is too
# new). To avoid a CodeBuild round-trip on every change, this entrypoint
# pulls the latest world-models main at container start and runs
# scripts/run_cloud.py against the freshly-cloned source.
#
# Once muzero-chess has stabilized, fold it into Dockerfile.train so we
# don't pay the git-clone latency every cold start.
#
# Env vars (defaults match scripts/run_cloud.py):
#   S3_BUCKET         — required, e.g. wm-chess-library-594561963943
#   S3_PREFIX         — required, e.g. muzero-chess-20260527
#   AWS_DEFAULT_REGION — required
#   RUN_ID            — defaults to UTC timestamp
#   SIMS              — MCTS sims per move (default 400, per user)
#   EVAL_EVERY        — eval cadence (default 5, per user)
#   EVAL_GAMES        — games per eval (default 20)
#   EVAL_STOCKFISH_ELO — anchor (default 1320)
#   ITERATIONS, TIME_BUDGET, BATCH_SIZE, LR, TRAIN_STEPS, ...
#     — see run_cloud.py for the full set

set -euo pipefail

: "${S3_BUCKET:?required (e.g. wm-chess-library-594561963943)}"
: "${S3_PREFIX:?required (e.g. muzero-chess-20260527)}"
: "${AWS_DEFAULT_REGION:?required}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)-muzero-chess}"
S3_BASE="s3://$S3_BUCKET/$S3_PREFIX/$RUN_ID"

WORK_REPO=/tmp/world-models
CKPT_DIR="${CKPT_DIR:-/work/checkpoints_muzero}"
mkdir -p "$CKPT_DIR"

echo "=== $(date -u) muzero-chess entrypoint ==="
echo "REGION=$AWS_DEFAULT_REGION  BUCKET=$S3_BUCKET  PREFIX=$S3_PREFIX  RUN_ID=$RUN_ID"
echo "S3_BASE=$S3_BASE  CKPT_DIR=$CKPT_DIR"
nvidia-smi || echo "(no nvidia-smi available)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

# Pull the muzero-chess source. The image only has wm_chess + distill-soft +
# selfplay installed; muzero-chess and the cloud runner come from here.
if [ ! -d "$WORK_REPO" ]; then
    echo "[clone] world-models main → $WORK_REPO"
    git clone --depth 1 https://github.com/shehio/world-models.git "$WORK_REPO"
fi

# Add muzero-chess to PYTHONPATH. wm_chess is already installed (-e) in the
# image, so wm_chess.{board,arena} still resolve there.
export PYTHONPATH="$WORK_REPO/experiments/muzero-chess/src:${PYTHONPATH:-}"

# Forward to the cloud script. The runner reads most hparams from env;
# anything unset uses run_cloud.py defaults.
exec python "$WORK_REPO/experiments/muzero-chess/scripts/run_cloud.py" \
    --ckpt-dir "$CKPT_DIR" \
    --s3-base "$S3_BASE"
