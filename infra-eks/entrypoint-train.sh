#!/usr/bin/env bash
# Training entrypoint:
#   1. Fetch s3://$S3_BUCKET/$S3_PREFIX/merged/data.npz
#   2. Run train.py with --device cuda
#   3. Sync local checkpoints/ to s3://$S3_BUCKET/$S3_PREFIX/checkpoints/
#
# All training knobs are env-var overridable. Defaults are paper-sized
# (20×256 ResNet, 20 epochs, batch 256, lr 1e-3).
#
# Resume: if any distilled_epoch*.pt exists under $S3_PREFIX/checkpoints/,
# this script downloads it and passes --init-from. Lets spot reclaims
# restart from the last checkpoint instead of from scratch.

set -euo pipefail

: "${S3_BUCKET:?required (e.g. wm-chess-library-594561963943)}"
: "${S3_PREFIX:?required (e.g. d15-mpv8-T1-g100000-20260512T1844Z)}"
: "${AWS_DEFAULT_REGION:?required}"

: "${EPOCHS:=20}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${WEIGHT_DECAY:=1e-4}"
: "${N_BLOCKS:=20}"
: "${N_FILTERS:=256}"
: "${VALUE_WEIGHT:=1.0}"
: "${SAVE_EVERY:=5}"
: "${HARD_TARGETS:=}"   # set to anything to enable --hard-targets

WORK=/work
DATA_DIR=$WORK/data
CKPT_DIR=$WORK/checkpoints
mkdir -p "$DATA_DIR" "$CKPT_DIR"

echo "=== $(date -u) ${S3_PREFIX} training ==="
echo "REGION=$AWS_DEFAULT_REGION BUCKET=$S3_BUCKET"
echo "EPOCHS=$EPOCHS BATCH=$BATCH_SIZE NET=${N_BLOCKS}x${N_FILTERS} LR=$LR"
nvidia-smi || echo "(no nvidia-smi)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

echo "=== fetching merged dataset ==="
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/merged/data.npz" "$DATA_DIR/data.npz" --no-progress
ls -la "$DATA_DIR/"

echo "=== checking for prior checkpoint to resume from ==="
INIT_ARG=()
LATEST_CKPT=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/" 2>/dev/null \
    | awk '{print $NF}' | grep -E '^distilled_epoch[0-9]+\.pt$' | sort -V | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "  resuming from $LATEST_CKPT"
    aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/$LATEST_CKPT" "$CKPT_DIR/$LATEST_CKPT" --no-progress
    INIT_ARG=(--init-from "$CKPT_DIR/$LATEST_CKPT")
else
    echo "  no prior checkpoint, training from scratch"
fi

HARD_ARG=()
if [ -n "$HARD_TARGETS" ]; then
    HARD_ARG=(--hard-targets)
fi

echo "=== training ==="
cd /work/experiments/distill-soft
python scripts/train.py \
    --data "$DATA_DIR/data.npz" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --n-blocks "$N_BLOCKS" \
    --n-filters "$N_FILTERS" \
    --value-weight "$VALUE_WEIGHT" \
    --device cuda \
    --ckpt-dir "$CKPT_DIR" \
    --save-every "$SAVE_EVERY" \
    "${INIT_ARG[@]}" \
    "${HARD_ARG[@]}"

echo "=== uploading checkpoints + history ==="
aws s3 sync "$CKPT_DIR/" "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/" --no-progress

echo "=== $(date -u) done ==="
