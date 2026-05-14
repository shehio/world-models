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

# Indexed checkpoint path that mirrors the library/games/ tree:
#   <run-prefix>/checkpoints/net-<R>x<F>/<run-id>/
# So two trainings on the same dataset with different architectures
# (or just different runs) don't overwrite each other.
ARCH="net-${N_BLOCKS}x${N_FILTERS}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)}"
CKPT_S3_BASE="s3://$S3_BUCKET/$S3_PREFIX/checkpoints/$ARCH/$RUN_ID"
echo "checkpoint S3 base: $CKPT_S3_BASE"

echo "=== $(date -u) ${S3_PREFIX} training ==="
echo "REGION=$AWS_DEFAULT_REGION BUCKET=$S3_BUCKET"
echo "EPOCHS=$EPOCHS BATCH=$BATCH_SIZE NET=${N_BLOCKS}x${N_FILTERS} LR=$LR"
echo "RUN_ID=$RUN_ID  ARCH=$ARCH"
nvidia-smi || echo "(no nvidia-smi)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

echo "=== fetching merged dataset ==="
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/merged/data.npz" "$DATA_DIR/data.npz" --no-progress
ls -la "$DATA_DIR/"

# Pre-flight: write a run_metadata.json with the hparams so checkpoints
# under this $RUN_ID are self-describing without needing the bash env.
cat > "$CKPT_DIR/run_metadata.json" <<META
{
  "run_id": "$RUN_ID",
  "arch": "$ARCH",
  "n_blocks": $N_BLOCKS,
  "n_filters": $N_FILTERS,
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "lr": $LR,
  "weight_decay": $WEIGHT_DECAY,
  "value_weight": $VALUE_WEIGHT,
  "save_every": $SAVE_EVERY,
  "hard_targets": $([ -n "$HARD_TARGETS" ] && echo "true" || echo "false"),
  "s3_bucket": "$S3_BUCKET",
  "s3_prefix": "$S3_PREFIX",
  "started_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META
cat "$CKPT_DIR/run_metadata.json"

echo "=== checking for prior checkpoint under $CKPT_S3_BASE to resume from ==="
INIT_ARG=()
LATEST_CKPT=$(aws s3 ls "$CKPT_S3_BASE/" 2>/dev/null \
    | awk '{print $NF}' | grep -E '^distilled_epoch[0-9]+\.pt$' | sort -V | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "  resuming from $LATEST_CKPT"
    aws s3 cp "$CKPT_S3_BASE/$LATEST_CKPT" "$CKPT_DIR/$LATEST_CKPT" --no-progress
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
# Optional: subsample very large datasets (e.g. d10's 30M positions
# subsampled to 5M makes one epoch tractable on a single GPU).
MAX_POS_ARG=()
if [ -n "${MAX_POSITIONS:-}" ]; then MAX_POS_ARG=(--max-positions "$MAX_POSITIONS"); fi
# Optional: --in-ram, force the dataset into RAM instead of memmap.
# 5-10x faster batch fetch when the instance has the RAM for it.
IN_RAM_ARG=()
if [ -n "${IN_RAM:-}" ]; then IN_RAM_ARG=(--in-ram); fi
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
    --s3-ckpt-base "$CKPT_S3_BASE" \
    "${INIT_ARG[@]}" \
    "${HARD_ARG[@]}" \
    "${MAX_POS_ARG[@]}" \
    "${IN_RAM_ARG[@]}"

echo "=== syncing checkpoints + history + metadata so they're durable before eval ==="
aws s3 sync "$CKPT_DIR/" "$CKPT_S3_BASE/" --no-progress --exclude "*.tmp.*" --exclude ".eval_progress_*"

# ---- EVAL ---------------------------------------------------------------
# Run eval.py vs Stockfish on the LAST checkpoint. Output goes to a log
# file alongside the checkpoints, then re-sync to S3.
if [ "${RUN_EVAL:-1}" != "0" ]; then
    LATEST_CKPT=$(ls -1 "$CKPT_DIR"/distilled_epoch*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "=== no checkpoint produced; skipping eval ==="
    else
        : "${EVAL_WORKERS:=4}"
        : "${EVAL_GAMES_PER_WORKER:=25}"
        : "${EVAL_SIMS:=800}"
        : "${STOCKFISH_ELO:=1320}"
        EVAL_LOG="$CKPT_DIR/eval_results.txt"
        echo "=== eval vs Stockfish elo=$STOCKFISH_ELO on $LATEST_CKPT ==="
        cd /work/experiments/distill-soft
        python scripts/eval.py \
            --ckpt "$LATEST_CKPT" \
            --workers "$EVAL_WORKERS" \
            --games-per-worker "$EVAL_GAMES_PER_WORKER" \
            --sims "$EVAL_SIMS" \
            --n-blocks "$N_BLOCKS" \
            --n-filters "$N_FILTERS" \
            --stockfish-elo "$STOCKFISH_ELO" \
            --agent-device cuda \
            2>&1 | tee "$EVAL_LOG" || echo "=== eval failed; continuing ==="
        echo "=== eval log written to $EVAL_LOG ==="
    fi
fi

echo "=== final S3 sync (training + eval artifacts) ==="
aws s3 sync "$CKPT_DIR/" "$CKPT_S3_BASE/" --no-progress --exclude "*.tmp.*" --exclude ".eval_progress_*"

echo "=== $(date -u) done ==="
