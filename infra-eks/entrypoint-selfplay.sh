#!/usr/bin/env bash
# AlphaZero-style self-play RL loop, initialized from a distilled prior.
# Lc0's recipe: distill first, RL second.
#
# Flow:
#   1. Download the prior checkpoint from $INIT_FROM_S3 (e.g. d15 ep 20).
#   2. Run selfplay_loop_mp.py with --resume on that ckpt.
#      Workers play self-play games with 800-sim MCTS, the trainer fits
#      the network to (state, MCTS visit distribution, game outcome)
#      triples. Iterates until $TIME_BUDGET seconds elapse.
#   3. Periodically sync local checkpoints/ to S3.
#
# Output layout (mirrors the distill-soft pattern):
#   s3://$S3_BUCKET/$S3_PREFIX/selfplay/net-<N>x<F>/<run-id>/
#     iter_NNN.pt           the AZ-loop's iter checkpoints
#     net_hour_NNN.pt       optional hourly dumps for evolution tracking
#     selfplay_log.txt      stdout/stderr mirror
#     run_metadata.json     hparams as JSON

set -euo pipefail

: "${S3_BUCKET:?required (e.g. wm-chess-library-594561963943)}"
: "${S3_PREFIX:?required (e.g. d15-mpv8-T1-g100000-20260512T1844Z)}"
: "${AWS_DEFAULT_REGION:?required}"
: "${INIT_FROM_S3:?required (e.g. s3://.../distilled_epoch019.pt)}"

# Self-play / training hyperparameters. Defaults match the AZ paper
# where possible (20x256 net, 800 sims, ~25 train-steps per iter).
: "${N_BLOCKS:=20}"
: "${N_FILTERS:=256}"
: "${SIMS:=800}"
: "${WORKERS:=24}"           # one worker per CPU core on g6e.8xlarge / g6.8xlarge
: "${GAMES_PER_WORKER:=2}"   # per iter, per worker. 24*2=48 games/iter.
: "${BATCH_SIZE:=8}"         # batched MCTS leaf-eval batch size (inside MCTS)
: "${TRAIN_STEPS:=200}"      # SGD steps per iter
: "${LR:=1e-3}"
: "${TIME_BUDGET:=43200}"    # 12 h
: "${EVAL_EVERY:=4}"
: "${EVAL_GAMES:=20}"
: "${EVAL_SIMS:=200}"
: "${REPLAY_SHARDS:=20}"
: "${N_HISTORY:=1}"          # 1 = legacy 19-plane encoding (matches d15 ckpts)

# PCR (KataGo) is on by default — gives ~+80-150 Elo at fixed compute.
: "${PCR:=1}"
: "${PCR_SIMS_FULL:=$SIMS}"
: "${PCR_SIMS_REDUCED:=$(($SIMS / 5))}"   # 800 -> 160; ~5x cheaper non-target moves
: "${PCR_P_FULL:=0.25}"

WORK=/work
CKPT_DIR=$WORK/checkpoints_selfplay
mkdir -p "$CKPT_DIR"

ARCH="net-${N_BLOCKS}x${N_FILTERS}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%MZ)}"
CKPT_S3_BASE="s3://$S3_BUCKET/$S3_PREFIX/selfplay/$ARCH/$RUN_ID"
echo "selfplay checkpoint S3 base: $CKPT_S3_BASE"

echo "=== $(date -u) selfplay-from-prior ==="
echo "REGION=$AWS_DEFAULT_REGION BUCKET=$S3_BUCKET PREFIX=$S3_PREFIX"
echo "ARCH=$ARCH SIMS=$SIMS WORKERS=$WORKERS GAMES_PER_WORKER=$GAMES_PER_WORKER"
echo "TIME_BUDGET=${TIME_BUDGET}s ($((TIME_BUDGET / 3600))h) RUN_ID=$RUN_ID"
nvidia-smi || echo "(no nvidia-smi)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

cat > "$CKPT_DIR/run_metadata.json" <<META
{
  "run_id": "$RUN_ID",
  "kind": "selfplay-from-prior",
  "arch": "$ARCH",
  "n_blocks": $N_BLOCKS,
  "n_filters": $N_FILTERS,
  "sims": $SIMS,
  "workers": $WORKERS,
  "games_per_worker": $GAMES_PER_WORKER,
  "train_steps_per_iter": $TRAIN_STEPS,
  "lr": $LR,
  "time_budget_s": $TIME_BUDGET,
  "n_history": $N_HISTORY,
  "init_from_s3": "$INIT_FROM_S3",
  "s3_ckpt_base": "$CKPT_S3_BASE",
  "pcr": $([ "$PCR" = "1" ] && echo "true" || echo "false"),
  "pcr_sims_full": $PCR_SIMS_FULL,
  "pcr_sims_reduced": $PCR_SIMS_REDUCED,
  "pcr_p_full": $PCR_P_FULL,
  "started_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META
cat "$CKPT_DIR/run_metadata.json"

# Download the distilled prior.
echo "=== fetching prior $INIT_FROM_S3 ==="
PRIOR_LOCAL="$CKPT_DIR/prior.pt"
aws s3 cp "$INIT_FROM_S3" "$PRIOR_LOCAL" --no-progress
ls -la "$PRIOR_LOCAL"

# Resume bookkeeping for an interrupted run: if any iter_*.pt has
# already landed in S3 under this RUN_ID, pull the latest as the
# resume point. Otherwise start fresh from the distilled prior.
RESUME_FROM="$PRIOR_LOCAL"
START_ITER=0
START_HOUR=0
LATEST_ITER_S3=$(aws s3 ls "$CKPT_S3_BASE/" 2>/dev/null \
    | awk '{print $NF}' | grep -E '^iter_[0-9]+\.pt$' | sort -V | tail -1 || true)
if [ -n "$LATEST_ITER_S3" ]; then
    echo "=== same-run resume: $LATEST_ITER_S3 already in S3 ==="
    aws s3 cp "$CKPT_S3_BASE/$LATEST_ITER_S3" "$CKPT_DIR/$LATEST_ITER_S3" --no-progress
    RESUME_FROM="$CKPT_DIR/$LATEST_ITER_S3"
    START_ITER=$(echo "$LATEST_ITER_S3" | sed -E 's/iter_0*([0-9]+)\.pt/\1/')
    START_ITER=$((START_ITER + 1))
fi

# Background S3-sync loop: every 10 minutes, push whatever new
# checkpoints have landed locally. Killed on EXIT.
(
    while true; do
        sleep 600
        aws s3 sync "$CKPT_DIR/" "$CKPT_S3_BASE/" \
            --no-progress --exclude "*.tmp.*" 2>&1 | tail -5
    done
) &
SYNC_PID=$!
trap 'echo "=== final sync ==="; kill $SYNC_PID 2>/dev/null || true; aws s3 sync "$CKPT_DIR/" "$CKPT_S3_BASE/" --no-progress --exclude "*.tmp.*" || true' EXIT

cd /work/experiments/selfplay

PCR_ARGS=()
if [ "$PCR" = "1" ]; then
    PCR_ARGS=(--pcr
              --pcr-sims-full "$PCR_SIMS_FULL"
              --pcr-sims-reduced "$PCR_SIMS_REDUCED"
              --pcr-p-full "$PCR_P_FULL")
fi

echo "=== launching self-play loop ==="
python scripts/selfplay_loop_mp.py \
    --resume "$RESUME_FROM" \
    --start-iter "$START_ITER" \
    --start-hour "$START_HOUR" \
    --time-budget "$TIME_BUDGET" \
    --workers "$WORKERS" \
    --games-per-worker "$GAMES_PER_WORKER" \
    --sims "$SIMS" \
    --batch-size "$BATCH_SIZE" \
    --n-blocks "$N_BLOCKS" \
    --n-filters "$N_FILTERS" \
    --train-steps "$TRAIN_STEPS" \
    --lr "$LR" \
    --replay-shards "$REPLAY_SHARDS" \
    --n-history "$N_HISTORY" \
    --train-device auto \
    --eval-every "$EVAL_EVERY" \
    --eval-games "$EVAL_GAMES" \
    --eval-sims "$EVAL_SIMS" \
    --hourly-dump \
    --ckpt-dir "$CKPT_DIR" \
    "${PCR_ARGS[@]}" \
    2>&1 | tee "$CKPT_DIR/selfplay_log.txt"

echo "=== $(date -u) selfplay done ==="
