#!/usr/bin/env bash
# Full 02c overnight pipeline: data gen → training → eval.
#
# All knobs are env-var overridable so the same script runs unchanged on a
# laptop (MPS, 6 workers) and on AWS (CUDA, 90 workers). Defaults match the
# laptop overnight setup.
#
# Run from 02c-distill-scaled root:
#     bash scripts/run_overnight.sh
#     # or with overrides:
#     DEVICE=cuda DATAGEN_WORKERS=90 EVAL_WORKERS=24 bash scripts/run_overnight.sh
#
# Observability:
#   - combined log:      $LOG  (every phase tee'd here)
#   - per-phase logs:    /tmp/az_02c_{phase1,phase2,phase3a,phase3b}.log
#   - in-flight progress: data/.progress_wNN.json (one per Phase-1 worker)
#     a background poller dumps an aggregated line into $LOG every PROGRESS_INTERVAL sec
#   - heartbeat file:    /tmp/az_02c_heartbeat (mtime updated every minute by orchestrator)

set -uo pipefail
cd "$(dirname "$0")/.."

# ---- config ---------------------------------------------------------------
: "${LOG:=/tmp/az_02c_overnight.log}"
: "${DATA:=data/sf_d15_mpv8_4000g.npz}"
: "${CKPT_DIR:=checkpoints/run01}"

# Phase-1 (data gen) knobs
: "${N_GAMES:=4000}"
: "${DATAGEN_WORKERS:=6}"          # laptop default; bump to ~vCPU on AWS
: "${DEPTH:=15}"
: "${MULTIPV:=8}"
: "${TEMP_PAWNS:=1.0}"

# Phase-2 (training) knobs
: "${EPOCHS:=20}"
: "${N_BLOCKS:=20}"
: "${N_FILTERS:=256}"
: "${DEVICE:=}"                    # empty → train.py auto-picks (MPS > CUDA > CPU)

# Phase-3 (eval) knobs
: "${EVAL_WORKERS:=4}"             # laptop default; bump on AWS
: "${EVAL_GAMES_PER_WORKER:=25}"   # so total = EVAL_WORKERS * 25
: "${SIMS_BASELINE:=800}"
: "${SIMS_STRETCH:=1600}"
: "${SF_ELO:=1320}"

: "${PROGRESS_INTERVAL:=60}"       # seconds between aggregated progress prints

FINAL_CKPT="${CKPT_DIR}/distilled_epoch$(printf '%03d' $((EPOCHS - 1))).pt"
PROGRESS_DIR="$(dirname "$DATA")"

# device flag for train.py
DEVICE_ARG=()
if [ -n "$DEVICE" ]; then DEVICE_ARG=(--device "$DEVICE"); fi

# ---- header ---------------------------------------------------------------
{
  echo "=== 02c overnight run started at $(date) ==="
  echo "config:"
  echo "  N_GAMES=$N_GAMES  DEPTH=$DEPTH  MULTIPV=$MULTIPV  T=$TEMP_PAWNS"
  echo "  DATAGEN_WORKERS=$DATAGEN_WORKERS  EVAL_WORKERS=$EVAL_WORKERS"
  echo "  EPOCHS=$EPOCHS  net=${N_BLOCKS}x${N_FILTERS}  DEVICE='${DEVICE:-auto}'"
  echo "  SIMS_BASELINE=$SIMS_BASELINE  SIMS_STRETCH=$SIMS_STRETCH  SF_ELO=$SF_ELO"
  echo "  LOG=$LOG  DATA=$DATA  CKPT_DIR=$CKPT_DIR"
} | tee "$LOG"

# ---- heartbeat ------------------------------------------------------------
# Touch /tmp/az_02c_heartbeat once a minute so external watchers can tell us
# apart from "process exists but is stuck."
(
  while true; do
    date -u +%s > /tmp/az_02c_heartbeat
    sleep 60
  done
) &
HEARTBEAT_PID=$!

# ---- Phase 1: data generation ---------------------------------------------
PHASE1_LOG=/tmp/az_02c_phase1.log
echo "" | tee -a "$LOG"
echo "=== Phase 1: data gen | depth=$DEPTH multipv=$MULTIPV games=$N_GAMES workers=$DATAGEN_WORKERS | $(date) ===" | tee -a "$LOG"
PHASE1_START=$(date +%s)

# Background progress poller — reads data/.progress_wNN.json every minute,
# prints one aggregated line into the log. Exits when all workers report done.
mkdir -p "$PROGRESS_DIR"
(
  while sleep "$PROGRESS_INTERVAL"; do
    if ! pgrep -f "scripts/generate_data.py" >/dev/null; then
      break  # data gen finished; stop polling
    fi
    line=$(uv run python scripts/progress.py "$PROGRESS_DIR" 2>&1 || true)
    echo "  [progress] $line" | tee -a "$LOG"
  done
) &
PROGRESS_PID=$!

# Run data gen (foreground; tee output to phase log + combined log)
uv run python scripts/generate_data.py \
    --n-games "$N_GAMES" --workers "$DATAGEN_WORKERS" \
    --depth "$DEPTH" --multipv "$MULTIPV" \
    --temperature-pawns "$TEMP_PAWNS" \
    --output "$DATA" \
    --progress-dir "$PROGRESS_DIR" \
    2>&1 | tee "$PHASE1_LOG" | tee -a "$LOG"

# Make sure progress poller stops even if generate_data.py exited cleanly.
kill "$PROGRESS_PID" 2>/dev/null || true
wait "$PROGRESS_PID" 2>/dev/null || true

PHASE1_END=$(date +%s)
echo "=== Phase 1 wall: $((PHASE1_END - PHASE1_START))s ===" | tee -a "$LOG"

if [ ! -f "$DATA" ]; then
    echo "FATAL: data generation did not produce $DATA — aborting." | tee -a "$LOG"
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    exit 1
fi

# ---- Phase 2: training ----------------------------------------------------
PHASE2_LOG=/tmp/az_02c_phase2.log
echo "" | tee -a "$LOG"
echo "=== Phase 2: training | net=${N_BLOCKS}x${N_FILTERS} epochs=$EPOCHS device='${DEVICE:-auto}' | $(date) ===" | tee -a "$LOG"
PHASE2_START=$(date +%s)

uv run python scripts/train.py \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    "${DEVICE_ARG[@]}" \
    --n-blocks "$N_BLOCKS" --n-filters "$N_FILTERS" \
    --ckpt-dir "$CKPT_DIR" \
    --save-every 5 \
    2>&1 | tee "$PHASE2_LOG" | tee -a "$LOG"

PHASE2_END=$(date +%s)
echo "=== Phase 2 wall: $((PHASE2_END - PHASE2_START))s ===" | tee -a "$LOG"

if [ ! -f "$FINAL_CKPT" ]; then
    echo "FATAL: training did not produce $FINAL_CKPT — aborting." | tee -a "$LOG"
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    exit 1
fi

# ---- Phase 3a: eval @ baseline sims ---------------------------------------
PHASE3A_LOG=/tmp/az_02c_phase3a.log
echo "" | tee -a "$LOG"
echo "=== Phase 3a: eval | $((EVAL_WORKERS * EVAL_GAMES_PER_WORKER))g vs SF$SF_ELO | $SIMS_BASELINE sims | $(date) ===" | tee -a "$LOG"
PHASE3A_START=$(date +%s)

uv run python scripts/eval.py \
    --ckpt "$FINAL_CKPT" \
    --workers "$EVAL_WORKERS" --games-per-worker "$EVAL_GAMES_PER_WORKER" \
    --sims "$SIMS_BASELINE" \
    --n-blocks "$N_BLOCKS" --n-filters "$N_FILTERS" \
    --stockfish-elo "$SF_ELO" \
    2>&1 | tee "$PHASE3A_LOG" | tee -a "$LOG"

PHASE3A_END=$(date +%s)
echo "=== Phase 3a wall: $((PHASE3A_END - PHASE3A_START))s ===" | tee -a "$LOG"

# ---- Phase 3b: stretch eval @ higher sims ---------------------------------
PHASE3B_LOG=/tmp/az_02c_phase3b.log
echo "" | tee -a "$LOG"
echo "=== Phase 3b: stretch eval | $((EVAL_WORKERS * EVAL_GAMES_PER_WORKER))g vs SF$SF_ELO | $SIMS_STRETCH sims | $(date) ===" | tee -a "$LOG"
PHASE3B_START=$(date +%s)

uv run python scripts/eval.py \
    --ckpt "$FINAL_CKPT" \
    --workers "$EVAL_WORKERS" --games-per-worker "$EVAL_GAMES_PER_WORKER" \
    --sims "$SIMS_STRETCH" \
    --n-blocks "$N_BLOCKS" --n-filters "$N_FILTERS" \
    --stockfish-elo "$SF_ELO" \
    2>&1 | tee "$PHASE3B_LOG" | tee -a "$LOG"

PHASE3B_END=$(date +%s)
echo "=== Phase 3b wall: $((PHASE3B_END - PHASE3B_START))s ===" | tee -a "$LOG"

# ---- Done ------------------------------------------------------------------
TOTAL_END=$(date +%s)
{
  echo ""
  echo "=== 02c overnight run COMPLETE at $(date) ==="
  echo "totals:"
  echo "  Phase 1 (data gen):  $((PHASE1_END - PHASE1_START))s"
  echo "  Phase 2 (training):  $((PHASE2_END - PHASE2_START))s"
  echo "  Phase 3a (eval @ $SIMS_BASELINE):  $((PHASE3A_END - PHASE3A_START))s"
  echo "  Phase 3b (eval @ $SIMS_STRETCH):   $((PHASE3B_END - PHASE3B_START))s"
} | tee -a "$LOG"

kill "$HEARTBEAT_PID" 2>/dev/null || true
