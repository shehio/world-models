#!/usr/bin/env bash
# Full 02c overnight pipeline: data gen → training → eval.
# Designed to fit ~8h on the laptop and produce one go/no-go number
# by morning: 100 games vs Stockfish UCI_Elo=1320 at 800 sims.
#
# Run from 02c-distill-scaled root.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=/tmp/az_02c_overnight.log
DATA=data/sf_d15_mpv8_4000g.npz
CKPT_DIR=checkpoints/run01
FINAL_CKPT="${CKPT_DIR}/distilled_epoch019.pt"

echo "=== 02c overnight run started at $(date) ===" | tee "$LOG"
echo "writing detailed log to $LOG"

# ----- Phase 1: data generation ---------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Phase 1: data gen | depth=15 | multipv=8 | 4000 games | $(date) ===" | tee -a "$LOG"
uv run python scripts/generate_data.py \
    --n-games 4000 --workers 6 --depth 15 --multipv 8 \
    --temperature-pawns 1.0 \
    --output "$DATA" \
    2>&1 | tee -a "$LOG"

if [ ! -f "$DATA" ]; then
    echo "FATAL: data generation did not produce $DATA — aborting." | tee -a "$LOG"
    exit 1
fi

# ----- Phase 2: training ----------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Phase 2: training | 20x256 | 20 epochs | MPS | $(date) ===" | tee -a "$LOG"
uv run python scripts/train.py \
    --data "$DATA" \
    --epochs 20 --device mps \
    --n-blocks 20 --n-filters 256 \
    --ckpt-dir "$CKPT_DIR" \
    --save-every 5 \
    2>&1 | tee -a "$LOG"

if [ ! -f "$FINAL_CKPT" ]; then
    echo "FATAL: training did not produce $FINAL_CKPT — aborting." | tee -a "$LOG"
    exit 1
fi

# ----- Phase 3: eval vs Stockfish UCI_Elo=1320 ------------------------------
echo "" | tee -a "$LOG"
echo "=== Phase 3a: eval | 100g vs SF1320 | 800 sims | $(date) ===" | tee -a "$LOG"
uv run python scripts/eval.py \
    --ckpt "$FINAL_CKPT" \
    --workers 4 --games-per-worker 25 --sims 800 \
    --n-blocks 20 --n-filters 256 \
    --stockfish-elo 1320 \
    2>&1 | tee -a "$LOG"

# Stretch: also eval at 1600 sims (more search = more Elo). Skip if time is tight.
echo "" | tee -a "$LOG"
echo "=== Phase 3b: stretch eval | 100g vs SF1320 | 1600 sims | $(date) ===" | tee -a "$LOG"
uv run python scripts/eval.py \
    --ckpt "$FINAL_CKPT" \
    --workers 4 --games-per-worker 25 --sims 1600 \
    --n-blocks 20 --n-filters 256 \
    --stockfish-elo 1320 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 02c overnight run COMPLETE at $(date) ===" | tee -a "$LOG"
