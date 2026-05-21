#!/usr/bin/env bash
# Dispatcher for the wm-go EKS container.
#
# Same shape as infra-eks/entrypoint.sh (the chess equivalent): one `gen`
# mode that produces N games, writes worker_*.npz chunks under
# $LIB_LOCAL, and periodically syncs them to S3 under shards_partial/.
#
# Required env (gen):
#   BASE_SEED           default 42
#   GAMES_PER_POD       default 4000
#   BOARD_SIZE          default 9
#   N_INPUT_PLANES      default 4
#   VISITS              default 400
#   TOP_K               default 8
#   TEMPERATURE         default 1.0
#   KOMI                default 7.5
#   RULES               default tromp-taylor
#   S3_BUCKET           e.g. wm-chess-library-594561963943
#   S3_PREFIX           e.g. go-9x9-mpv8-v400-20260521T0700Z

set -euo pipefail

MODE="${1:-}"
: "${S3_BUCKET:?Set S3_BUCKET=<your-bucket>}"
: "${S3_PREFIX:?Set S3_PREFIX=<run-prefix>}"

case "$MODE" in
    gen)
        : "${BASE_SEED:=42}"
        : "${GAMES_PER_POD:=4000}"
        : "${BOARD_SIZE:=9}"
        : "${N_INPUT_PLANES:=4}"
        : "${VISITS:=400}"
        : "${TOP_K:=8}"
        : "${TEMPERATURE:=1.0}"
        : "${KOMI:=7.5}"
        : "${RULES:=tromp-taylor}"
        : "${PARTIAL_SYNC_INTERVAL:=300}"   # tighter than chess; Go games are shorter
        IDX="${JOB_COMPLETION_INDEX:-0}"
        SEED=$(( BASE_SEED + IDX * 1000 ))
        echo "[gen-go] pod $IDX  seed=$SEED  games=$GAMES_PER_POD  board=${BOARD_SIZE}x${BOARD_SIZE}  visits=$VISITS  K=$TOP_K"

        # KataGo analysis engines are multi-threaded internally (see
        # katago-analysis.cfg: numAnalysisThreads=4 + numSearchThreads=4).
        # We want WORKERS * 8 ≈ vCPUs to fully use the core count without
        # heavy oversubscription. On c7a.8xlarge (32 vCPUs) that's ~4 workers.
        VCPUS="$(nproc)"
        WORKERS="${WORKERS:-$(( VCPUS / 8 ))}"
        [ "$WORKERS" -lt 1 ] && WORKERS=1
        echo "[gen-go] $VCPUS vCPUs → $WORKERS worker procs × ~8 katago threads each"

        cd /work/experiments/distill-go
        LIB_LOCAL=/tmp/go-library
        mkdir -p "$LIB_LOCAL"

        PARTIAL_S3="s3://$S3_BUCKET/$S3_PREFIX/shards_partial/pod-$IDX"

        # ---- RESUME ----
        # Pull any chunks left from a prior pod incarnation. The
        # generate_data.py CLI's chunk_idx resume logic skips past
        # existing files — matches the chess pipeline contract from
        # the feedback_datagen_resume memory.
        echo "[gen-go] checking for partial chunks at $PARTIAL_S3/"
        if aws s3 ls "$PARTIAL_S3/" >/dev/null 2>&1; then
            echo "[gen-go] resuming: pulling partial chunks from $PARTIAL_S3"
            aws s3 sync "$PARTIAL_S3/" "$LIB_LOCAL/" --no-progress
            CHUNK_COUNT=$(find "$LIB_LOCAL" -name "worker_*.npz" 2>/dev/null | wc -l || true)
            echo "[gen-go] resumed with $CHUNK_COUNT prior chunks"
        else
            echo "[gen-go] no prior partial chunks — fresh start"
        fi

        # ---- BACKGROUND PARTIAL UPLOADER ----
        (
            while true; do
                sleep "$PARTIAL_SYNC_INTERVAL"
                aws s3 sync "$LIB_LOCAL/" "$PARTIAL_S3/" --no-progress --quiet \
                    && echo "[partial-sync] $(date -u +%H:%M:%S) snapshot ok" \
                    || echo "[partial-sync] $(date -u +%H:%M:%S) snapshot FAILED (will retry)"
            done
        ) &
        PARTIAL_PID=$!
        trap "kill $PARTIAL_PID 2>/dev/null" EXIT

        # Games per worker = total per pod / N workers. Round up so we
        # always hit the target.
        GAMES_PER_WORKER=$(( (GAMES_PER_POD + WORKERS - 1) / WORKERS ))
        echo "[gen-go] $GAMES_PER_WORKER games per worker × $WORKERS workers"

        uv run python scripts/generate_data.py \
            --workers "$WORKERS" \
            --games-per-worker "$GAMES_PER_WORKER" \
            --board-size "$BOARD_SIZE" \
            --n-input-planes "$N_INPUT_PLANES" \
            --visits "$VISITS" \
            --top-k "$TOP_K" \
            --temperature "$TEMPERATURE" \
            --komi "$KOMI" \
            --rules "$RULES" \
            --chunk-size 10 \
            --output-dir "$LIB_LOCAL"

        kill "$PARTIAL_PID" 2>/dev/null || true

        echo "[gen-go] uploading final chunks to s3://$S3_BUCKET/$S3_PREFIX/shards/pod-$IDX/"
        aws s3 sync "$LIB_LOCAL/" "s3://$S3_BUCKET/$S3_PREFIX/shards/pod-$IDX/" --no-progress

        # Clean up partial staging once final shard is up — the merge step
        # reads from shards/, not shards_partial/.
        aws s3 rm "$PARTIAL_S3/" --recursive --quiet 2>/dev/null || true
        echo "[gen-go] pod $IDX done."
        ;;

    "" )
        echo "usage: gen" >&2; exit 2 ;;

    * )
        echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
