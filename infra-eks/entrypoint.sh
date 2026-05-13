#!/usr/bin/env bash
# Dispatcher for the wm-chess EKS container.
#
# Modes:
#   gen     — runs generate_data.py with seed derived from
#             $JOB_COMPLETION_INDEX (k8s indexed Job env). Writes the
#             resulting library leaf directly to S3.
#   merge   — pulls all shards from S3, runs merge_shards.py, pushes
#             the merged dataset back to S3.
#
# Required env (gen):
#   BASE_SEED           default 42
#   GAMES_PER_POD       default 12500
#   DEPTH               default 15
#   MULTIPV             default 8
#   T_PAWNS             default 1.0
#   S3_BUCKET           e.g. wm-chess-library
#   S3_PREFIX           e.g. d15-mpv8-T1-g100000-2026-05-12
#
# Required env (merge):
#   S3_BUCKET, S3_PREFIX as above
#   N_PODS              total shards to expect (default 8)

set -euo pipefail

MODE="${1:-}"
: "${S3_BUCKET:?Set S3_BUCKET=<your-bucket>}"
: "${S3_PREFIX:?Set S3_PREFIX=<run-prefix>}"

case "$MODE" in
    gen)
        : "${BASE_SEED:=42}"
        : "${GAMES_PER_POD:=12500}"
        : "${DEPTH:=15}"
        : "${MULTIPV:=8}"
        : "${T_PAWNS:=1.0}"
        : "${PARTIAL_SYNC_INTERVAL:=600}"   # seconds between partial S3 syncs
        IDX="${JOB_COMPLETION_INDEX:-0}"
        SEED=$(( BASE_SEED + IDX * 1000 ))
        echo "[gen] pod $IDX  seed=$SEED  games=$GAMES_PER_POD  d$DEPTH  mpv$MULTIPV  T=$T_PAWNS"

        WORKERS="$(nproc)"
        WORKERS=$((WORKERS - 2)); [ "$WORKERS" -lt 1 ] && WORKERS=1

        cd /work/experiments/distill-soft
        LIB_LOCAL=/tmp/library
        mkdir -p "$LIB_LOCAL"

        # Partial staging lives under a SEPARATE top-level prefix so the
        # merge glob `shards/pod-*` can never match it accidentally.
        PARTIAL_S3="s3://$S3_BUCKET/$S3_PREFIX/shards_partial/pod-$IDX"

        # ---- RESUME ----
        # If a prior attempt for this pod uploaded partial chunks, pull
        # them back so generate_data.py picks up where it left off
        # (chunk granularity = up to chunk_size games of waste per worker).
        echo "[gen] checking for partial chunks at $PARTIAL_S3/"
        if aws s3 ls "$PARTIAL_S3/" >/dev/null 2>&1; then
            echo "[gen] resuming: pulling partial chunks from $PARTIAL_S3"
            aws s3 sync "$PARTIAL_S3/" "$LIB_LOCAL/games/" --no-progress
            CHUNK_COUNT=$(find "$LIB_LOCAL/games" -name "chunk_*.npz" 2>/dev/null | wc -l)
            echo "[gen] resumed with $CHUNK_COUNT prior chunks"
        else
            echo "[gen] no prior partial chunks — fresh start"
        fi

        # ---- BACKGROUND PARTIAL UPLOADER ----
        # Every PARTIAL_SYNC_INTERVAL seconds, snapshot /tmp/library/games to
        # s3://.../pod-N-partial/. On reclamation, retry pod pulls this back.
        (
            while true; do
                sleep "$PARTIAL_SYNC_INTERVAL"
                aws s3 sync "$LIB_LOCAL/games/" "$PARTIAL_S3/" --no-progress --quiet 2>/dev/null \
                    && echo "[partial-sync] $(date -u +%H:%M:%S) snapshot ok" \
                    || echo "[partial-sync] $(date -u +%H:%M:%S) snapshot FAILED (will retry)"
            done
        ) &
        PARTIAL_PID=$!
        # Make sure background loop dies with the script.
        trap "kill $PARTIAL_PID 2>/dev/null" EXIT

        uv run python scripts/generate_data.py \
            --n-games "$GAMES_PER_POD" --workers "$WORKERS" \
            --depth "$DEPTH" --multipv "$MULTIPV" \
            --temperature-pawns "$T_PAWNS" --seed "$SEED" \
            --chunk-size 50 \
            --library-root "$LIB_LOCAL"

        # Stop the background uploader before final sync so we don't race.
        kill "$PARTIAL_PID" 2>/dev/null || true

        echo "[gen] uploading final shard to s3://$S3_BUCKET/$S3_PREFIX/shards/pod-$IDX/"
        aws s3 sync "$LIB_LOCAL/games/" "s3://$S3_BUCKET/$S3_PREFIX/shards/pod-$IDX/" --no-progress

        # Clean up partial staging — final shard supersedes it.
        aws s3 rm "$PARTIAL_S3/" --recursive --quiet 2>/dev/null || true
        echo "[gen] pod $IDX done."
        ;;

    merge)
        : "${N_PODS:=8}"
        : "${MULTIPV:=8}"
        : "${T_PAWNS:=1.0}"
        echo "[merge] pulling $N_PODS shards from s3://$S3_BUCKET/$S3_PREFIX/shards/"

        LOCAL=/tmp/shards
        mkdir -p "$LOCAL"
        aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/shards/" "$LOCAL/" --no-progress
        echo "[merge] tree pulled:"
        find "$LOCAL" -maxdepth 5 -type d | head

        # Each pod-N/ contains an sf-<v>/d.../g<N>-seed<S>/ subtree.
        # Glob for any data.npz under the synced tree to find all shards.
        SHARDS_GLOB="$LOCAL/pod-*/sf-*/d*-mpv*-T*/g*-seed*"

        OUT=/tmp/merged
        mkdir -p "$OUT"
        uv run --project /work/wm_chess python /work/wm_chess/scripts/merge_shards.py \
            --shards-glob "$SHARDS_GLOB" \
            --output "$OUT" \
            --multipv "$MULTIPV" --temperature "$T_PAWNS"

        echo "[merge] uploading merged dataset to s3://$S3_BUCKET/$S3_PREFIX/merged/"
        aws s3 sync "$OUT/" "s3://$S3_BUCKET/$S3_PREFIX/merged/" --no-progress
        echo "[merge] done. Merged dataset at s3://$S3_BUCKET/$S3_PREFIX/merged/data.npz"
        ;;

    "" )
        echo "usage: gen | merge" >&2; exit 2 ;;

    * )
        echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
