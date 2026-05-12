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
        IDX="${JOB_COMPLETION_INDEX:-0}"
        SEED=$(( BASE_SEED + IDX * 1000 ))
        echo "[gen] pod $IDX  seed=$SEED  games=$GAMES_PER_POD  d$DEPTH  mpv$MULTIPV  T=$T_PAWNS"

        WORKERS="$(nproc)"
        WORKERS=$((WORKERS - 2)); [ "$WORKERS" -lt 1 ] && WORKERS=1

        cd /work/experiments/distill-soft
        LIB_LOCAL=/tmp/library
        mkdir -p "$LIB_LOCAL"

        uv run python scripts/generate_data.py \
            --n-games "$GAMES_PER_POD" --workers "$WORKERS" \
            --depth "$DEPTH" --multipv "$MULTIPV" \
            --temperature-pawns "$T_PAWNS" --seed "$SEED" \
            --chunk-size 50 \
            --library-root "$LIB_LOCAL"

        echo "[gen] uploading shard to s3://$S3_BUCKET/$S3_PREFIX/shards/pod-$IDX/"
        aws s3 sync "$LIB_LOCAL/games/" "s3://$S3_BUCKET/$S3_PREFIX/shards/pod-$IDX/" --no-progress
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
