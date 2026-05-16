# Container for wm-chess generation + merge jobs on EKS.
# Two modes selected by entrypoint: `gen` runs generate_data.py with seed
# derived from JOB_COMPLETION_INDEX; `merge` runs merge_shards.py against
# S3-staged shards.

FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

# System deps: stockfish + AWS CLI (for S3 I/O).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        stockfish \
        ca-certificates \
        curl \
        unzip \
    && ln -sf /usr/games/stockfish /usr/local/bin/stockfish \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscli.zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY wm_chess /work/wm_chess
COPY experiments/distill-soft /work/experiments/distill-soft

# Install runtime deps + our editable packages.
#
# This used to be `uv sync --frozen` from inside experiments/distill-soft.
# That broke after the workspace migration (a36a324) because the
# distill-soft pyproject's `tool.uv.sources = { wm-chess = { workspace = true } }`
# only resolves with the workspace root pyproject.toml + uv.lock in
# scope, neither of which is copied into the image.
#
# Fix: same plain-pip pattern as Dockerfile.train. CPU-only torch (no
# CUDA — datagen workload is Stockfish, not the NN). `--no-deps` on the
# editable installs prevents re-pulling torch/numpy.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" \
    && pip install --no-cache-dir "numpy>=1.26" "python-chess>=1.999" "tqdm>=4.66" \
    && pip install --no-cache-dir --no-deps -e /work/wm_chess \
    && pip install --no-cache-dir --no-deps -e /work/experiments/distill-soft

# Entrypoint script: dispatch on first arg.
COPY infra-eks/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
