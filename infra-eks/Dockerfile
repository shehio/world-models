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
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscli.zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv (fast python pkg manager), used to install our wm_chess + distill-soft packages.
RUN pip install --no-cache-dir uv

WORKDIR /work
COPY wm_chess /work/wm_chess
COPY experiments/distill-soft /work/experiments/distill-soft

# Install both packages in editable mode so the layout matches the repo
# (distill-soft pyproject.toml has ../../wm_chess relative path).
RUN cd /work/experiments/distill-soft && uv sync --frozen 2>/dev/null || cd /work/experiments/distill-soft && uv sync

# Entrypoint script: dispatch on first arg.
COPY infra-eks/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
