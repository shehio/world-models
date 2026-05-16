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

# uv (fast python pkg manager). The entrypoint shells out to `uv run`
# for both gen and merge, so we install uv globally and let it manage
# the workspace .venv.
RUN pip install --no-cache-dir uv

# Mirror the repo's workspace layout into /work. Required because
# distill-soft's pyproject declares wm-chess as a workspace dep
# (tool.uv.sources = { workspace = true }), which only resolves when
# the workspace root pyproject + uv.lock are in scope AND all members
# listed in [tool.uv.workspace] exist on disk. Total git-tracked
# content across all four members is ~700 KB.
WORKDIR /work
COPY pyproject.toml uv.lock /work/
COPY wm_chess /work/wm_chess
COPY experiments/selfplay /work/experiments/selfplay
COPY experiments/distill-hard /work/experiments/distill-hard
COPY experiments/distill-soft /work/experiments/distill-soft

# Sync the locked deps into /work/.venv. `--frozen` makes uv.lock
# authoritative (no resolver run); `--package distill-soft` installs
# distill-soft and its transitive deps (including wm-chess) only —
# selfplay and distill-hard pyprojects are present so the workspace
# graph validates, but their deps aren't installed.
RUN uv sync --frozen --package distill-soft

# Entrypoint script: dispatch on first arg.
COPY infra-eks/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
