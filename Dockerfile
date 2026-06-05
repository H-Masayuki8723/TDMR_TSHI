FROM python:3.11-slim

# Headless matplotlib, unbuffered logs, no pip cache.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build toolchain (kept minimal; wheels cover numpy/pandas/matplotlib on slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY pyproject.toml setup.cfg README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ".[dev]"

# Project assets (configs, codebook cache dir, tests).
COPY configs ./configs
COPY data ./data
COPY tests ./tests

# Output tree (also bind-mounted by docker-compose so results land on the host).
RUN mkdir -p outputs/runs outputs/summaries outputs/figures outputs/reports

# The `tdmr2d` console script is on PATH. docker-compose passes the actual
# command, e.g. `docker compose run --rm sim tdmr2d smoke`.
CMD ["tdmr2d", "smoke"]
