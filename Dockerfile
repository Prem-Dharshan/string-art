# Reproducible Linux environment for the string-art pipeline, tests and experiments.
#   docker compose build
#   docker compose run --rm stringart pytest -q
# Datasets, face models and outputs live in named volumes (see compose.yaml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

# OpenCV (contrib, non-headless wheel) needs libGL and glib at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project --extra demo
COPY . .
RUN uv sync --frozen --extra demo \
    && mkdir -p data/raw data/raw_heldout models outputs

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync"]
CMD ["pytest", "-q"]
