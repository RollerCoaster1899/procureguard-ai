# ProcureGuard AI - container image
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install project dependencies from the lock file first for caching.
COPY uv.lock pyproject.toml README.md ./
RUN pip install uv \
    && uv sync --frozen --no-dev

# Copy source, configs, and scripts (no stale data/ artifacts).
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# Generate fresh deterministic processed data during the build. No external
# model downloads or network access are required for this step.
RUN uv run python scripts/generate_data.py

# Non-root user.
RUN useradd --create-home --uid 10001 procureguard \
    && chown -R procureguard:procureguard /app
USER procureguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uv", "run", "uvicorn", "procureguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
