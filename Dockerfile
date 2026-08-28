FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# uv itself, not a pip install of it — the image ships pinned to a known uv
# release rather than whatever pip resolves that day.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# --- dependency layer --------------------------------------------------------
# Locked, and separated from the application layer so an app-only change
# doesn't invalidate Docker's dependency cache. --no-install-project: the
# venv gets every dependency but not vault_ask itself yet — that copy hasn't
# happened, and installing it here would just be re-done by the next RUN.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# --- application layer --------------------------------------------------------
# config.yaml baked in as the deployment default (README "Configuration") —
# env vars still override it (VAULTASK_<SECTION>__<KEY>), same as local dev.
COPY vault_ask ./vault_ask
COPY config.yaml ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# Non-root, and the rootfs is read-only at runtime (see docker-compose.nas.yml)
# — the only thing this writes is /data, which is bind-mounted.
RUN groupadd -g 1000 appuser && useradd -g appuser -u 1000 appuser \
    && mkdir -p /data && chown -R appuser:appuser /data
USER appuser

EXPOSE 8080

# Unauthenticated by design (README "How to test it") — fails 503 when the
# SQLite index itself can't be queried (see api/app.py::healthz).
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=8).status==200 else 1)"

CMD ["python", "-m", "vault_ask", "serve"]
