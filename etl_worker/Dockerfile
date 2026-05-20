# ── etl_worker/Dockerfile ─────────────────────────────────────────────────────
# Worker de ETL sobre dados FHIR. Sem GPU — usa python:slim.
# Build context: raiz do workspace (docker-compose usa context: .)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim

# Instala uv a partir da imagem oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# curl é necessário para o healthcheck do hapi_fhir (opcional aqui, mas útil para debug)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Camada de dependências (cache-friendly) ───────────────────────────────────
COPY pyproject.toml uv.lock .python-version ./
COPY etl_worker/pyproject.toml  etl_worker/
COPY ai_client/pyproject.toml   ai_client/
COPY fl_server/pyproject.toml   fl_server/
COPY evaluation/pyproject.toml  evaluation/

RUN uv sync --frozen --no-install-project --no-dev

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY etl_worker/ etl_worker/

RUN uv sync --frozen --package etl-worker --no-dev

CMD ["/app/.venv/bin/etl-worker"]
