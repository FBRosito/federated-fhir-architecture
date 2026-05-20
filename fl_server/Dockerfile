# ── fl_server/Dockerfile ──────────────────────────────────────────────────────
# Orquestrador Flower. Não precisa de GPU — usa python:slim.
# Build context: raiz do workspace (docker-compose usa context: .)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim

# Instala uv a partir da imagem oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── Camada de dependências (cache-friendly) ───────────────────────────────────
# Copia apenas os arquivos de manifesto do workspace para resolver o lock
COPY pyproject.toml uv.lock .python-version ./
COPY etl_worker/pyproject.toml  etl_worker/
COPY ai_client/pyproject.toml   ai_client/
COPY fl_server/pyproject.toml   fl_server/
COPY evaluation/pyproject.toml  evaluation/

# Instala todas as dependências declaradas no workspace (sem instalar src ainda)
RUN uv sync --frozen --no-install-project --no-dev

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY fl_server/ fl_server/

# Instala o pacote fl-server no ambiente já resolvido
RUN uv sync --frozen --package fl-server --no-dev

EXPOSE 9091 9092

CMD ["/app/.venv/bin/fl-server"]
