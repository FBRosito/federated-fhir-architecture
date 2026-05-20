# ── ai_client/Dockerfile ──────────────────────────────────────────────────────
# Cliente Flower com suporte a GPU NVIDIA (CUDA 12.4 + cuDNN 9).
# Requer NVIDIA Container Toolkit instalado no host.
# Build context: raiz do workspace (docker-compose usa context: .)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── Sistema base ──────────────────────────────────────────────────────────────
# Ubuntu 22.04 não distribui Python 3.13 via apt; o uv instala o interpretador
# standalone a partir dos builds gerenciados pela Astral (sem PPA necessário).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv gerencia o Python 3.13 standalone — não depende de python3 do sistema
ENV UV_PYTHON_PREFERENCE=managed

WORKDIR /app

# ── Camada de dependências (cache-friendly) ───────────────────────────────────
COPY pyproject.toml uv.lock .python-version ./
COPY etl_worker/pyproject.toml  etl_worker/
COPY ai_client/pyproject.toml   ai_client/
COPY fl_server/pyproject.toml   fl_server/
COPY evaluation/pyproject.toml  evaluation/

# Em Linux, o uv.lock já aponta para torch+cu124 (via [tool.uv.sources])
RUN uv sync --frozen --no-install-project --no-dev

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY ai_client/ ai_client/
COPY evaluation/ evaluation/

RUN uv sync --frozen --package ai-client --package evaluation --no-dev

# Variáveis de ambiente NVIDIA exigidas pelo Container Toolkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/app/.venv/bin/ai-client"]
