# ── ai_client/Dockerfile ──────────────────────────────────────────────────────
# Cliente Flower com suporte a GPU NVIDIA (CUDA 12.4 + cuDNN 9).
# Requer NVIDIA Container Toolkit instalado no host.
# Build context: raiz do workspace (docker-compose usa context: .)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

# ── Sistema base ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3-pip \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instala uv (gerencia o venv e as deps sem depender do pip do sistema)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Garante que uv use o Python 3.13 do sistema
ENV UV_PYTHON=/usr/bin/python3.13

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

RUN uv sync --frozen --package ai-client --no-dev

# Variáveis de ambiente NVIDIA exigidas pelo Container Toolkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uv", "run", "ai-client"]
