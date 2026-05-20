#!/usr/bin/env bash
# cloud_setup.sh
# --------------
# Configura uma instância Linux limpa (Ubuntu 22.04) com GPU A100/L4
# para rodar a matriz experimental federated-fhir-architecture.
#
# Uso (na instância da nuvem, como root ou com sudo):
#   bash cloud_setup.sh
#
# Pré-requisitos:
#   - Ubuntu 22.04 LTS
#   - GPU NVIDIA (A100, L4, etc.) com driver já instalado pelo provedor
#   - Acesso à internet
set -euo pipefail

log() { echo "[SETUP $(date '+%H:%M:%S')] $*"; }

# ── 1. Pacotes do sistema ─────────────────────────────────────────────────────
log "Atualizando pacotes do sistema..."
apt-get update -qq
apt-get install -y -qq \
    curl wget git unzip \
    ca-certificates gnupg lsb-release \
    htop nvtop screen tmux

# ── 2. Docker Engine ──────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log "Instalando Docker Engine..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    log "Docker instalado: $(docker --version)"
else
    log "Docker já instalado: $(docker --version)"
fi

# Adiciona o usuário atual ao grupo docker (evita sudo)
usermod -aG docker "${SUDO_USER:-$USER}" 2>/dev/null || true

# ── 3. NVIDIA Container Toolkit ───────────────────────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    log "Instalando NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    log "NVIDIA Container Toolkit instalado."
else
    log "NVIDIA Container Toolkit já instalado."
fi

# ── 4. uv (gerenciador de pacotes Python) ─────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    log "uv instalado: $(uv --version)"
else
    log "uv já instalado: $(uv --version)"
fi

# ── 5. Verificação GPU ────────────────────────────────────────────────────────
log "Verificando GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L \
    && log "GPU acessível via Docker." \
    || { log "ERRO: GPU não acessível via Docker. Verifique o NVIDIA Container Toolkit."; exit 1; }

# ── 6. Disco ──────────────────────────────────────────────────────────────────
log "Espaço em disco disponível:"
df -h / | tail -1

log ""
log "════════════════════════════════════════════════════════"
log "Setup concluído. Próximos passos:"
log "  1. Feche e reabra o terminal (ou: newgrp docker)"
log "  2. Clone o repositório:"
log "       git clone <URL_DO_SEU_REPO>"
log "  3. Siga o DEPLOY.md para baixar o MIMIC-IV e executar os experimentos."
log "════════════════════════════════════════════════════════"
