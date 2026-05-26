#!/usr/bin/env bash
# cloud_setup.sh
# --------------
# Configures a clean Linux instance (Ubuntu 22.04) with A100/L4 GPU
# for running the federated-fhir-architecture experiment matrix.
#
# Usage (on the cloud instance, as root or with sudo):
#   bash cloud_setup.sh
#
# Prerequisites:
#   - Ubuntu 22.04 LTS
#   - NVIDIA GPU (A100, L4, etc.) with driver pre-installed by the provider
#   - Internet access
set -euo pipefail

log() { echo "[SETUP $(date '+%H:%M:%S')] $*"; }

# ── 1. System packages ────────────────────────────────────────────────────────
log "Updating system packages..."
apt-get update -qq
apt-get install -y -qq \
    curl wget git unzip \
    ca-certificates gnupg lsb-release \
    htop nvtop screen tmux

# ── 2. Docker Engine ──────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log "Installing Docker Engine..."
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
    log "Docker installed: $(docker --version)"
else
    log "Docker already installed: $(docker --version)"
fi

# Add current user to docker group (avoids sudo for docker commands)
usermod -aG docker "${SUDO_USER:-$USER}" 2>/dev/null || true

# ── 3. NVIDIA Container Toolkit ───────────────────────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    log "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    log "NVIDIA Container Toolkit installed."
else
    log "NVIDIA Container Toolkit already installed."
fi

# ── 4. uv (Python package manager) ───────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    log "uv installed: $(uv --version)"
else
    log "uv already installed: $(uv --version)"
fi

# ── 5. GPU verification ───────────────────────────────────────────────────────
log "Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L \
    && log "GPU accessible via Docker." \
    || { log "ERROR: GPU not accessible via Docker. Check NVIDIA Container Toolkit."; exit 1; }

# ── 6. Disk ───────────────────────────────────────────────────────────────────
log "Available disk space:"
df -h / | tail -1

log ""
log "════════════════════════════════════════════════════════"
log "Setup complete. Next steps:"
log "  1. Close and reopen the terminal (or: newgrp docker)"
log "  2. Clone the repository:"
log "       git clone <YOUR_REPO_URL>"
log "  3. Follow docs/deploy.md to download MIMIC-IV and run the experiments."
log "════════════════════════════════════════════════════════"
