#!/usr/bin/env bash
set -euo pipefail
# run_dp_lora.sh <sigma> <seed> [rounds] [tag_suffix]
#
# Artigo 3 — DP-LoRA baseline (FL_LORA_MODE=standard, both A and B trainable):
# 1 adaptive-clipping FL server + 5 silos, PubMedBERT/ICD-10, q=0.01 subsampling.
#
# Reuses the Article 2 adaptive-clipping-server/-client binaries with
# ADAPTIVE_CLIPPING_STRATEGY=baseline (flat/global clipping, matching this
# experiment's clipping_strategy: global config) — NOT because this is a
# clipping-strategy experiment, but because per-round JSONL logging
# (logs/{tag}/{silo}/round_{r:03d}.jsonl, consumed by the idempotency check
# below and all downstream analysis) only exists in adaptive_clipping's
# logging_utils.py; the plain production fl-server/ai-client entry points
# have no equivalent. The trainable-parameter freeze that matters for this
# experiment (FL_LORA_MODE) happens in ai_client.model_setup_bert.load_bert_model,
# which is shared by both entry points via FHIRFederatedClient._load_model —
# so which entry point is used does not affect FFA-LoRA correctness.
# ADAPTIVE_CLIPPING_STRATEGY=baseline's clip+noise (_clip_and_noise_per_layer)
# only ever receives requires_grad=True parameters (PerLayerClipper.update_history
# filters on requires_grad before grouping), so frozen lora_A parameters are
# never touched.
#
# [rounds] defaults to 100 (the Article 3 target). [tag_suffix] defaults to
# "" and lets a short validation run (e.g. rounds=20) log to a separate
# directory instead of colliding with the eventual full R=100 matrix run's
# log directory and idempotency check.
#
# Assumes the FHIR R4 server + ETL data load are ALREADY running (via the
# root repo's `make up-infra` / `run_nodocker.sh`).

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <sigma> <seed> [rounds] [tag_suffix]" >&2
    exit 1
fi
SIGMA="$1"
SEED="$2"
ROUNDS="${3:-100}"
TAG_SUFFIX="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"
A2_ROOT="$REPO_ROOT/experiments/adaptive-clipping"

TAG="a3_dp_lora_sigma${SIGMA}_seed${SEED}${TAG_SUFFIX}"
LOGS_DIR="$EXP_ROOT/logs"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

ROUNDS_PADDED=$(printf "%03d" "$ROUNDS")
DONE=true
for i in 0 1 2 3 4; do
    [[ -f "$LOGS_DIR/$TAG/$i/round_${ROUNDS_PADDED}.jsonl" ]] || DONE=false
done
if [[ "$DONE" == "true" ]]; then
    log "[$TAG] already complete — skipping."
    exit 0
fi

log "[$TAG] starting"
START=$(date +%s)

export ADAPTIVE_CLIPPING_STRATEGY=baseline
export ADAPTIVE_EXPERIMENT_TAG="$TAG"
export ADAPTIVE_LOGS_DIR="$LOGS_DIR"
export FL_LORA_MODE=standard
export FL_NOISE_MULTIPLIER="$SIGMA"
export FL_SEED="$SEED"
export FL_NUM_ROUNDS="$ROUNDS"
export FL_MIN_CLIENTS=5
export FL_STRATEGY=fedprox
export FL_PROXIMAL_MU=0.01
export FL_MAX_GRAD_NORM=1.0
export FL_LEARNING_RATE=2e-4
export FL_TARGET_DELTA=1e-5
export FL_DP_SUBSAMPLE_RATE=0.01
export MODEL_BACKEND=bert
export BERT_BENCHMARK=top50
export MAX_SEQ_LEN=512
export FL_BATCH_SIZE=8
export FL_GRADIENT_ACCUM_STEPS=8
export ETL_DIRICHLET_ALPHA=0.5
export FHIR_SERVER_URL="${FHIR_SERVER_URL:-http://localhost:8080/fhir}"
# Distinct port from both the host-network production fl_server container
# (9091) and the Article 2 adaptive-clipping experiments (9095), so no
# concurrent run can silently cross-attach.
export FL_SERVER_ADDRESS="localhost:9096"
export BERT_LABEL_INDEX_PATH="${BERT_LABEL_INDEX_PATH:-$REPO_ROOT/etl_worker/data/label_index.json}"
export GPU_LOCK_PATH="${GPU_LOCK_PATH:-/tmp/article3_gpu.lock}"
# bitsandbytes/triton JIT-compiles a CUDA util at import time (even for the
# unquantised BERT path) and hangs for ~1h before failing if no C compiler
# is on PATH. gcc is installed as gcc-12 but not aliased to `gcc`.
export CC="${CC:-/usr/bin/gcc-12}"

# Fase 0 preflight — aborts before any GPU time is spent if the environment
# doesn't match docs/validated_environment.md (dataset size, LR, per-silo counts).
(cd "$REPO_ROOT" && uv run python scripts/preflight_check.py) || { log "[$TAG] preflight FAILED — aborting."; exit 1; }

cd "$A2_ROOT"
mkdir -p "$LOGS_DIR"

pkill -f "adaptive-clipping-server" 2>/dev/null || true
sleep 1
nohup uv run adaptive-clipping-server > "$LOGS_DIR/${TAG}_server.log" 2>&1 &
SERVER_PID=$!

tries=0
until python3 -c "import socket; s=socket.create_connection(('localhost',9096),2);s.close()" 2>/dev/null; do
    sleep 3
    tries=$((tries + 1))
    if [[ $tries -gt 40 ]]; then
        log "[$TAG] ERROR: adaptive-clipping-server failed to start. Check $LOGS_DIR/${TAG}_server.log"
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
done
log "[$TAG] server ready (pid=$SERVER_PID)"

pids=()
for i in 0 1 2 3 4; do
    ETL_PARTITION_ID="$i" uv run adaptive-clipping-client > "$LOGS_DIR/${TAG}_silo${i}.log" 2>&1 &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

END=$(date +%s)
log "[$TAG] finished — duration $((END - START))s — ${failed} silo(s) failed"
[[ $failed -eq 0 ]]
