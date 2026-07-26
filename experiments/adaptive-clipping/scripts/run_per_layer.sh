#!/usr/bin/env bash
set -euo pipefail
# run_per_layer.sh <sigma> <seed>
#
# Per-attention-layer adaptive clipping (ADAPTIVE_CLIPPING_STRATEGY=per_layer):
# 1 adaptive-clipping FL server + 5 silos, PubMedBERT/ICD-10, 20 rounds.
# Structurally identical to run_baseline.sh — see that script for the
# infra-prerequisite note (FHIR server + ETL load must already be running).

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <sigma> <seed>" >&2
    exit 1
fi
SIGMA="$1"
SEED="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"

TAG="per_layer_sigma${SIGMA}_seed${SEED}"
LOGS_DIR="$EXP_ROOT/logs"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

DONE=true
for i in 0 1 2 3 4; do
    [[ -f "$LOGS_DIR/$TAG/$i/round_020.jsonl" ]] || DONE=false
done
if [[ "$DONE" == "true" ]]; then
    log "[$TAG] already complete — skipping."
    exit 0
fi

log "[$TAG] starting"
START=$(date +%s)

export ADAPTIVE_CLIPPING_STRATEGY=per_layer
export ADAPTIVE_WARMUP_ROUNDS=5
export ADAPTIVE_PERCENTILE=75.0
export ADAPTIVE_MIN_CLIP=0.001
export ADAPTIVE_MAX_CLIP=2.0
export ADAPTIVE_EXPERIMENT_TAG="$TAG"
export ADAPTIVE_LOGS_DIR="$LOGS_DIR"
export FL_NOISE_MULTIPLIER="$SIGMA"
export FL_SEED="$SEED"
export FL_NUM_ROUNDS=20
export FL_MIN_CLIENTS=5
export FL_STRATEGY=fedprox
export FL_PROXIMAL_MU=0.01
export FL_MAX_GRAD_NORM=1.0
export FL_LEARNING_RATE=2e-4
export FL_TARGET_DELTA=1e-5
export FL_DP_SUBSAMPLE_RATE=0.1
export MODEL_BACKEND=bert
export BERT_BENCHMARK=top50
export MAX_SEQ_LEN=512
export FL_BATCH_SIZE=8
export FL_GRADIENT_ACCUM_STEPS=8
export ETL_DIRICHLET_ALPHA=0.5
export FHIR_SERVER_URL="${FHIR_SERVER_URL:-http://localhost:8080/fhir}"
# Port 9091 is squatted by the host-network `fl_server` Docker container —
# using a distinct port avoids clients silently attaching to that container
# instead of this experiment's own server.
export FL_SERVER_ADDRESS="localhost:9095"
export BERT_LABEL_INDEX_PATH="${BERT_LABEL_INDEX_PATH:-$REPO_ROOT/etl_worker/data/label_index.json}"
export GPU_LOCK_PATH="${GPU_LOCK_PATH:-/tmp/adaptive_clipping_gpu.lock}"
# bitsandbytes/triton JIT-compiles a CUDA util at import time (even for the
# unquantised BERT path) and hangs for ~1h before failing if no C compiler
# is on PATH. gcc is installed as gcc-12 but not aliased to `gcc`.
export CC="${CC:-/usr/bin/gcc-12}"

cd "$EXP_ROOT"
mkdir -p "$LOGS_DIR"

pkill -f "adaptive-clipping-server" 2>/dev/null || true
sleep 1
nohup uv run adaptive-clipping-server > "$LOGS_DIR/${TAG}_server.log" 2>&1 &
SERVER_PID=$!

tries=0
until python3 -c "import socket; s=socket.create_connection(('localhost',9095),2);s.close()" 2>/dev/null; do
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
