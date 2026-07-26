#!/usr/bin/env bash
set -euo pipefail
# run_dual_lora.sh <sigma> <seed> [rounds] [tag_suffix]
#
# Artigo 3 — Dual LoRA / HERALD-PFL (FL_LORA_MODE=dual): global adapter r=8
# trained with DP-SGD and transmitted to the server; local adapter r=4
# trained without DP on the silo's own data, persisted to a per-silo
# checkpoint on disk, never transmitted. 1 dual-lora FL server (this
# package's own dual-lora-server, not the Article 2 adaptive-clipping
# wrapper — dual-adapter switching + local-checkpoint persistence needed a
# real client-side extension, see src/article3/dual_training.py) + 5 silos,
# PubMedBERT/ICD-10, q=0.01 subsampling.
#
# [rounds] defaults to 100 (the Article 3 target). [tag_suffix] defaults to
# "" and lets a short validation run (e.g. rounds=20) log to a separate
# directory instead of colliding with the eventual full R=100 matrix run's
# log directory and idempotency check.
#
# Assumes the FHIR R4 server + ETL data load are ALREADY running (via the
# root repo's `make up-infra` / `run_nodocker.sh`), and that `uv sync` has
# already been run once in this package's own directory.

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

TAG="a3_dual_lora_sigma${SIGMA}_seed${SEED}${TAG_SUFFIX}"
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

# A partial/interrupted previous attempt at this exact tag would leave a
# local_adapter.pt checkpoint on disk (see src/article3/dual_training.py) —
# if left in place, a "fresh" restart would silently resume the local
# component from that partial state while the global side and DP epsilon
# accounting restart from round 0, an inconsistent mix. Since this tag is
# NOT yet complete (checked above), any local_adapter.pt found here can only
# be from an incomplete attempt — safe to clear before starting fresh.
for i in 0 1 2 3 4; do
    rm -f "$LOGS_DIR/$TAG/$i/local_adapter.pt"
done

log "[$TAG] starting"
START=$(date +%s)

export FL_LORA_MODE=dual
export ADAPTIVE_EXPERIMENT_TAG="$TAG"
export ADAPTIVE_LOGS_DIR="$LOGS_DIR"
export FL_DP_SUBSAMPLE_RATE=0.01
export FL_NUM_ROUNDS="$ROUNDS"
export FL_NOISE_MULTIPLIER="$SIGMA"
export FL_SEED="$SEED"
export FL_LEARNING_RATE=2e-4
export FL_MIN_CLIENTS=5
export FL_STRATEGY=fedprox
export FL_PROXIMAL_MU=0.01
export FL_MAX_GRAD_NORM=1.0
export FL_TARGET_DELTA=1e-5
export MODEL_BACKEND=bert
export BERT_BENCHMARK=top50
export MAX_SEQ_LEN=512
export FL_BATCH_SIZE=8
export FL_GRADIENT_ACCUM_STEPS=8
export ETL_DIRICHLET_ALPHA=0.5
export FHIR_SERVER_URL="${FHIR_SERVER_URL:-http://localhost:8080/fhir}"
# Distinct port from the production fl_server container (9091) and the
# other Article 3 strategies (dp_lora/ffa_lora use 9096 via the Article 2
# wrapper) — dual_lora runs its own server binary.
export FL_SERVER_ADDRESS="localhost:9097"
export BERT_LABEL_INDEX_PATH="${BERT_LABEL_INDEX_PATH:-$REPO_ROOT/etl_worker/data/label_index.json}"
# Same lock file as dp_lora/ffa_lora — intentional, serializes GPU access
# across ALL Article 3 silos/strategies sharing the one GPU.
export GPU_LOCK_PATH="${GPU_LOCK_PATH:-/tmp/article3_gpu.lock}"
# bitsandbytes/triton JIT-compiles a CUDA util at import time (even for the
# unquantised BERT path) and hangs for ~1h before failing if no C compiler
# is on PATH. gcc is installed as gcc-12 but not aliased to `gcc`.
export CC="${CC:-/usr/bin/gcc-12}"

cd "$EXP_ROOT"
mkdir -p "$LOGS_DIR"

pkill -f "dual-lora-server" 2>/dev/null || true
sleep 1
nohup uv run dual-lora-server > "$LOGS_DIR/${TAG}_server.log" 2>&1 &
SERVER_PID=$!

tries=0
until python3 -c "import socket; s=socket.create_connection(('localhost',9097),2);s.close()" 2>/dev/null; do
    sleep 3
    tries=$((tries + 1))
    if [[ $tries -gt 40 ]]; then
        log "[$TAG] ERROR: dual-lora-server failed to start. Check $LOGS_DIR/${TAG}_server.log"
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
done
log "[$TAG] server ready (pid=$SERVER_PID)"

pids=()
for i in 0 1 2 3 4; do
    # Per-silo path: the final round's global params (final_global_params.npz)
    # land next to that silo's own local_adapter.pt, so eval_cross_silo.sh
    # can reconstruct each silo's trained model from its own directory.
    ETL_PARTITION_ID="$i" FL_SAVE_CHECKPOINT="$LOGS_DIR/$TAG/$i" \
        uv run dual-lora-client > "$LOGS_DIR/${TAG}_silo${i}.log" 2>&1 &
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
