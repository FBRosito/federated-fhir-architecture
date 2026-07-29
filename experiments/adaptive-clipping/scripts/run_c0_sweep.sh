#!/usr/bin/env bash
set -euo pipefail
# run_c0_sweep.sh <c0> <seed>
#
# Flat/global-clip C0 sweep at fixed sigma=1.0: 1 adaptive-clipping FL server
# + 5 silos, PubMedBERT/ICD-10, 20 rounds. Reuses the exact same
# ADAPTIVE_CLIPPING_STRATEGY=baseline code path as run_baseline.sh — the
# only thing that varies here is FL_MAX_GRAD_NORM (C0); sigma is pinned to
# 1.0 across every point in the sweep so epsilon is identical at every C0
# (see RDPAccountant verification: epsilon depends only on noise_multiplier,
# not on C0 — confirmed at 19.0163 for all 4 sweep points). The sweep exists
# to show that no single global C0 resolves the ~20-500x scale gap between
# encoder gradients (0.002-0.054) and the classifier head (~0.97): a C0
# small enough for the encoder crushes the head, and a C0 large enough for
# the head leaves the encoder drowned in noise, regardless of sigma.
#
# Structurally identical to run_baseline.sh otherwise — see that script for
# the infra-prerequisite note (FHIR server + ETL load must already be running).

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <c0> <seed>" >&2
    exit 1
fi
C0="$1"
SEED="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"

TAG="c0_sweep_c0${C0}_seed${SEED}"
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

export ADAPTIVE_CLIPPING_STRATEGY=baseline
export ADAPTIVE_EXPERIMENT_TAG="$TAG"
export ADAPTIVE_LOGS_DIR="$LOGS_DIR"
export FL_NOISE_MULTIPLIER=1.0
export FL_SEED="$SEED"
export FL_NUM_ROUNDS=20
export FL_MIN_CLIENTS=5
export FL_STRATEGY=fedprox
export FL_PROXIMAL_MU=0.01
export FL_MAX_GRAD_NORM="$C0"
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

# Fase 0 preflight — aborts before any GPU time is spent if the environment
# doesn't match docs/validated_environment.md (dataset size, LR, per-silo counts).
(cd "$REPO_ROOT" && uv run python scripts/preflight_check.py) || { log "[$TAG] preflight FAILED — aborting."; exit 1; }

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
