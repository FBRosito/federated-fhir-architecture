#!/usr/bin/env bash
# run_nodocker.sh — Experiment matrix identical to run_experiments.sh but without Docker.
# Requires: Java ≥ 17, uv, CUDA-enabled GPU, HAPI FHIR CLI jar (auto-downloaded).
#
# Usage: bash run_nodocker.sh [--smoke] [--mini] [--exp {A|B|all}]
#   --smoke   local validation: 20 examples, 2 rounds, 1 seed (~2-3h)
#   --mini    statistical: 20 examples, 2 rounds, 3 seeds — produces real p-values (~6-7h)
#   --exp A   only Experiment A (ICD-10 coding, PubMedBERT)
#   --exp B   only Experiment B (discharge summary, Llama-3.2)
#   --exp all both (default)
#
# Environment variables for A100 40/80 GB:
#   export FL_BATCH_SIZE=8 FL_GRADIENT_ACCUM_STEPS=8 FL_PARALLEL_GPU=true

set -euo pipefail

# Load .env (HF_TOKEN etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ── Flags ─────────────────────────────────────────────────────────────────────
SMOKE=false
MINI=false
EXP="all"
OVERRIDE_MAX_EXAMPLES=""
ALPHA="0.5"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)        SMOKE=true ;;
        --mini)         MINI=true ;;
        --exp)          shift; EXP="${1:-all}" ;;
        --max-examples) shift; OVERRIDE_MAX_EXAMPLES="${1:-}" ;;
        --alpha)        shift; ALPHA="${1:-0.5}" ;;
    esac
    shift
done
export ETL_DIRICHLET_ALPHA="$ALPHA"

LOGS="experiment_logs"
mkdir -p "$LOGS"

RUN_MODE=$([ "$MINI" = "true" ] && echo "mini" || ([ "$SMOKE" = "true" ] && echo "smoke" || echo "full"))
MASTER_LOG="$LOGS/run_$(date -u '+%Y%m%d_%H%M%S')_${RUN_MODE}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Master log: $MASTER_LOG"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# ── Parameters ────────────────────────────────────────────────────────────────
if $SMOKE && ! $MINI; then
    MAX_EXAMPLES=20; CENTRAL_MAX_EXAMPLES=20; FL_ROUNDS=2; CENTRAL_EPOCHS=2; SEEDS=(42)
    [[ -n "$OVERRIDE_MAX_EXAMPLES" ]] && MAX_EXAMPLES="$OVERRIDE_MAX_EXAMPLES" && CENTRAL_MAX_EXAMPLES="$OVERRIDE_MAX_EXAMPLES"
    log "SMOKE MODE: examples=$MAX_EXAMPLES rounds=$FL_ROUNDS seeds=${SEEDS[*]}"
elif $MINI; then
    MAX_EXAMPLES=20; CENTRAL_MAX_EXAMPLES=20; FL_ROUNDS=2; CENTRAL_EPOCHS=2; SEEDS=(42 43 44)
    log "MINI MODE: examples=$MAX_EXAMPLES rounds=$FL_ROUNDS seeds=${SEEDS[*]}"
else
    MAX_EXAMPLES=0; CENTRAL_MAX_EXAMPLES="${CENTRAL_MAX_EXAMPLES:-10000}"; FL_ROUNDS=10; CENTRAL_EPOCHS=10
    SEEDS=(42 43 44)
    log "FULL MODE: rounds=$FL_ROUNDS central_max=$CENTRAL_MAX_EXAMPLES seeds=${SEEDS[*]}"
fi

# ── Hardware ──────────────────────────────────────────────────────────────────
LLM_BATCH_SIZE="${FL_BATCH_SIZE:-1}"
LLM_GRADIENT_ACCUM="${FL_GRADIENT_ACCUM_STEPS:-64}"
BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-8}"
BERT_GRADIENT_ACCUM="${BERT_GRADIENT_ACCUM:-8}"
export FL_PARALLEL_GPU="${FL_PARALLEL_GPU:-false}"
log "Hardware: LLM batch=$LLM_BATCH_SIZE accum=$LLM_GRADIENT_ACCUM | BERT batch=$BERT_BATCH_SIZE accum=$BERT_GRADIENT_ACCUM | parallel_gpu=$FL_PARALLEL_GPU"

# ── Paths ─────────────────────────────────────────────────────────────────────
export MIMIC_HOSP_DIR="$(pwd)/physionet.org/files/mimiciv/3.1/hosp"
export MIMIC_NOTE_DIR="$(pwd)/physionet.org/files/mimic-iv-note/2.2/note"
export BERT_LABEL_INDEX_PATH="${BERT_LABEL_INDEX_PATH:-$(pwd)/etl_worker/data/label_index.json}"
export GPU_LOCK_PATH="${GPU_LOCK_PATH:-/tmp/fl_gpu.lock}"

FHIR_SERVER_SCRIPT="fhir_server.py"   # minimal Python FHIR server (stdlib only)

# ── FHIR server (Python, no Docker / no Java required) ────────────────────────

start_hapi_fhir() {
    # Kill any stale process on port 8080 before starting fresh
    pkill -f "$FHIR_SERVER_SCRIPT" 2>/dev/null || true
    fuser -k 8080/tcp 2>/dev/null || true
    sleep 1
    local py
    py="${VIRTUAL_ENV:-.venv}/bin/python"
    [ -x "$py" ] || py="$(command -v python3)"
    nohup "$py" "$FHIR_SERVER_SCRIPT" 8080 \
        > "$LOGS/hapi_fhir.log" 2>&1 &
    echo $! > /tmp/hapi_fhir.pid
    log "FHIR server starting (PID=$(cat /tmp/hapi_fhir.pid))..."
}

wait_hapi_ready() {
    log "Waiting for FHIR server at http://localhost:8080/fhir/metadata..."
    local tries=0
    until curl -sf http://localhost:8080/fhir/metadata > /dev/null 2>&1; do
        printf '.'; sleep 2
        tries=$((tries+1))
        if [ $tries -gt 15 ]; then
            log "ERROR: FHIR server failed to start after 30s. Check $LOGS/hapi_fhir.log"
            exit 1
        fi
    done
    echo " FHIR server ready."
}

# ── FL Server ─────────────────────────────────────────────────────────────────

start_fl_server() {
    local min_clients="$1" noise="$2" rounds="$3" strategy="${4:-fedprox}" mu="${5:-0.01}" lr="${6:-5e-5}"
    export FL_MIN_CLIENTS="$min_clients" FL_NUM_ROUNDS="$rounds" FL_NOISE_MULTIPLIER="$noise"
    export FL_STRATEGY="$strategy" FL_LEARNING_RATE="$lr" FL_NUM_EPOCHS=1 FL_PROXIMAL_MU="$mu"
    pkill -f "fl-server" 2>/dev/null || true
    sleep 2
    : > "$LOGS/fl_server_current.log"
    nohup uv run fl-server > "$LOGS/fl_server_current.log" 2>&1 &
    echo $! > /tmp/fl_server.pid
    wait_fl_healthy
}

wait_fl_healthy() {
    log "Waiting for fl_server on port 9091..."
    local tries=0
    until python3 -c "import socket; s=socket.create_connection(('localhost',9091),2);s.close()" 2>/dev/null; do
        printf '.'; sleep 3
        tries=$((tries+1))
        if [ $tries -gt 40 ]; then
            log "ERROR: fl_server failed to start. Check $LOGS/fl_server_current.log"
            exit 1
        fi
    done
    echo " fl_server ready."
}

# ── JSON serialisers (verbatim from run_experiments.sh) ───────────────────────

# save_run_json <tag> <backend> <strategy> <noise> <n_silos> <seed> <rounds> <mu>
save_run_json() {
    local tag="$1" backend="$2" strategy="$3" noise="$4"
    local n_silos="${5:-5}" seed="${6:-42}" rounds="${7:-5}" mu="${8:-0.01}"
    local server_log="$LOGS/${tag}_server.log"
    local json_out="$LOGS/${tag}.json"
    local c0="${FL_MAX_GRAD_NORM:-1.0}"
    [ -f "$server_log" ] || { log "WARN: $server_log not found — skipping JSON"; return; }
    python3 - "$server_log" "$json_out" "$tag" "$backend" "$strategy" \
             "$noise" "$n_silos" "$seed" "$rounds" "$mu" "$c0" <<'PYEOF'
import sys, json, re, ast, datetime, math
srv, out, tag, backend, strategy, noise, n_silos, seed, rounds, mu, c0 = sys.argv[1:]

eval_pat = re.compile(r"Aggregated eval metrics \(\d+ clients\): ({.*?})\s*$")
fit_pat  = re.compile(r"Aggregated fit metrics \(\d+ clients\): ({.*?})\s*$")

def safe_parse(s):
    try:
        raw = ast.literal_eval(s)
        return {k: float(v) for k, v in raw.items()
                if not (isinstance(v, float) and math.isnan(v))}
    except Exception:
        return {}

eval_rounds: list[dict] = []
fit_rounds:  list[dict] = []

with open(srv) as f:
    for line in f:
        m = eval_pat.search(line)
        if m:
            eval_rounds.append(safe_parse(m.group(1)))
            continue
        m = fit_pat.search(line)
        if m:
            fit_rounds.append(safe_parse(m.group(1)))

FIT_KEYS_TO_MERGE = {"epsilon_spent", "epsilon_cumulative", "train_loss",
                     "lora_b_norm_end", "lora_b_drift_ratio", "proximal_mu"}
rounds_data = []
for i, eval_m in enumerate(eval_rounds):
    merged = dict(eval_m)
    if i < len(fit_rounds):
        for k in FIT_KEYS_TO_MERGE:
            if k in fit_rounds[i] and k not in merged:
                merged[k] = fit_rounds[i][k]
    rounds_data.append(merged)

result = {
    "tag": tag,
    "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    "config": {
        "backend": backend, "strategy": strategy,
        "noise_multiplier": float(noise), "n_silos": int(n_silos),
        "seed": int(seed), "num_rounds": int(rounds), "proximal_mu": float(mu),
        "max_grad_norm": float(c0),
    },
    "per_round_eval": rounds_data,
    "final_metrics": rounds_data[-1] if rounds_data else {},
    "n_rounds_completed": len(rounds_data),
}
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"[JSON] {out}  (rounds={len(rounds_data)}, "
      f"has_epsilon={'epsilon_spent' in (rounds_data[-1] if rounds_data else {})})")
PYEOF
}

# save_centralizado_json <tag> <backend> <seed> <epochs>
save_centralizado_json() {
    local tag="$1" backend="$2" seed="${3:-42}" epochs="${4:-5}"
    local client_log="$LOGS/${tag}.log"
    local json_out="$LOGS/${tag}.json"
    [ -f "$client_log" ] || { log "WARN: $client_log not found — skipping JSON"; return; }
    python3 - "$client_log" "$json_out" "$tag" "$backend" "$seed" "$epochs" <<'PYEOF'
import sys, json, re, datetime, math
clog, out, tag, backend, seed, epochs = sys.argv[1:]

def safe_float(s):
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (ValueError, TypeError):
        return None

rounds_data = []
final_metrics = {}

with open(clog) as f:
    content = f.read()

if backend == "llm":
    for m in re.finditer(r"Epoch (\d+)/\d+ completed.*?loss=([\d.nan]+).*?ppl=([\d.nan]+)", content, re.IGNORECASE):
        rounds_data.append({"epoch": int(m.group(1)),
                            "train_loss": safe_float(m.group(2)),
                            "train_perplexity": safe_float(m.group(3))})
    m = re.search(r"Final evaluation:.*?loss=([\d.nan]+).*?ppl=([\d.nan]+)", content, re.IGNORECASE)
    if m:
        final_metrics = {"eval_loss": safe_float(m.group(1)),
                         "eval_perplexity": safe_float(m.group(2))}
else:
    for m in re.finditer(r"BERT Epoch (\d+)/\d+ completed.*?loss=([\d.nan]+)", content, re.IGNORECASE):
        rounds_data.append({"epoch": int(m.group(1)), "train_loss": safe_float(m.group(2))})
    idx = content.rfind("BERT evaluation:")
    if idx >= 0:
        block = content[idx:idx+600]
        def g(pat): m = re.search(pat, block, re.IGNORECASE); return safe_float(m.group(1)) if m else None
        final_metrics = {
            "eval_loss":     g(r"BERT evaluation: loss=([\d.nan]+)"),
            "micro_f1":      g(r"Micro-F1:\s*([\d.nan]+)"),
            "macro_f1":      g(r"Macro-F1:\s*([\d.nan]+)"),
            "auc_roc_micro": g(r"AUC-ROC micro:\s*([\d.nan]+)"),
            "auc_roc_macro": g(r"AUC-ROC macro:\s*([\d.nan]+)"),
            "P@8":  g(r"P@8=([\d.nan]+)"),  "R@8":  g(r"R@8=([\d.nan]+)"),  "F1@8":  g(r"F1@8=([\d.nan]+)"),
            "P@15": g(r"P@15=([\d.nan]+)"), "R@15": g(r"R@15=([\d.nan]+)"), "F1@15": g(r"F1@15=([\d.nan]+)"),
        }

result = {
    "tag": tag,
    "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    "config": {
        "backend": backend, "strategy": "centralised",
        "noise_multiplier": 0.0, "n_silos": 1,
        "seed": int(seed), "num_rounds": int(epochs), "proximal_mu": 0.0,
    },
    "per_round_eval": rounds_data,
    "final_metrics": final_metrics,
    "n_rounds_completed": len(rounds_data),
}
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"[JSON] {out}")
PYEOF
}

read_calib_C0() {
    local backend="$1"
    local calib_log="$LOGS/calibration_${backend}.log"
    local recommended
    recommended=$(grep "RECOMMENDATION: use max_grad_norm=" "$calib_log" 2>/dev/null \
                  | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [[ -n "$recommended" ]] && python3 -c "v=float('$recommended'); assert v>0" 2>/dev/null; then
        echo "$recommended"
    else
        echo "1.0"
    fi
}

# ── Experiment runners ────────────────────────────────────────────────────────

_run_silo() {
    local part="$1" tag="$2"
    ETL_PARTITION_ID="$part" \
    FL_SERVER_ADDRESS="localhost:9091" \
    FHIR_SERVER_URL="http://localhost:8080/fhir" \
    NVIDIA_VISIBLE_DEVICES=all \
    GPU_LOCK_PATH="$GPU_LOCK_PATH" \
    MIMIC_HOSP_DIR="$MIMIC_HOSP_DIR" \
    MIMIC_NOTE_DIR="$MIMIC_NOTE_DIR" \
    BERT_LABEL_INDEX_PATH="$BERT_LABEL_INDEX_PATH" \
    FL_LORA_COMPRESS=false \
    FL_COMPRESS_BITS=4 \
    MODEL_BASE_PRECISION="${MODEL_BASE_PRECISION:-nf4}" \
    uv run ai-client > "$LOGS/${tag}_silo${part}.log" 2>&1
}

# run_fl <tag> <n_silos> <strategy> <noise> <rounds> <backend> <seed> [proximal_mu]
run_fl() {
    local tag="$1" n_silos="$2" strategy="$3" noise="$4" rounds="$5"
    local backend="${6:-llm}" seed="${7:-42}" mu="${8:-0.01}"

    log "FL run: tag=$tag backend=$backend silos=$n_silos strategy=$strategy noise=$noise rounds=$rounds seed=$seed mu=$mu"

    local fl_lr="5e-5"; [ "$backend" = "bert" ] && fl_lr="2e-4"
    start_fl_server "$n_silos" "$noise" "$rounds" "$strategy" "$mu" "$fl_lr"

    if [ "$backend" = "bert" ]; then
        export FL_BATCH_SIZE="$BERT_BATCH_SIZE" FL_GRADIENT_ACCUM_STEPS="$BERT_GRADIENT_ACCUM"
        export MAX_SEQ_LEN=512 BERT_BENCHMARK=top50
    else
        export FL_BATCH_SIZE="$LLM_BATCH_SIZE" FL_GRADIENT_ACCUM_STEPS="$LLM_GRADIENT_ACCUM"
        export MAX_SEQ_LEN=1024 BERT_BENCHMARK=full
    fi
    export FL_MAX_EXAMPLES="$MAX_EXAMPLES" FL_EVAL_ACCURACY=true FL_TOP_K=5
    export FL_NOISE_MULTIPLIER="$noise" FL_TARGET_DELTA=1e-5
    export MODEL_BACKEND="$backend" FL_SEED="$seed"
    if [ "$backend" = "bert" ]; then
        export FL_MAX_GRAD_NORM="${FL_MAX_GRAD_NORM_BERT:-1.0}"
    else
        export FL_MAX_GRAD_NORM="${FL_MAX_GRAD_NORM_LLM:-1.0}"
    fi
    log "  C₀=$FL_MAX_GRAD_NORM (FL_MAX_GRAD_NORM)"

    # Launch all silos in background
    local pids=()
    for i in $(seq 0 $((n_silos - 1))); do
        _run_silo "$i" "$tag" &
        pids+=($!)
    done

    # Wait for silos to finish
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$((failed+1))
    done
    [ $failed -gt 0 ] && log "WARNING: $failed silo(s) exited with non-zero status"

    # Merge silo logs, capture server log
    cat "$LOGS/${tag}_silo"*.log > "$LOGS/${tag}_clients.log" 2>/dev/null || true
    sleep 5
    cp "$LOGS/fl_server_current.log" "$LOGS/${tag}_server.log" 2>/dev/null || true

    save_run_json "$tag" "$backend" "$strategy" "$noise" "$n_silos" "$seed" "$rounds" "$mu"
    log "Run $tag completed."
}

# run_centralizado <seed> <backend>
run_centralizado() {
    local seed="$1" backend="${2:-llm}"
    local tag="centralizado_${backend}_seed${seed}"
    log ">>> $tag"

    local seq_len lr bs accum bert_bm
    if [ "$backend" = "bert" ]; then
        seq_len=512; lr="2e-4"; bs="$BERT_BATCH_SIZE"; accum="$BERT_GRADIENT_ACCUM"; bert_bm="top50"
    else
        seq_len=1024; lr="5e-5"; bs="$LLM_BATCH_SIZE"; accum="$LLM_GRADIENT_ACCUM"; bert_bm="full"
    fi

    FL_MAX_EXAMPLES="$CENTRAL_MAX_EXAMPLES" FL_EVAL_ACCURACY=true FL_TOP_K=5 \
    FL_NUM_ROUNDS="$CENTRAL_EPOCHS" MAX_SEQ_LEN="$seq_len" \
    FL_BATCH_SIZE="$bs" FL_GRADIENT_ACCUM_STEPS="$accum" \
    FL_LEARNING_RATE="$lr" FL_SEED="$seed" MODEL_BACKEND="$backend" \
    BERT_BENCHMARK="$bert_bm" ETL_PARTITION_ID=0 \
    FL_SERVER_ADDRESS=localhost:9091 FHIR_SERVER_URL=http://localhost:8080/fhir \
    NVIDIA_VISIBLE_DEVICES=all GPU_LOCK_PATH="$GPU_LOCK_PATH" \
    MIMIC_HOSP_DIR="$MIMIC_HOSP_DIR" MIMIC_NOTE_DIR="$MIMIC_NOTE_DIR" \
    BERT_LABEL_INDEX_PATH="$BERT_LABEL_INDEX_PATH" \
    FL_LORA_COMPRESS=false MODEL_BASE_PRECISION="${MODEL_BASE_PRECISION:-nf4}" \
    uv run python -m ai_client.centralized_baseline \
        2>&1 | tee "$LOGS/${tag}.log"

    save_centralizado_json "$tag" "$backend" "$seed" "$CENTRAL_EPOCHS"
    log "$tag completed."
}

# run_calibration <backend>
run_calibration() {
    local backend="${1:-llm}"
    local tag="calibration_${backend}"
    log ">>> GRAD NORM CALIBRATION: backend=$backend"

    local calib_seq_len=512; [ "$backend" = "llm" ] && calib_seq_len=1024
    local bert_bm="full"; [ "$backend" = "bert" ] && bert_bm="top50"
    local calib_lr="5e-5"; [ "$backend" = "bert" ] && calib_lr="2e-4"
    start_fl_server 1 0.0 1 fedprox 0.01 "$calib_lr"

    FL_MAX_EXAMPLES=50 FL_CALIBRATE_GRAD_NORM=true FL_NOISE_MULTIPLIER=0.0 \
    MODEL_BACKEND="$backend" BERT_BENCHMARK="$bert_bm" \
    MAX_SEQ_LEN="$calib_seq_len" FL_BATCH_SIZE=1 FL_GRADIENT_ACCUM_STEPS=1 \
    FL_CALIB_OUTPUT="/tmp/grad_norm_${backend}.json" \
    ETL_PARTITION_ID=0 FL_SERVER_ADDRESS=localhost:9091 \
    FHIR_SERVER_URL=http://localhost:8080/fhir \
    NVIDIA_VISIBLE_DEVICES=all GPU_LOCK_PATH="$GPU_LOCK_PATH" \
    FL_LORA_COMPRESS=false MODEL_BASE_PRECISION="${MODEL_BASE_PRECISION:-nf4}" \
    uv run ai-client 2>&1 | tee "$LOGS/${tag}.log"

    log "Calibration $backend done. Check /tmp/grad_norm_${backend}.json"
    unset FL_CALIBRATE_GRAD_NORM 2>/dev/null || true
}

# run_experiment_matrix <backend> <exp_label>
run_experiment_matrix() {
    local backend="$1" exp_label="$2"
    log "════ $exp_label (backend=$backend) ════"

    log "CONFIG 1/6 — CENTRALISED"
    for seed in "${SEEDS[@]}"; do run_centralizado "$seed" "$backend"; done

    log "CONFIG 2/6 — FL FedProx α=${ALPHA} σ=0 (FL baseline)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_nodp_${backend}_seed${seed}" \
               5 fedprox 0.0 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    log "CONFIG 3/6 — FL FedAvg α=${ALPHA} σ=0 (FedProx vs FedAvg)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedavg_alpha${ALPHA}_nodp_${backend}_seed${seed}" \
               5 fedavg 0.0 "$FL_ROUNDS" "$backend" "$seed" 0.0
    done

    log "CONFIG 4/6 — FL FedProx α=${ALPHA} σ=0.5 (light DP)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_dp0.5_${backend}_seed${seed}" \
               5 fedprox 0.5 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    log "CONFIG 5/6 — FL FedProx α=${ALPHA} σ=1.0 (moderate DP)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_dp1.0_${backend}_seed${seed}" \
               5 fedprox 1.0 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    log "CONFIG 6/6 — FL FedProx α=${ALPHA} σ=2.0 (strong DP)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_dp2.0_${backend}_seed${seed}" \
               5 fedprox 2.0 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    log "=== $exp_label COMPLETED ==="
}

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT — start HAPI FHIR and ensure data is loaded
# ═══════════════════════════════════════════════════════════════════════════════
log "Starting HAPI FHIR..."
start_hapi_fhir
wait_hapi_ready

FHIR_PATIENT_COUNT=$(curl -sf "http://localhost:8080/fhir/Patient?_count=1&_summary=count" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo "0")

if [ "${FHIR_PATIENT_COUNT:-0}" -lt 1000 ] 2>/dev/null; then
    log "FHIR has $FHIR_PATIENT_COUNT patients — running ETL worker to load bundles..."
    FHIR_SERVER_URL=http://localhost:8080/fhir \
    ETL_BUNDLES_PATH="$(pwd)/etl_worker/data/bundles" \
    ETL_DIRICHLET_ALPHA="$ALPHA" \
    PYTHONUNBUFFERED=1 \
    uv run etl-worker
    log "ETL worker completed."
    log "Waiting for HAPI FHIR to index loaded patients (max 3 min)..."
    j=0
    while [ $j -lt 36 ]; do
        FHIR_PATIENT_COUNT=$(curl -sf "http://localhost:8080/fhir/Patient?_count=1&_summary=count" 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo "0")
        [ "${FHIR_PATIENT_COUNT:-0}" -ge 1000 ] 2>/dev/null && break
        printf '.'; sleep 5; j=$((j+1))
    done
    echo " FHIR indexed: $FHIR_PATIENT_COUNT patients."
    log "FHIR now has $FHIR_PATIENT_COUNT patients."
else
    log "FHIR already has $FHIR_PATIENT_COUNT patients — skipping ETL reload."
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — GRAD NORM CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
log "════ PHASE 0: GRAD NORM CALIBRATION ════"
[ "$EXP" = "A" ] || [ "$EXP" = "all" ] && run_calibration "bert"
[ "$EXP" = "B" ] || [ "$EXP" = "all" ] && run_calibration "llm"

FL_MAX_GRAD_NORM_BERT=1.0
FL_MAX_GRAD_NORM_LLM=1.0
_calib_bert_p75="n/a"
_calib_llm_p75="n/a"
[ "$EXP" = "A" ] || [ "$EXP" = "all" ] && _calib_bert_p75=$(read_calib_C0 "bert")
[ "$EXP" = "B" ] || [ "$EXP" = "all" ] && _calib_llm_p75=$(read_calib_C0 "llm")

if [[ "${FL_USE_LITERATURE_C0:-true}" == "true" ]]; then
    log "C₀=1.0 (literature, data-independent — formal DP guarantee valid)"
    log "  Calibration p75 reference: BERT=${_calib_bert_p75} | LLM=${_calib_llm_p75}"
else
    FL_MAX_GRAD_NORM_BERT="${_calib_bert_p75}"
    FL_MAX_GRAD_NORM_LLM="${_calib_llm_p75}"
    log "WARNING: using data-dependent C₀ (FL_USE_LITERATURE_C0=false) — results not formally DP"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — ICD-10 Coding (PubMedBERT)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$EXP" = "A" ] || [ "$EXP" = "all" ]; then
    run_experiment_matrix "bert" "EXPERIMENT A — ICD-10 Coding (PubMedBERT)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Discharge Summary (Llama-3.2)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$EXP" = "B" ] || [ "$EXP" = "all" ]; then
    run_experiment_matrix "llm" "EXPERIMENT B — Discharge Summary (Llama-3.2)"
fi

log "════════════════════════════════════════════════════════"
log "ALL EXPERIMENTS COMPLETED — exp=$EXP | mode=$RUN_MODE | seeds=${SEEDS[*]}"
log "Results in: $(pwd)/$LOGS/"
log "JSON files: $(ls "$LOGS"/*.json 2>/dev/null | wc -l) runs saved"
log "════════════════════════════════════════════════════════"
