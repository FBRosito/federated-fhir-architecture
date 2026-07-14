#!/usr/bin/env bash
# Experimental matrix FL+FHIR for Qualis A1 publication
#
# EXPERIMENT A — ICD-10 Coding (PubMedBERT + per-label attention + LoRA)
# EXPERIMENT B — Discharge Summary (Llama-3.2 + LoRA)
#
# 6 configurations × 3 seeds × 2 experiments = 36 main runs
# + 2 grad_norm calibration = 38 runs total
#
# Configurations:
#   1. Centralised                        — 3 seeds (upper bound without FL)
#   2. FL FedProx, Non-IID (α=0.5, σ=0)  — 3 seeds (FL baseline, no privacy)
#   3. FL FedAvg,  Non-IID (α=0.5, σ=0)  — 3 seeds (FedAvg vs FedProx)
#   4. FL FedProx + DP σ=0.5 (α=0.5)     — 3 seeds (light privacy)
#   5. FL FedProx + DP σ=1.0 (α=0.5)     — 3 seeds (moderate privacy)
#   6. FL FedProx + DP σ=2.0 (α=0.5)     — 3 seeds (strong privacy)
#
# Usage:
#   bash run_experiments.sh [--smoke] [--mini] [--exp {A|B|all}]
#
#   --smoke      local validation: 20 examples, 2 rounds, single seed, all
#                code paths (~2-3h). MUST pass before going to the cloud.
#   --mini       statistical validation: 20 examples, 2 rounds, 3 seeds (42/43/44),
#                all 6 configs × 2 backends. Produces real Wilcoxon p-values and
#                saves epsilon_spent per round. Est. ~6-7h on RTX 3060.
#                Run BEFORE cloud to confirm pipeline is correct end-to-end.
#   --exp A      only Experiment A (ICD-10 coding with PubMedBERT)
#   --exp B      only Experiment B (discharge summary with Llama)
#   --exp all    both (default)
#
# Environment variables for hardware tuning (set BEFORE calling the script):
#
#   For A100 40/80 GB (parallel silos, larger batch):
#     export FL_BATCH_SIZE=8
#     export FL_GRADIENT_ACCUM_STEPS=8
#     export FL_PARALLEL_GPU=true
#
#   For RTX 3060 12 GB (script defaults):
#     FL_BATCH_SIZE=1, FL_GRADIENT_ACCUM_STEPS=64, FL_PARALLEL_GPU=false
#
#   CENTRAL_MAX_EXAMPLES  cap on examples for the centralised baseline
#                         (default: 5000; 0 = no cap, can exceed 100k)
#   PHYSIONET_DIR         root of PhysioNet data on the host
#   MIMIC_HOSP_DIR        path to mimiciv/3.1/hosp inside container
#   MIMIC_NOTE_DIR        path to mimic-iv-note/2.2/note inside container
set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
SMOKE=false
MINI=false
EXP="all"
OVERRIDE_MAX_EXAMPLES=""
ALPHA="0.5"     # Dirichlet heterogeneity: 0.1 | 0.5 | 1.0 (set via --alpha)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)        SMOKE=true ;;
        --mini)         MINI=true ;;
        --exp)          shift; EXP="${1:-all}" ;;
        --max-examples) shift; OVERRIDE_MAX_EXAMPLES="${1:-}" ;;
        --alpha)        shift; ALPHA="${1:-0.5}" ;;  # Dirichlet α for Non-IID sweep
    esac
    shift
done
export ETL_DIRICHLET_ALPHA="$ALPHA"

LOGS="experiment_logs"
mkdir -p "$LOGS"

# ── Master log file — captures ALL output (stdout + stderr) ──────────────────
# Filename: experiment_logs/run_YYYYMMDD_HHMMSS_{smoke|full}.log
# To tail live:  tail -f experiment_logs/run_*.log
# To share:      cat experiment_logs/run_*.log | gzip > run.log.gz
RUN_MODE=$([ "$MINI" = "true" ] && echo "mini" || ([ "$SMOKE" = "true" ] && echo "smoke" || echo "full"))
MASTER_LOG="$LOGS/run_$(date -u '+%Y%m%d_%H%M%S')_${RUN_MODE}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Master log: $MASTER_LOG"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# ── Parameters based on mode ──────────────────────────────────────────────────
if $SMOKE && ! $MINI; then
    MAX_EXAMPLES=20
    CENTRAL_MAX_EXAMPLES=20
    FL_ROUNDS=2
    CENTRAL_EPOCHS=2
    SEEDS=(42)
    # --max-examples overrides the smoke default (useful for metric validation)
    if [[ -n "$OVERRIDE_MAX_EXAMPLES" ]]; then
        MAX_EXAMPLES="$OVERRIDE_MAX_EXAMPLES"
        CENTRAL_MAX_EXAMPLES="$OVERRIDE_MAX_EXAMPLES"
    fi
    log "SMOKE MODE: examples=$MAX_EXAMPLES rounds=$FL_ROUNDS seeds=${SEEDS[*]}"
    log "Covers ALL code paths: centralised+bert, centralised+llm,"
    log "  FL+FedProx+nodp, FL+FedAvg+nodp, FL+FedProx+DP for both backends."
elif $MINI; then
    # Mini mode: same scope as smoke (20 ex, 2 rounds) but 3 seeds for Wilcoxon.
    # Produces real p-values and validates epsilon_spent per round.
    # All 6 configs × 2 backends × 3 seeds = 36 FL + 6 central runs.
    # Est. ~6-7h on RTX 3060 (vs ~2h smoke / 48h+ full).
    MAX_EXAMPLES=20
    CENTRAL_MAX_EXAMPLES=20
    FL_ROUNDS=2
    CENTRAL_EPOCHS=2
    SEEDS=(42 43 44)
    RUN_MODE="mini"
    log "MINI MODE: examples=$MAX_EXAMPLES rounds=$FL_ROUNDS seeds=${SEEDS[*]}"
    log "  → Wilcoxon-ready: 3 seeds per config"
    log "  → epsilon_spent logged per round for DP configs"
    log "  → All 6 configs × 2 backends (identical to cloud, reduced scope)"
else
    MAX_EXAMPLES=0
    CENTRAL_MAX_EXAMPLES="${CENTRAL_MAX_EXAMPLES:-5000}"
    FL_ROUNDS=10
    CENTRAL_EPOCHS=10
    SEEDS=(42 123 777)
    log "FULL MODE: rounds=$FL_ROUNDS central_max=$CENTRAL_MAX_EXAMPLES seeds=${SEEDS[*]}"
fi

# ── Hardware configuration ────────────────────────────────────────────────────
# LLM (Llama): memory-intensive — small batch by default (safe for RTX 3060)
LLM_BATCH_SIZE="${FL_BATCH_SIZE:-1}"
LLM_GRADIENT_ACCUM="${FL_GRADIENT_ACCUM_STEPS:-64}"

# BERT (PubMedBERT 110M fp32): no quantisation, fits comfortably on any GPU
# Uses batch 8 even on RTX 3060 — ~1.8 GB VRAM with batch=8
BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-8}"
BERT_GRADIENT_ACCUM="${BERT_GRADIENT_ACCUM:-8}"

# GPU parallelism across silos
export FL_PARALLEL_GPU="${FL_PARALLEL_GPU:-false}"

log "Hardware: LLM batch=$LLM_BATCH_SIZE accum=$LLM_GRADIENT_ACCUM | BERT batch=$BERT_BATCH_SIZE accum=$BERT_GRADIENT_ACCUM | parallel_gpu=$FL_PARALLEL_GPU"

SILOS=(ai_client_silo_0 ai_client_silo_1 ai_client_silo_2 ai_client_silo_3 ai_client_silo_4)

# ── Helper functions ───────────────────────────────────────────────────────────

wait_fl_healthy() {
    log "Waiting for fl_server to become healthy..."
    until [ "$(docker inspect --format='{{.State.Health.Status}}' fl_server 2>/dev/null)" = "healthy" ]; do
        printf '.'; sleep 3
    done
    echo " fl_server healthy."
}

# start_fl_server <min_clients> <noise_multiplier> <rounds> <strategy> [proximal_mu]
start_fl_server() {
    local min_clients="$1" noise="$2" rounds="$3" strategy="${4:-fedprox}" mu="${5:-0.01}"
    export FL_MIN_CLIENTS="$min_clients"
    export FL_NUM_ROUNDS="$rounds"
    export FL_NOISE_MULTIPLIER="$noise"
    export FL_STRATEGY="$strategy"
    export FL_LEARNING_RATE=5e-5
    export FL_NUM_EPOCHS=1
    export FL_PROXIMAL_MU="$mu"
    export FL_NETWORK_MODE="${FL_NETWORK_MODE:-simulated}"
    export FL_CA_CERT_PATH="${FL_CA_CERT_PATH:-}"
    export FL_SERVER_CERT_PATH="${FL_SERVER_CERT_PATH:-}"
    export FL_SERVER_KEY_PATH="${FL_SERVER_KEY_PATH:-}"
    # Use docker rm -f directly to avoid docker compose rm deadlock on containers
    # stuck in Created/Dead/Removing states (compose rm hangs waiting for stop).
    docker stop fl_server 2>/dev/null || true
    docker rm -f fl_server 2>/dev/null || true
    docker compose up -d --force-recreate fl_server
    wait_fl_healthy
}

# save_run_json <tag> <backend> <strategy> <noise> <n_silos> <seed> <rounds> <mu>
# Parses the FL server log and writes experiment_logs/<tag>.json
save_run_json() {
    local tag="$1" backend="$2" strategy="$3" noise="$4"
    local n_silos="${5:-5}" seed="${6:-42}" rounds="${7:-5}" mu="${8:-0.01}"
    local server_log="$LOGS/${tag}_server.log"
    local json_out="$LOGS/${tag}.json"
    # C₀ used for this run — resolved at call time from FL_MAX_GRAD_NORM
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

# Collect eval and fit metrics in round order.
# Both are logged once per round: fit first, eval second.
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

# Merge epsilon_spent (and other fit-only fields) into each eval round dict.
# epsilon_spent comes from the RDP accountant in fit() — it is NOT in evaluate().
FIT_KEYS_TO_MERGE = {"epsilon_spent", "train_loss", "lora_b_norm_end",
                     "lora_b_drift_ratio", "proximal_mu"}
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
# Parses the centralised client log and writes experiment_logs/<tag>.json.
# Uses backend-specific patterns to extract eval metrics from the log.
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
    # LLM training: "Epoch X/N completed — loss=Y | ppl=Z | ..."
    for m in re.finditer(r"Epoch (\d+)/\d+ completed.*?loss=([\d.nan]+).*?ppl=([\d.nan]+)", content, re.IGNORECASE):
        rounds_data.append({"epoch": int(m.group(1)),
                            "train_loss": safe_float(m.group(2)),
                            "train_perplexity": safe_float(m.group(3))})
    # LLM eval: "Final evaluation: loss=X.XXXX | ppl=Y.YY"
    m = re.search(r"Final evaluation:.*?loss=([\d.nan]+).*?ppl=([\d.nan]+)", content, re.IGNORECASE)
    if m:
        final_metrics = {"eval_loss": safe_float(m.group(1)),
                         "eval_perplexity": safe_float(m.group(2))}
else:  # bert
    # BERT training: "BERT Epoch X/N completed — loss=Y.YYYY"
    for m in re.finditer(r"BERT Epoch (\d+)/\d+ completed.*?loss=([\d.nan]+)", content, re.IGNORECASE):
        rounds_data.append({"epoch": int(m.group(1)), "train_loss": safe_float(m.group(2))})
    # BERT eval: parse ICD-10 Metrics text block after "BERT evaluation:"
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
        # Keep all expected metric keys even if None (nan AUC with 1 eval sample → null in JSON)

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

# run_fl <tag> <n_silos> <strategy> <noise> <rounds> <backend> <seed> [proximal_mu]
run_fl() {
    local tag="$1" n_silos="$2" strategy="$3" noise="$4" rounds="$5"
    local backend="${6:-llm}" seed="${7:-42}" mu="${8:-0.01}"
    local silo_list=("${SILOS[@]:0:$n_silos}")

    log "FL run: tag=$tag backend=$backend silos=$n_silos strategy=$strategy noise=$noise rounds=$rounds seed=$seed mu=$mu"

    start_fl_server "$n_silos" "$noise" "$rounds" "$strategy" "$mu"

    # Batch/accum per backend
    if [ "$backend" = "bert" ]; then
        export FL_BATCH_SIZE="$BERT_BATCH_SIZE"
        export FL_GRADIENT_ACCUM_STEPS="$BERT_GRADIENT_ACCUM"
        export MAX_SEQ_LEN=512
        export BERT_BENCHMARK=top50
    else
        export FL_BATCH_SIZE="$LLM_BATCH_SIZE"
        export FL_GRADIENT_ACCUM_STEPS="$LLM_GRADIENT_ACCUM"
        export MAX_SEQ_LEN=1024
        export BERT_BENCHMARK=full
    fi

    export FL_MAX_EXAMPLES="$MAX_EXAMPLES"
    export FL_EVAL_ACCURACY=true
    export FL_TOP_K=5
    export FL_NOISE_MULTIPLIER="$noise"
    export FL_TARGET_DELTA=1e-5
    export MODEL_BACKEND="$backend"
    export FL_SEED="$seed"
    export FL_NETWORK_MODE="${FL_NETWORK_MODE:-simulated}"
    export FL_CA_CERT_PATH="${FL_CA_CERT_PATH:-}"
    # C₀ from empirical calibration — backend-specific
    if [ "$backend" = "bert" ]; then
        export FL_MAX_GRAD_NORM="${FL_MAX_GRAD_NORM_BERT:-1.0}"
    else
        export FL_MAX_GRAD_NORM="${FL_MAX_GRAD_NORM_LLM:-1.0}"
    fi
    log "  C₀=$FL_MAX_GRAD_NORM (FL_MAX_GRAD_NORM)"

    docker compose up "${silo_list[@]}" 2>&1 | tee "$LOGS/${tag}_clients.log"
    docker logs fl_server > "$LOGS/${tag}_server.log" 2>&1
    save_run_json "$tag" "$backend" "$strategy" "$noise" "$n_silos" "$seed" "$rounds" "$mu"
    log "Run $tag completed."
}

# run_centralizado <seed> <backend>
run_centralizado() {
    local seed="$1" backend="${2:-llm}"
    local tag="centralizado_${backend}_seed${seed}"
    log ">>> $tag"

    # Backend-specific parameters
    local seq_len lr bs accum bert_bm
    if [ "$backend" = "bert" ]; then
        seq_len=512
        lr="2e-4"
        bs="$BERT_BATCH_SIZE"
        accum="$BERT_GRADIENT_ACCUM"
        bert_bm="top50"
    else
        seq_len=1024
        lr="5e-5"
        bs="$LLM_BATCH_SIZE"
        accum="$LLM_GRADIENT_ACCUM"
        bert_bm="full"
    fi

    docker compose run --rm --no-deps \
        -e FL_MAX_EXAMPLES="$CENTRAL_MAX_EXAMPLES" \
        -e FL_EVAL_ACCURACY=true \
        -e FL_TOP_K=5 \
        -e FL_NUM_ROUNDS="$CENTRAL_EPOCHS" \
        -e MAX_SEQ_LEN="$seq_len" \
        -e FL_BATCH_SIZE="$bs" \
        -e FL_GRADIENT_ACCUM_STEPS="$accum" \
        -e FL_LEARNING_RATE="$lr" \
        -e FL_SEED="$seed" \
        -e MODEL_BACKEND="$backend" \
        -e BERT_BENCHMARK="$bert_bm" \
        -e MIMIC_HOSP_DIR="${MIMIC_HOSP_DIR:-/physionet/mimiciv/3.1/hosp}" \
        -e MIMIC_NOTE_DIR="${MIMIC_NOTE_DIR:-/physionet/mimic-iv-note/2.2/note}" \
        -v "${PHYSIONET_DIR:-$(pwd)/physionet.org/files}:/physionet:ro" \
        ai_client_silo_0 \
        /app/.venv/bin/python -m ai_client.centralized_baseline \
        2>&1 | tee "$LOGS/${tag}.log"
    save_centralizado_json "$tag" "$backend" "$seed" "$CENTRAL_EPOCHS"
    log "$tag completed."
}

# run_calibration <backend>
run_calibration() {
    local backend="${1:-llm}"
    local tag="calibration_${backend}"
    log ">>> GRAD NORM CALIBRATION: backend=$backend"

    start_fl_server 1 0.0 1 fedprox 0.01

    # PubMedBERT: hard cap 512. LLM: 1024 for long MIMIC-IV notes.
    local calib_seq_len=512
    [ "$backend" = "llm" ] && calib_seq_len=1024

    local bert_bm="full"
    [ "$backend" = "bert" ] && bert_bm="top50"

    docker compose run --rm --no-deps \
        -e FL_MAX_EXAMPLES=50 \
        -e FL_CALIBRATE_GRAD_NORM=true \
        -e FL_NOISE_MULTIPLIER=0.0 \
        -e MODEL_BACKEND="$backend" \
        -e BERT_BENCHMARK="$bert_bm" \
        -e MAX_SEQ_LEN="$calib_seq_len" \
        -e FL_BATCH_SIZE=1 \
        -e FL_GRADIENT_ACCUM_STEPS=1 \
        -e FL_CALIB_OUTPUT="/tmp/grad_norm_${backend}.json" \
        ai_client_silo_0 2>&1 | tee "$LOGS/${tag}.log"

    log "Calibration $backend completed. Check /tmp/grad_norm_${backend}.json"
    unset FL_CALIBRATE_GRAD_NORM
}

# read_calib_C0 <backend>
# Extracts the recommended max_grad_norm (p75) from the calibration log.
# Falls back to 1.0 if not found (e.g. calibration was skipped or failed).
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

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT: ensure FHIR server is up and data is loaded
# The FHIR server has no persistent volume — data must be reloaded after every
# container restart via the ETL worker.
# ═══════════════════════════════════════════════════════════════════════════════
log "Ensuring FHIR server is running..."
docker compose up -d hapi_fhir
log "Waiting for FHIR server at http://localhost:8080/fhir/metadata ..."
until curl -sf http://localhost:8080/fhir/metadata > /dev/null 2>&1; do
    printf '.'; sleep 5
done
echo " FHIR server ready."

# Check if FHIR already has data (count Patient resources).
# If fewer than 1000 patients, the database is effectively empty and the ETL
# worker must be run to reload all bundles from etl_worker/data/bundles/.
FHIR_PATIENT_COUNT=$(curl -sf "http://localhost:8080/fhir/Patient?_count=1&_summary=count" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo "0")
if [ "${FHIR_PATIENT_COUNT:-0}" -lt 1000 ] 2>/dev/null; then
    log "FHIR has only $FHIR_PATIENT_COUNT patients — starting ETL worker to reload bundles..."
    docker compose up -d etl_worker
    log "Waiting for ETL worker to finish (max 15 min)..."
    i=0
    while [ $i -lt 180 ]; do
        s=$(docker inspect --format='{{.State.Status}}' etl_worker 2>/dev/null || echo "gone")
        [ "$s" != "running" ] && break
        printf '.'; sleep 5; i=$((i+1))
    done
    echo " ETL Worker: $(docker inspect --format='{{.State.Status}} (exit={{.State.ExitCode}})' etl_worker 2>/dev/null || echo gone)."
    # Wait for FHIR server search index to reflect the new data (async indexing).
    log "Waiting for FHIR server to index loaded patients (max 3 min)..."
    j=0
    while [ $j -lt 36 ]; do
        FHIR_PATIENT_COUNT=$(curl -sf "http://localhost:8080/fhir/Patient?_count=1&_summary=count" 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo "0")
        [ "${FHIR_PATIENT_COUNT:-0}" -ge 1000 ] 2>/dev/null && break
        printf '.'; sleep 5; j=$((j+1))
    done
    echo " FHIR indexed: $FHIR_PATIENT_COUNT patients."
    log "FHIR now has $FHIR_PATIENT_COUNT patients after ETL reload."
else
    log "FHIR already has $FHIR_PATIENT_COUNT patients — skipping ETL reload."
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — GRAD NORM CALIBRATION (before DP experiments)
# ═══════════════════════════════════════════════════════════════════════════════
log "════ PHASE 0: GRAD NORM CALIBRATION ════"
[ "$EXP" = "A" ] || [ "$EXP" = "all" ] && run_calibration "bert"
[ "$EXP" = "B" ] || [ "$EXP" = "all" ] && run_calibration "llm"

# Determine C₀ for DP runs.
# FL_USE_LITERATURE_C0=true (default): C₀=1.0 (Yu et al. 2022 / Anil et al. 2022).
#   Calibration is informational only — confirms p75 order of magnitude, does NOT
#   feed into the DP guarantee. This is required for formal data-independence of C₀.
# FL_USE_LITERATURE_C0=false: uses calibrated p75, which is data-dependent.
#   Only for ablations; results are NOT citable as formally DP.
FL_MAX_GRAD_NORM_BERT=1.0
FL_MAX_GRAD_NORM_LLM=1.0
_calib_bert_p75="n/a"
_calib_llm_p75="n/a"
if [ "$EXP" = "A" ] || [ "$EXP" = "all" ]; then
    _calib_bert_p75=$(read_calib_C0 "bert")
fi
if [ "$EXP" = "B" ] || [ "$EXP" = "all" ]; then
    _calib_llm_p75=$(read_calib_C0 "llm")
fi
if [[ "${FL_USE_LITERATURE_C0:-true}" == "true" ]]; then
    log "C₀=1.0 (literature, data-independent — formal DP guarantee valid)"
    log "  Calibration p75 reference: BERT=${_calib_bert_p75} | LLM=${_calib_llm_p75}"
else
    FL_MAX_GRAD_NORM_BERT="${_calib_bert_p75}"
    FL_MAX_GRAD_NORM_LLM="${_calib_llm_p75}"
    log "WARNING: using data-dependent C₀ (FL_USE_LITERATURE_C0=false)"
    log "  FL_MAX_GRAD_NORM_BERT=${FL_MAX_GRAD_NORM_BERT} | FL_MAX_GRAD_NORM_LLM=${FL_MAX_GRAD_NORM_LLM}"
    log "  Results are NOT citable as formally differentially private."
fi

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: runs the 6 configurations for a given backend
# ═══════════════════════════════════════════════════════════════════════════════
run_experiment_matrix() {
    local backend="$1"
    local exp_label="$2"

    log "════ $exp_label (backend=$backend) ════"

    # ── Config 1: Centralised ─────────────────────────────────────────────────
    log "CONFIG 1/6 — CENTRALISED"
    for seed in "${SEEDS[@]}"; do run_centralizado "$seed" "$backend"; done

    # ── Config 2: FL FedProx, moderate Non-IID (α=${ALPHA}), no DP ──────────────
    log "CONFIG 2/6 — FL FedProx α=${ALPHA} σ=0 (FL baseline)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_nodp_${backend}_seed${seed}" \
               5 fedprox 0.0 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    # ── Config 3: FL FedAvg, moderate Non-IID (α=${ALPHA}), no DP ────────────
    log "CONFIG 3/6 — FL FedAvg α=${ALPHA} σ=0 (FedProx vs FedAvg)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedavg_alpha${ALPHA}_nodp_${backend}_seed${seed}" \
               5 fedavg 0.0 "$FL_ROUNDS" "$backend" "$seed" 0.0
    done

    # ── Config 4: FL FedProx + DP σ=0.5 (light privacy) ──────────────────────
    log "CONFIG 4/6 — FL FedProx α=${ALPHA} σ=0.5 (light DP)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_dp0.5_${backend}_seed${seed}" \
               5 fedprox 0.5 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    # ── Config 5: FL FedProx + DP σ=1.0 (moderate privacy) ───────────────────
    log "CONFIG 5/6 — FL FedProx α=${ALPHA} σ=1.0 (moderate DP)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_dp1.0_${backend}_seed${seed}" \
               5 fedprox 1.0 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    # ── Config 6: FL FedProx + DP σ=2.0 (strong privacy) ─────────────────────
    log "CONFIG 6/6 — FL FedProx α=${ALPHA} σ=2.0 (strong DP)"
    for seed in "${SEEDS[@]}"; do
        run_fl "fl_fedprox_alpha${ALPHA}_dp2.0_${backend}_seed${seed}" \
               5 fedprox 2.0 "$FL_ROUNDS" "$backend" "$seed" 0.01
    done

    log "=== $exp_label COMPLETED ==="
}

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

# ═══════════════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════"
log "ALL EXPERIMENTS COMPLETED — exp=$EXP | mode=$RUN_MODE | seeds=${SEEDS[*]}"
log "Results in: $(pwd)/$LOGS/"
log "JSON files: $(ls "$LOGS"/*.json 2>/dev/null | wc -l) runs saved"
log "════════════════════════════════════════════════════════"
