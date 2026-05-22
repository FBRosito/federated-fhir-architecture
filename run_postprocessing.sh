#!/usr/bin/env bash
# =============================================================================
# run_postprocessing.sh — post-processing pipeline after run_experiments.sh
#
# Usage:
#   bash run_postprocessing.sh [options]
#
# Options:
#   --logs-dir   <dir>   Directory with experiment JSON files (default: experiment_logs)
#   --exp        A|B|all Which experiments to process (default: all)
#   --checkpoint <dir>   LoRA checkpoint dir for LLM judge eval (Exp B only)
#   --skip-judges        Skip LLM-as-judge evaluation
#   --skip-gi            Skip gradient inversion attack
#   --max-samples <n>    Judge subsample size (passed as --judge-samples; default: 150)
#   --fhir-url   <url>   FHIR server URL for post-eval (default: http://localhost:8080/fhir)
#
# Steps:
#   1. Statistical analysis  — bootstrap CI + Wilcoxon + Bonferroni
#   2. Publication plots     — ε×F1, F1×α, convergence curves
#   3. LLM-as-judge (Exp B) — requires OPENROUTER_API_KEY
#   4. Gradient inversion    — DLG attack security evaluation
#
# Environment:
#   OPENROUTER_API_KEY — required for step 3 (get at openrouter.ai/keys)
# =============================================================================
set -euo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────────
LOGS="experiment_logs"
EXP="all"
CHECKPOINT=""
SKIP_JUDGES=false
SKIP_GI=false
MAX_SAMPLES=150
FHIR_URL="${FHIR_SERVER_URL:-http://localhost:8080/fhir}"

# ─── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --logs-dir)   shift; LOGS="${1:?--logs-dir requires a value}" ;;
        --exp)        shift; EXP="${1:?--exp requires a value}" ;;
        --checkpoint) shift; CHECKPOINT="${1:?--checkpoint requires a value}" ;;
        --skip-judges)  SKIP_JUDGES=true ;;
        --skip-gi)      SKIP_GI=true ;;
        --max-samples)  shift; MAX_SAMPLES="${1:?--max-samples requires a value}" ;;
        --fhir-url)     shift; FHIR_URL="${1:?--fhir-url requires a value}" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
section() { log "════ $* ════"; }

require_logs_dir() {
    if [[ ! -d "$LOGS" ]]; then
        echo "ERROR: logs directory '$LOGS' not found. Run run_experiments.sh first."
        exit 1
    fi
    n_json=$(ls "$LOGS"/*.json 2>/dev/null | grep -v "statistical_summary\|gradient_inversion" | wc -l)
    if [[ "$n_json" -eq 0 ]]; then
        echo "ERROR: no experiment JSON files found in '$LOGS/'."
        exit 1
    fi
    log "Found $n_json experiment JSON files in $LOGS/"
}

FIGURES_DIR="$LOGS/figures"
mkdir -p "$FIGURES_DIR"

# ─── Step 1: Statistical analysis ─────────────────────────────────────────────
run_statistical_analysis() {
    section "STEP 1: Statistical Analysis (Bootstrap CI + Wilcoxon + Bonferroni)"
    uv run python - "$LOGS" "$EXP" <<'PYEOF'
import sys, json, glob, os, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("stats")

logs_dir, exp_filter = sys.argv[1], sys.argv[2]

from evaluation.statistical_analysis import (
    summarize_runs, compare_all_vs_baseline, apply_bonferroni, confidence_interval
)

# Load all experiment JSONs
runs: dict[str, dict] = {}
for path in sorted(glob.glob(f"{logs_dir}/*.json")):
    name = Path(path).stem
    if name in ("statistical_summary", "gradient_inversion"):
        continue
    try:
        with open(path) as f:
            runs[name] = json.load(f)
    except Exception as e:
        log.warning("Skipping %s: %s", path, e)

if not runs:
    log.error("No experiment JSONs loaded.")
    sys.exit(1)

log.info("Loaded %d runs: %s", len(runs), list(runs.keys()))

# Extract final metric per run (last round)
def get_final_metric(run: dict, metric: str) -> float | None:
    per_round = run.get("per_round_eval", [])
    if per_round:
        return per_round[-1].get(metric)
    return run.get("final_metrics", {}).get(metric)

# Group runs by config (strip seed suffix)
import re
config_metrics: dict[str, dict[str, list[float]]] = {}
for name, run in runs.items():
    backend = run.get("config", {}).get("backend", "unknown")
    # Strip seed suffix: ..._seed42 → base config name
    config = re.sub(r"_seed\d+$", "", name)
    # Primary metric per backend:
    #   bert → micro_f1  (stored as "micro_f1" by icd_metrics.py)
    #   llm  → eval_perplexity
    metric = "micro_f1" if backend == "bert" else "eval_perplexity"
    val = get_final_metric(run, metric)
    if val is None:
        continue
    config_metrics.setdefault(backend, {}).setdefault(config, []).append(val)

summary_out: dict = {}
for backend, configs in config_metrics.items():
    log.info("--- Backend: %s ---", backend)
    ci_map = summarize_runs(configs)
    baseline_key = next((k for k in configs if "centraliz" in k.lower()), list(configs.keys())[0])
    comparisons = compare_all_vs_baseline(configs, baseline_key=baseline_key)
    comparisons_adj = apply_bonferroni(comparisons)
    summary_out[backend] = {
        "primary_metric": "micro_f1" if backend == "bert" else "eval_perplexity",
        "baseline": baseline_key,
        "configs": {
            cfg: {"ci": str(ci), "mean": ci.mean, "ci_lower": ci.ci_lower, "ci_upper": ci.ci_upper}
            for cfg, ci in ci_map.items()
        },
        "comparisons_vs_baseline": comparisons_adj,
    }

out_path = f"{logs_dir}/statistical_summary.json"
with open(out_path, "w") as f:
    json.dump(summary_out, f, indent=2, default=str)
log.info("Statistical summary written to %s", out_path)
PYEOF
    log "Statistical summary: $LOGS/statistical_summary.json"
}

# ─── Step 2: Publication plots ─────────────────────────────────────────────────
run_plots() {
    section "STEP 2: Publication Plots"
    if [[ ! -f "$LOGS/statistical_summary.json" ]]; then
        log "WARN: statistical_summary.json not found — run step 1 first. Skipping plots."
        return
    fi
    uv run python - "$LOGS" "$FIGURES_DIR" <<'PYEOF'
import sys, json, glob, re, logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("plots")

logs_dir, figures_dir = sys.argv[1], sys.argv[2]

try:
    from evaluation.plots import curva_epsilon_vs_f1, curva_f1_vs_alpha, convergence_curves
except ImportError as e:
    log.error("Cannot import plots module: %s", e)
    sys.exit(0)   # non-fatal

# ── Load all experiment runs ──
all_runs: dict[str, dict] = {}
for path in sorted(glob.glob(f"{logs_dir}/*.json")):
    if re.search(r"(statistical_summary|gradient_inversion)", path):
        continue
    try:
        with open(path) as fp:
            all_runs[re.sub(r"_seed\d+$", "", __import__("pathlib").Path(path).stem)] = json.load(fp)
    except Exception:
        pass

def get_metric(run: dict, key: str):
    """Returns final metric value: checks final_metrics first, then last per_round_eval."""
    v = run.get("final_metrics", {}).get(key)
    if v is not None:
        return v
    per = run.get("per_round_eval", [])
    return per[-1].get(key) if per else None

def get_round_series(run: dict, key: str) -> list[float]:
    """Returns per-round metric series for convergence plots."""
    return [r.get(key) for r in run.get("per_round_eval", []) if r.get(key) is not None]

# ── Figure 1: ε vs F1 (DP ablation — FedProx + BERT, varying σ) ──
log.info("Generating ε vs F1 figure...")
try:
    dp_bert = {
        name: run for name, run in all_runs.items()
        if "fedprox" in name and "bert" in run.get("config", {}).get("backend", "")
        and run.get("config", {}).get("noise_multiplier", 0) > 0
    }
    eps_vals, f1_vals = [], []
    for name in sorted(dp_bert, key=lambda n: all_runs[n]["config"]["noise_multiplier"]):
        run = dp_bert[name]
        eps = get_metric(run, "epsilon_cumulative") or get_metric(run, "epsilon_spent") or get_metric(run, "epsilon")
        f1  = get_metric(run, "micro_f1")
        if eps is not None and f1 is not None:
            eps_vals.append(eps)
            f1_vals.append(f1)
    if len(eps_vals) >= 2:
        curva_epsilon_vs_f1(eps_vals, f1_vals, output_path=f"{figures_dir}/epsilon_vs_f1.pdf")
        log.info("Saved epsilon_vs_f1.pdf  (n=%d DP configs)", len(eps_vals))
    else:
        log.warning("epsilon_vs_f1 skipped: need epsilon_cumulative in run JSON (n=%d)", len(eps_vals))
except Exception as e:
    log.warning("epsilon_vs_f1 failed: %s", e)

# ── Figure 2: F1 vs α (heterogeneity ablation — FedProx vs FedAvg) ──
log.info("Generating F1 vs α figure...")
try:
    alpha_re = re.compile(r"alpha([\d.]+)")
    def extract_alpha(name):
        m = alpha_re.search(name); return float(m.group(1)) if m else None

    fedprox_by_alpha: dict[float, list[float]] = {}
    fedavg_by_alpha:  dict[float, list[float]] = {}
    for name, run in all_runs.items():
        backend = run.get("config", {}).get("backend", "")
        alpha = extract_alpha(name)
        if alpha is None or backend not in ("bert", "llm"):
            continue
        metric = "micro_f1" if backend == "bert" else "eval_perplexity"
        val = get_metric(run, metric)
        if val is None:
            continue
        if "fedprox" in name:
            fedprox_by_alpha.setdefault(alpha, []).append(val)
        elif "fedavg" in name:
            fedavg_by_alpha.setdefault(alpha, []).append(val)

    alphas = sorted(set(list(fedprox_by_alpha) + list(fedavg_by_alpha)))
    if len(alphas) >= 1:
        import statistics as _st
        f1_fp = [_st.mean(fedprox_by_alpha[a]) for a in alphas if a in fedprox_by_alpha]
        f1_fa = [_st.mean(fedavg_by_alpha[a])  for a in alphas if a in fedavg_by_alpha] or None
        valid_alphas = [a for a in alphas if a in fedprox_by_alpha]
        if valid_alphas:
            curva_f1_vs_alpha(valid_alphas, f1_fp, f1_fedavg=f1_fa if len(f1_fa or [])==len(valid_alphas) else None,
                              output_path=f"{figures_dir}/f1_vs_alpha.pdf")
            log.info("Saved f1_vs_alpha.pdf  (α=%s)", valid_alphas)
        else:
            log.warning("f1_vs_alpha skipped: no FedProx runs with alpha in name")
    else:
        log.warning("f1_vs_alpha skipped: need runs with alpha<value> in name")
except Exception as e:
    log.warning("f1_vs_alpha failed: %s", e)

# ── Figure 3: Convergence curves (per-round metric for all configs) ──
log.info("Generating convergence curves...")
try:
    metrics_by_config: dict[str, list[float]] = {}
    for name, run in all_runs.items():
        backend = run.get("config", {}).get("backend", "")
        metric = "micro_f1" if backend == "bert" else "eval_perplexity"
        series = get_round_series(run, metric)
        if series:
            metrics_by_config[name] = series

    if metrics_by_config:
        max_rounds = max(len(v) for v in metrics_by_config.values())
        rounds = list(range(1, max_rounds + 1))
        convergence_curves(rounds, metrics_by_config, output_path=f"{figures_dir}/convergence.pdf")
        log.info("Saved convergence.pdf  (%d configs, %d rounds)", len(metrics_by_config), max_rounds)
    else:
        log.warning("convergence_curves skipped: no per-round metric data found")
except Exception as e:
    log.warning("convergence_curves failed: %s", e)
PYEOF
    log "Figures: $FIGURES_DIR/"
}

# ─── Step 3: LLM-as-judge (Experiment B) ──────────────────────────────────────
run_llm_judges() {
    section "STEP 3: LLM-as-Judge (Experiment B — Discharge Summary)"

    if [[ "$SKIP_JUDGES" == true ]]; then
        log "Skipped (--skip-judges)."
        return
    fi

    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        log "WARN: OPENROUTER_API_KEY not set — skipping LLM-as-judge."
        log "      Get a key at https://openrouter.ai/keys and add to .env."
        return
    fi

    if [[ "$EXP" == "A" ]]; then
        log "Skipped (--exp A — LLM judges only apply to Experiment B)."
        return
    fi

    if [[ -z "$CHECKPOINT" ]]; then
        log "WARN: --checkpoint not specified — skipping LLM judge (need LoRA checkpoint path)."
        log "      Re-run with: --checkpoint <path-to-fl-checkpoint-dir>"
        return
    fi

    # LLM judges run ONLY on the "final boss" configs to control API cost.
    # ROUGE/BERTScore (step 1) already covers all runs; judges add expensive
    # qualitative signal only where it matters for the paper's conclusions:
    #   1. Zero-shot / centralized baseline — no fine-tuning
    #   2. FedProx best run (no DP, seed 42) — the paper's main result
    #   3. FedAvg ablation (no DP, seed 42) — FedProx vs FedAvg comparison
    #
    # Pattern matching (all case-insensitive, seed-42 representatives):
    #   *central*llm*        → zero-shot / centralized
    #   *fedprox*nodp*llm*seed42*  → FedProx main result
    #   *fedavg*nodp*llm*seed42*   → FedAvg ablation
    declare -a FINAL_BOSS_JSONS=()
    for f in "$LOGS"/*_llm_*.json; do
        [[ -f "$f" ]] || continue
        name=$(basename "$f" .json)
        # Skip summary/inversion files
        [[ "$name" == "statistical_summary" || "$name" == "gradient_inversion" ]] && continue
        # Match final boss patterns (case-insensitive via tr)
        name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
        if echo "$name_lower" | grep -qE '(central.*llm|fedprox.*nodp.*llm.*seed42|fedavg.*nodp.*llm.*seed42)'; then
            FINAL_BOSS_JSONS+=("$f")
        fi
    done

    if [[ "${#FINAL_BOSS_JSONS[@]}" -eq 0 ]]; then
        log "WARN: no 'final boss' Experiment B JSONs found in $LOGS/."
        log "      Expected patterns: *central*llm*, *fedprox*nodp*llm*seed42*, *fedavg*nodp*llm*seed42*"
        log "      Skipping LLM-as-judge."
        return
    fi

    log "Final-boss configs selected for LLM-as-judge (${#FINAL_BOSS_JSONS[@]} runs):"
    for f in "${FINAL_BOSS_JSONS[@]}"; do log "  → $(basename "$f")"; done

    # Run post_eval for each final-boss run JSON
    # --max-samples 0   → fetch all examples (ROUGE/BERTScore on 100% test set)
    # --judge-samples   → random subsample for paid LLM judges (150 ≈ Wilcoxon p<0.05)
    for json_path in "${FINAL_BOSS_JSONS[@]}"; do
        log "Running post_eval for: $json_path"
        uv run python -m evaluation.post_eval \
            --run-json "$json_path" \
            --checkpoint "$CHECKPOINT" \
            --fhir-url "$FHIR_URL" \
            --max-samples 0 \
            --judge-samples "$MAX_SAMPLES" \
            --seed 42 \
            2>&1 | sed 's/^/  /' || log "WARN: post_eval failed for $json_path (continuing)"
    done
    log "LLM-as-judge evaluation complete."
}

# ─── Step 4: Gradient inversion ───────────────────────────────────────────────
run_gradient_inversion() {
    section "STEP 4: Gradient Inversion Attack (DP Protection Evaluation)"

    if [[ "$SKIP_GI" == true ]]; then
        log "Skipped (--skip-gi)."
        return
    fi

    log "WARNING: This step takes ~30 min per σ value on RTX 3060."
    log "         Use --skip-gi to skip if time is limited."

    uv run python -m evaluation.run_gradient_inversion \
        --fhir-url "$FHIR_URL" \
        --output "$LOGS/gradient_inversion.json" \
        2>&1 | sed 's/^/  /' || log "WARN: gradient inversion failed (non-fatal)"

    if [[ -f "$LOGS/gradient_inversion.json" ]]; then
        log "Gradient inversion results: $LOGS/gradient_inversion.json"
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    log "Post-processing pipeline — logs=$LOGS | exp=$EXP"
    require_logs_dir

    run_statistical_analysis
    run_plots
    run_llm_judges
    run_gradient_inversion

    section "POST-PROCESSING COMPLETE"
    log "Artifacts:"
    [[ -f "$LOGS/statistical_summary.json" ]] && log "  ✓ $LOGS/statistical_summary.json"
    for fig in epsilon_vs_f1.pdf f1_vs_alpha.pdf convergence.pdf; do
        [[ -f "$FIGURES_DIR/$fig" ]] && log "  ✓ $FIGURES_DIR/$fig"
    done
    [[ -f "$LOGS/gradient_inversion.json" ]] && log "  ✓ $LOGS/gradient_inversion.json"
    log "LLM judge results embedded in each Experiment B run JSON (post_eval key)."
}

main "$@"
