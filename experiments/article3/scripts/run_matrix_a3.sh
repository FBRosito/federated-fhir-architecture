#!/usr/bin/env bash
set -uo pipefail
# run_matrix_a3.sh
#
# Full Article 3 matrix: 3 strategies x 1 sigma value x 10 seeds = 30 runs,
# each at R=100 (script default). Idempotent (each run script skips if
# already complete) and does not abort on individual run failure —
# continues the matrix and reports a final failure count. dp_lora/ffa_lora
# (20 runs) are already complete as of this matrix's previous run; adding
# dual_lora only adds its 10 new runs, the rest are skipped by the
# idempotency check in each run_*.sh script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESTIMATED_SECONDS_PER_RUN="${ESTIMATED_SECONDS_PER_RUN:-15000}"

STRATEGIES=(dp_lora ffa_lora dual_lora)
SIGMAS=(1.0)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

TOTAL_RUNS=$(( ${#STRATEGIES[@]} * ${#SIGMAS[@]} * ${#SEEDS[@]} ))
TOTAL_EST_S=$(( TOTAL_RUNS * ESTIMATED_SECONDS_PER_RUN ))
echo "Matrix: ${TOTAL_RUNS} runs, ~$((TOTAL_EST_S / 3600))h estimated (${ESTIMATED_SECONDS_PER_RUN}s/run)."

FAILED=0
for strategy in "${STRATEGIES[@]}"; do
    SCRIPT="run_dp_lora.sh"
    [[ "$strategy" == "ffa_lora" ]] && SCRIPT="run_ffa_lora.sh"
    [[ "$strategy" == "dual_lora" ]] && SCRIPT="run_dual_lora.sh"

    for sigma in "${SIGMAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN_START=$(date +%s)
            if bash "$SCRIPT_DIR/${SCRIPT}" "$sigma" "$seed"; then
                RUN_END=$(date +%s)
                echo "[OK]    strategy=$strategy sigma=$sigma seed=$seed duration=$((RUN_END - RUN_START))s"
            else
                RUN_END=$(date +%s)
                echo "[ERROR] strategy=$strategy sigma=$sigma seed=$seed duration=$((RUN_END - RUN_START))s — continuing matrix."
                FAILED=$((FAILED + 1))
            fi
        done
    done
done

echo "Matrix complete. ${FAILED} run(s) failed out of ${TOTAL_RUNS}."
