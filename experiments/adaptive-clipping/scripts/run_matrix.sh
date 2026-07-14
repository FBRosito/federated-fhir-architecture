#!/usr/bin/env bash
set -uo pipefail
# run_matrix.sh
#
# Full experiment matrix: 2 strategies x 2 sigma values x 10 seeds = 40 runs.
# Idempotent (each run script skips if already complete) and does not abort
# on individual run failure — continues the matrix and reports a final
# failure count.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESTIMATED_SECONDS_PER_RUN="${ESTIMATED_SECONDS_PER_RUN:-5400}"

STRATEGIES=(baseline per_layer)
SIGMAS=(1.0 2.0)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

TOTAL_RUNS=$(( ${#STRATEGIES[@]} * ${#SIGMAS[@]} * ${#SEEDS[@]} ))
TOTAL_EST_S=$(( TOTAL_RUNS * ESTIMATED_SECONDS_PER_RUN ))
echo "Matrix: ${TOTAL_RUNS} runs, ~$((TOTAL_EST_S / 3600))h estimated (${ESTIMATED_SECONDS_PER_RUN}s/run)."

FAILED=0
for strategy in "${STRATEGIES[@]}"; do
    SCRIPT="run_baseline.sh"
    [[ "$strategy" == "per_layer" ]] && SCRIPT="run_per_layer.sh"

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
