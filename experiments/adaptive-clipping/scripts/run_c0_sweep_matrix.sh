#!/usr/bin/env bash
set -uo pipefail
# run_c0_sweep_matrix.sh
#
# C0 sweep: 4 C0 values x 5 seeds = 20 runs, sigma=1.0 fixed throughout.
# Idempotent (each run script skips if already complete) and does not abort
# on individual run failure — continues the sweep and reports a final
# failure count. Mirrors run_matrix.sh's pattern.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESTIMATED_SECONDS_PER_RUN="${ESTIMATED_SECONDS_PER_RUN:-900}"

C0_VALUES=(1.0 0.989 0.22 0.05)
SEEDS=(0 1 2 3 4)

TOTAL_RUNS=$(( ${#C0_VALUES[@]} * ${#SEEDS[@]} ))
TOTAL_EST_S=$(( TOTAL_RUNS * ESTIMATED_SECONDS_PER_RUN ))
echo "C0 sweep: ${TOTAL_RUNS} runs, ~$((TOTAL_EST_S / 3600))h estimated (${ESTIMATED_SECONDS_PER_RUN}s/run)."

FAILED=0
for c0 in "${C0_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_START=$(date +%s)
        if bash "$SCRIPT_DIR/run_c0_sweep.sh" "$c0" "$seed"; then
            RUN_END=$(date +%s)
            echo "[OK]    c0=$c0 seed=$seed duration=$((RUN_END - RUN_START))s"
        else
            RUN_END=$(date +%s)
            echo "[ERROR] c0=$c0 seed=$seed duration=$((RUN_END - RUN_START))s — continuing sweep."
            FAILED=$((FAILED + 1))
        fi
    done
done

echo "C0 sweep complete. ${FAILED} run(s) failed out of ${TOTAL_RUNS}."
