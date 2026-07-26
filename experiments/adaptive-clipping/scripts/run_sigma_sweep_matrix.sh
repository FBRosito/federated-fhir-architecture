#!/usr/bin/env bash
set -uo pipefail
# run_sigma_sweep_matrix.sh
#
# Sigma sweep: 4 sigma values x 5 seeds = 20 runs, C0=1.0 fixed throughout.
# Idempotent (each run script skips if already complete) and does not abort
# on individual run failure — continues the sweep and reports a final
# failure count. Mirrors run_matrix.sh / run_c0_sweep_matrix.sh's pattern.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Small sigma can mean larger effective gradients surviving the clip step
# less predictably (more non-finite-guard retries), so this sweep defaults
# to a longer per-run estimate than the C0 sweep's 900s.
ESTIMATED_SECONDS_PER_RUN="${ESTIMATED_SECONDS_PER_RUN:-1800}"

SIGMA_VALUES=(0.1 0.3 0.5 2.0)
SEEDS=(0 1 2 3 4)

TOTAL_RUNS=$(( ${#SIGMA_VALUES[@]} * ${#SEEDS[@]} ))
TOTAL_EST_S=$(( TOTAL_RUNS * ESTIMATED_SECONDS_PER_RUN ))
echo "Sigma sweep: ${TOTAL_RUNS} runs, ~$((TOTAL_EST_S / 3600))h estimated (${ESTIMATED_SECONDS_PER_RUN}s/run)."

FAILED=0
for sigma in "${SIGMA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_START=$(date +%s)
        if bash "$SCRIPT_DIR/run_sigma_sweep.sh" "$sigma" "$seed"; then
            RUN_END=$(date +%s)
            echo "[OK]    sigma=$sigma seed=$seed duration=$((RUN_END - RUN_START))s"
        else
            RUN_END=$(date +%s)
            echo "[ERROR] sigma=$sigma seed=$seed duration=$((RUN_END - RUN_START))s — continuing sweep."
            FAILED=$((FAILED + 1))
        fi
    done
done

echo "Sigma sweep complete. ${FAILED} run(s) failed out of ${TOTAL_RUNS}."
