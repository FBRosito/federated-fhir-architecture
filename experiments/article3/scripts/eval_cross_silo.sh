#!/usr/bin/env bash
set -euo pipefail
# eval_cross_silo.sh <tag>
#
# Article 3, Part 4: for each silo k in the given dual_lora run, evaluates
# its final global adapter (only — never its local adapter) on every other
# silo's held-out eval data, appending records to logs/cross_silo_eval.jsonl.
# Requires the FHIR server to be up (same data the run itself used).
#
# Usage: bash scripts/eval_cross_silo.sh a3_dual_lora_sigma1.0_seed0

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <tag>" >&2
    exit 1
fi
TAG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"

export FHIR_SERVER_URL="${FHIR_SERVER_URL:-http://localhost:8080/fhir}"
export BERT_LABEL_INDEX_PATH="${BERT_LABEL_INDEX_PATH:-$REPO_ROOT/etl_worker/data/label_index.json}"

cd "$EXP_ROOT"
uv run python -m article3.cross_silo_eval \
    --tag "$TAG" \
    --logs-dir "$EXP_ROOT/logs" \
    --fhir-url "$FHIR_SERVER_URL" \
    --label-index-path "$BERT_LABEL_INDEX_PATH"
