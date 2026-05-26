# evaluation — Evaluation Pipeline

Post-training evaluation modules for both experiments. All modules are installable as a package (`uv run python -m evaluation.<module>`).

## Modules

| Module | Purpose |
|--------|---------|
| `src/evaluation/icd_metrics.py` | Multi-label ICD-10 metrics: Micro/Macro-F1, AUC-ROC, P@k, R@k (Mullenbach et al. 2018 methodology) |
| `src/evaluation/statistical_analysis.py` | 95% bootstrap CI (B=10,000) and Wilcoxon signed-rank test over n=3 seeds |
| `src/evaluation/metrics_logger.py` | `FederatedRunLogger` — parses FL server logs into structured JSON; GPU timing |
| `src/evaluation/plots.py` | Publication-quality figures (IEEE Healthcom style): ε vs F1, F1 vs α |
| `src/evaluation/gradient_inversion.py` | DLG attack (Zhu et al. 2019) — quantifies text reconstruction risk from gradients |
| `src/evaluation/run_gradient_inversion.py` | CLI wrapper for gradient inversion evaluation |
| `src/evaluation/post_eval.py` | Post-training LLM-as-judge evaluation pipeline (Experiment B extension) |
| `src/evaluation/statistical_analysis.py` | Bootstrap CI + Wilcoxon test |
| `src/evaluation/summarization_metrics.py` | ROUGE-1/2/L + BERTScore (Experiment B extension) |
| `src/evaluation/llm_judge.py` | LLM-as-judge ensemble (Qwen + Gemma + DeepSeek via OpenRouter) (Experiment B extension) |
| `src/evaluation/fhir_benchmark.py` | FHIR server latency and completeness benchmark |

## Running statistical analysis

```bash
uv run python -m evaluation.statistical_analysis \
  --results-dir experiment_logs/ \
  --output experiment_logs/statistical_summary.json
```

## Running gradient inversion

```bash
uv run python -m evaluation.run_gradient_inversion \
  --fhir-url http://localhost:8080/fhir \
  --sigma-values 0.0 0.5 1.0 2.0 \
  --n-samples 10 \
  --output experiment_logs/gradient_inversion.json
```

## Key environment variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Required for LLM-as-judge evaluation (Experiment B only) |
| `METRICS_LOG_DIR` | Directory for CSV metrics output (default: `evaluation/logs`) |
