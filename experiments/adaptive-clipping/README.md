# Adaptive Clipping — Article 2 (HERALD)

Per-layer adaptive DP-SGD clipping for federated PubMedBERT ICD-10 coding, built on top of [HERALD](../../README.md) without modifying any file outside this directory.

## Hypothesis

The original HERALD paper (Article 1) found that uniform gradient clipping (C₀=1.0, one threshold for every LoRA parameter) is the dominant driver of the Micro-F1 drop under DP-SGD (~63% relative drop), while Gaussian noise itself has marginal impact (ΔF1 < 0.013 between σ=0.5 and σ=2.0). This experiment tests whether calibrating clipping **per attention layer**, using each layer's historical gradient-norm percentile (after a warmup period), improves the Micro-F1/ε trade-off without increasing the privacy budget.

The clipping thresholds are grouped by **transformer attention layer**, not by individual LoRA projection: all LoRA parameters (`query`, `key`, `value`, `output`) belonging to the same encoder layer index share one threshold. This granularity has a cleaner theoretical justification than per-projection grouping — an attention layer is a functional unit of the transformer, responsible for one level of semantic representation. Calibrating the clipping threshold at this granularity respects the model's structure and produces thresholds that are interpretable in terms of network depth.

Per-layer thresholds are rescaled so the vector sensitivity of the whole gradient (`sqrt(sum(C_l^2))`) always equals the baseline's global C₀ — the same technique Opacus's own `clipping="per_layer"` mode uses to preserve its DP guarantee, except here the per-layer *shares* are adaptive (percentile-calibrated) rather than equal. This is what makes ε identical to the baseline's by construction, not by argument — see `src/adaptive_clipping/training.py`'s module docstring for the full derivation.

## Hardware requirements

- NVIDIA GPU, CUDA ≥ 12.4, 12 GB VRAM (RTX 3060 class) — PubMedBERT uses relatively little VRAM (~440 MB base + LoRA), so the 5 silos may fit in parallel depending on `FL_PARALLEL_GPU`.
- Full matrix: 40 runs (2 strategies × 2 σ values × 10 seeds), 20 FL rounds each. At the default `ESTIMATED_SECONDS_PER_RUN=5400` (90 min/run), that's **~60h serial** on an RTX 3060; less if silos run in parallel on the same GPU or across multiple GPUs.

## Prerequisites

This package does **not** start the FHIR server / ETL data load itself — it assumes the root HERALD infrastructure is already running and loaded:

```bash
cd ../..              # repo root
make up-infra          # or: bash run_nodocker.sh (bare-metal path)
```

## Setup

```bash
cd experiments/adaptive-clipping
uv sync
```

This is a **standalone uv project** (its own `.venv`/`uv.lock`), not a member of the root repo's workspace — the root `pyproject.toml` is outside this directory and cannot be modified. `ai-client`, `fl-server`, and `evaluation` are declared as local editable path dependencies, so `uv sync` pulls in the full HERALD training stack (torch, transformers, peft, opacus, etc.) a second time into this project's own virtual environment.

## Running the full matrix

```bash
bash scripts/run_matrix.sh
```

Runs all 40 configurations (idempotent — skips any run whose final-round JSONL files already exist for all 5 silos; continues past individual run failures and reports a final failure count). Set `ESTIMATED_SECONDS_PER_RUN` to override the 5400s default used for the printed time estimate.

Single runs:

```bash
bash scripts/run_baseline.sh <sigma> <seed>     # e.g. bash scripts/run_baseline.sh 1.0 0
bash scripts/run_per_layer.sh <sigma> <seed>
```

Before any real (GPU, multi-hour) run, validate the core `PerLayerClipper` logic in isolation (CPU, sub-second, no model download):

```bash
uv run python scripts/smoke_check_clipper.py
```

## Reproducing figures and tables

```bash
uv run python -m analysis.bootstrap_ci        # writes logs/bootstrap_ci.csv
uv run python -m analysis.plot_convergence     # writes figures/convergence.{png,pdf}
uv run python -m analysis.generate_tables      # writes logs/table_iii.{csv,tex}
```

## Log format

Each FL round, for each silo, one JSONL file is written to `logs/{experiment_tag}/{silo_id}/round_{round:03d}.jsonl` — one file per round, each file containing exactly one JSON line:

```json
{
  "round": 20,
  "silo_id": 0,
  "sigma": 1.0,
  "clipping_strategy": "per_layer",
  "global_c0": 1.0,
  "per_layer_thresholds": {"encoder.layer.0": 0.42, "encoder.layer.1": 0.38, "...": "...", "other": 1.0},
  "per_layer_norms": {"encoder.layer.0": {"mean": 0.5, "std": 0.1, "min": 0.2, "max": 0.9, "history": [...]}},
  "micro_f1": 0.31,
  "epsilon": 2.29,
  "delta": 1e-05,
  "wall_clock_seconds": 187.4
}
```

`experiment_tag` follows `{strategy}_sigma{sigma}_seed{seed}` (e.g. `per_layer_sigma1.0_seed3`), matching the directory names `scripts/run_*.sh` and `analysis/*.py` both expect.

## Reference

Builds on [HERALD](../../README.md) — *Healthcare fEderated leaRning Architecture with LoRA and Differential-privacy* — see `../../CLAUDE.md` and `../../docs/architecture.md` for the base system's 4-tier architecture. `MODEL_BACKEND=bert` (PubMedBERT + PLM-ICD, the paper's Experiment A) is the default and only backend covered by the 40-run matrix above; `training.py` also implements the equivalent per-layer clipping for `MODEL_BACKEND=llm` (Llama-3.2 LoRA), available for manual runs but not part of the default matrix or `configs/*.yaml`.
