# HERALD: FHIR-Native Federated Learning for Clinical NLP

[![CI](https://github.com/FBRosito/federated-fhir-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/FBRosito/federated-fhir-architecture/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Privacy-preserving clinical NLP across distributed healthcare silos using HL7 FHIR, FedProx, and Differential Privacy.**

> ⚠️ **Research Software**: This is a research prototype published for reproducibility. It is **not validated for clinical use** and must not be used to process real patient data outside an authorized, credentialed environment.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Experiments](#experiments)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Metrics](#metrics)
6. [Quickstart](#quickstart)
7. [Environment Variables](#environment-variables)
8. [Expected Results](#expected-results)
9. [Privacy and Differential Privacy](#privacy-and-differential-privacy)
10. [Repository Structure](#repository-structure)
11. [Citation](#citation)
12. [License](#license)
13. [Contributing](#contributing)
14. [Security](#security)

---

## Architecture Overview

The system implements a **Privacy-Preserving Federated e-Health Architecture** organized in four tiers.

### Tier 1 — Data Interoperability Tier

**Components:** FHIR R4 in-memory server (Python stdlib, port 8080) + ETL Worker

Isolates AI from legacy hospital database complexity. The ETL Worker transforms relational tables (MIMIC-IV CSV) into standardized HL7 FHIR R4 Transaction Bundles (Patient + Condition + DocumentReference), then loads them into the FHIR R4 server. Any hospital that speaks FHIR can plug into this federated network without changing a line of AI code.

### Tier 2 — Federated Orchestration Tier

**Components:** Flower SuperLink (gRPC, port 9091)

Manages the star topology. Never touches patient data. Synchronizes FL training rounds, distributes the global LoRA adapter weights to selected clients, and aggregates gradient updates using FedProx — robust to Non-IID data distribution across silos. gRPC (not REST) minimizes latency for tensor matrix transmission.

### Tier 3 — Edge Computation Tier (Silos)

**Components:** AI Clients (silo_0..N), local GPU, FHIR Consumer, PEFT Module

Each silo runs inside the hospital's firewall. It downloads its assigned data partition via paginated REST calls to the local FHIR R4 server (`/fhir/Condition`, `/fhir/DocumentReference`). The foundation model (PubMedBERT for Experiment A; optionally Llama-3.2 for Experiment B) stays in local GPU memory and **never traverses the network**. Only the LoRA adapter deltas (~15 MB for PubMedBERT) are sent back — a 99.9% reduction in network bandwidth vs. sending the full model.

### Tier 4 — Cross-Cutting Privacy Layer

**Components:** DP-SGD (gradient clipping + Gaussian noise) + TLS in transit

Security is a cross-cutting requirement, not a module. Before LoRA deltas leave the Edge Tier, gradients are clipped (L2 norm ≤ C₀ = 1.0, so no single patient dominates the update) and calibrated Gaussian noise is injected (σ·C₀). Privacy budget (ε, δ) is tracked per-round by the RDP accountant. This directly mitigates the Gradient Inversion Attack (Zhu et al., 2019).

### System Diagram

```
CSV (MIMIC-IV)
     │
     ▼  [Tier 1 — Data Interoperability]
┌─────────────────────────────────────┐
│          etl_worker                 │
│  MIMIC-IV → FHIR Transaction        │
│  Bundles (Patient + Condition +     │
│  Composition + DocumentReference)   │
│  Dirichlet(α=0.5) Non-IID split     │
└──────────────────┬──────────────────┘
                   │ HTTP POST /fhir
                   ▼
     ┌─────────────────────────┐
     │     fhir_r4_server      │
     │  HL7 FHIR R4  :8080     │
     │  Resources: Patient,    │
     │  Condition,             │
     │  DocumentReference      │
     └──────┬──────────────────┘
            │ HTTP GET (paginated)
   [Tier 3 — Edge Computation ─────────────────────────────]
   │                                                        │
   ▼                      ▼                     ▼           ▼
┌──────────┐        ┌──────────┐         ┌──────────┐  ┌──────────┐
│ silo_0   │        │ silo_1   │         │ silo_2   │  │ silo_N   │
│ ai_client│        │ ai_client│         │ ai_client│  │ ai_client│
│ Partition│        │ Partition│         │ Partition│  │ Partition│
│ 0 data   │        │ 1 data   │         │ 2 data   │  │ N data   │
└────┬─────┘        └────┬─────┘         └────┬─────┘  └────┬─────┘
     │[Tier 4 — DP clip + noise before leaving silo]         │
     │ LoRA delta NDArray (gRPC)                             │
     └──────────────┬────────────────────────────────────────┘
                    │ gRPC :9091
                    ▼  [Tier 2 — Federated Orchestration]
     ┌──────────────────────────────┐
     │         fl_server            │
     │  FedProx (μ=0.01)            │
     │  Clean aggregator            │
     │  (no DP noise server-side)   │
     │  Aggregated global LoRA      │
     └──────────────────────────────┘
```

### Round Lifecycle (1 FL Round — 5 Steps)

| Step | Actor | Action |
|------|-------|--------|
| **1. Initialization** | FL Server | Initializes empty LoRA weight vector → broadcasts to selected silos |
| **2. Data Fetching** | AI Client | Queries local FHIR R4 server; builds dataset **in memory** (no disk writes) |
| **3. DP Local Training** | AI Client | Foundation model frozen; gradients flow only through LoRA; Opacus clips (≤ C₀) + injects Gaussian noise (σ·C₀); 1 local epoch |
| **4. Transmission** | AI Client | Sends privatized LoRA delta back to server via gRPC (~3–8 MB for 1B model) |
| **5. Aggregation** | FL Server | FedProx weighted average of deltas; RDP accountant updates (ε, δ); next round begins |

---

## Experiments

**Experiment A (ICD-10 coding with PubMedBERT) is the paper experiment.** Experiment B (discharge summary with Llama-3.2) is an extension included for future work and is not evaluated in the paper.

| | **Experiment A — ICD-10 Coding** *(paper)* | **Experiment B — Discharge Summary** *(extension)* |
|---|---|---|
| **Task** | Multi-label ICD-10 code prediction from clinical progress notes | Abstractive discharge summary generation from structured clinical data |
| **Model** | PubMedBERT-base (110M, no quantization) + PLM-ICD head + LoRA (r=8) | Llama-3.2-1B (NF4 4-bit, BitsAndBytes) + LoRA (r=16) |
| **Input** | LOINC 11506-3 progress notes (DocumentReference) | Admission data + labs + medications (structured prompt) |
| **Output** | Binary vector over ICD-10 code vocabulary | Free-text discharge summary |
| **Primary metric** | Micro-F1 @ 5 | ROUGE-L |
| **Secondary metrics** | Macro-F1, AUC-ROC (micro), P@8, R@8 | ROUGE-1/2, BERTScore(PubMedBERT), LLM-judge ensemble |
| **FHIR resources** | Condition, DocumentReference (LOINC 11506-3), Patient | DocumentReference (LOINC 18842-5), Patient |
| **Reference dataset** | MIMIC-IV-full (codes ≥ 10 occurrences) | MIMIC-IV-Note discharge.csv.gz |

---

## Dataset

### MIMIC-IV v3.1

- **Source:** PhysioNet — requires credentialed access.
- **Clinical records:** ~431K hospital admissions, ~299K patients.
- **ICD-10-CM coding:** Admissions from Oct 2015 onward (MIMIC-IV transition date).
- **Discharge notes:** MIMIC-IV-Note module (`discharge.csv.gz`), ~331K notes.

### Temporal Split

MIMIC-IV v3.1 applies per-patient date-shifting (real dates 2008–2022 → shifted to 2105–2214). Split boundaries calibrated for 80/10/10 by admission volume:

| Split | Condition | Approx. fraction |
|-------|-----------|-----------------|
| Train | `admittime < 2179-01-01` | 80% |
| Val   | `2179 ≤ admittime < 2190` | 10% |
| Test  | `admittime ≥ 2190` (held-out) | 10% |

### Non-IID Partitioning

Dirichlet(α) partitioning over ICD-10 chapters, following Hsu et al. (2019) and McMahan et al. (2017):

| α value | Distribution | Silo dominance |
|---------|-------------|----------------|
| 0.1 | Highly Non-IID | Each silo dominated by 1–2 ICD chapters |
| **0.5** | **Moderate (default)** | **Realistic cross-hospital heterogeneity** |
| 1.0 | Near-IID | Uniform distribution across silos |

**Benchmark subsets (Mullenbach 2018 methodology):**

- `--benchmark full`: all ICD-10 codes with ≥ 10 occurrences (~8,900 unique codes)
- `--benchmark top50`: 50 most frequent codes (MIMIC-IV-50 benchmark)

---

## Models

### Experiment A — PubMedBERT + PLM-ICD

| Property | Value |
|----------|-------|
| Base model | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| Parameters | 110M (no quantization — fits in 12 GB VRAM) |
| Architecture | PLM-ICD (Huang et al., 2022): per-label attention over BERT CLS embeddings |
| LoRA rank | r=8, α=16, dropout=0.1 |
| LoRA targets | `query`, `value` (attention layers) |
| Training objective | BCEWithLogitsLoss (multi-label binary cross-entropy) |
| FL weight exchange | LoRA tensors + classifier head (~15 MB/round) |
| Batch size | 16 |
| Learning rate | 2e-5 |

### Experiment B — Llama-3.2-1B + LoRA

| Property | Value |
|----------|-------|
| Base model | `meta-llama/Llama-3.2-1B` (or `meta-llama/Llama-3.1-8B` — see `MODEL_NAME`) |
| Quantization | NF4 4-bit via BitsAndBytes (double quantization, ~0.8–6 GB VRAM) |
| LoRA rank | r=16, α=32, dropout=0.05 |
| LoRA targets | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Training objective | CrossEntropyLoss (next-token prediction) |
| FL weight exchange | LoRA tensors only (~3–8 MB/round for 1B model; ~40 MB for 8B) |
| Max sequence length | 512 tokens |
| Batch size | 4 (gradient accumulation = 4 steps) |
| Learning rate | 2e-4 |
| Prompt format | Alpaca-style with Portuguese response separator (fine-tuned on Brazilian clinical records) |

---

## Metrics

### ICD-10 Coding (Experiment A)

All metrics follow Mullenbach et al. (2018) and are computed on the held-out test split.

| Metric | Definition |
|--------|-----------|
| **Micro-F1 @ k** | F1 computed over all (sample, label) pairs after thresholding at top-k predictions per sample |
| **Macro-F1 @ k** | F1 averaged per label (equal weight to rare and frequent codes) |
| **AUC-ROC (micro)** | Area under ROC curve, micro-averaged over all labels |
| **P @ k** (Precision@k) | Fraction of the k predicted labels that are in the ground-truth label set |
| **R @ k** (Recall@k) | Fraction of the ground-truth labels recovered in the top-k predictions |

> **Note:** k=5 for MIMIC-IV-50; k=8 for MIMIC-IV-full (Mullenbach 2018 standard).

### Discharge Summary (Experiment B)

| Metric | Definition |
|--------|-----------|
| **ROUGE-1** | Unigram overlap (F1) between generated and reference summary |
| **ROUGE-2** | Bigram overlap (F1) |
| **ROUGE-L** | Longest common subsequence F1 (captures sentence-level structure) |
| **BERTScore (F1)** | Semantic similarity using `microsoft/BiomedNLP-BiomedBERT-base` token embeddings |
| **LLM-judge ensemble** | Mean score from Qwen2.5-72B + Gemma-3-27B + DeepSeek-R1 via OpenRouter; 5 dimensions: accuracy, completeness, conciseness, coherence, medical_safety; range 1–5 |

### Statistical Rigour

- **Confidence intervals:** 95% bootstrap CI (n=1000 resamples) over 3 random seeds (42, 43, 44).
- **Significance testing:** Wilcoxon signed-rank test (centralised vs federated, and no-DP vs DP); p<0.05.
- **Reported format:** `mean ± margin [95% CI: lower–upper] (n=3 seeds)`.

---

## Quickstart

### Prerequisites

| Dependency | Min version | Purpose |
|------------|-------------|---------|
| [uv](https://docs.astral.sh/uv/) | 0.5+ | Python package management |
| NVIDIA GPU + CUDA ≥ 12.4 | **≥ 12 GB VRAM** | Model training (ai_client only) |
| Python | 3.13 | Runtime (managed by uv) |
| Docker + Docker Compose | 24.0 / 2.20 | Optional — for containerised deployment |
| NVIDIA Container Toolkit | 1.14 | Optional — GPU passthrough in Docker |
| HuggingFace token | — | Optional — only required for Experiment B (gated Llama-3 model) |

### 1. Clone and install

```bash
git clone https://github.com/FBRosito/federated-fhir-architecture.git
cd federated-fhir-architecture
uv sync --frozen
```

### 2. Configure tokens (Experiment A requires no tokens)

```bash
# HuggingFace token — only required for Experiment B (gated Llama-3 model)
export HF_TOKEN=hf_your_token_here

# OpenRouter API key — only required for LLM-as-judge post-evaluation (Experiment B)
export OPENROUTER_API_KEY=sk-or-your_key_here
```

### 3. Run the smoke test

The smoke test runs all 12 experiment configurations with 5 silos, 2 FL rounds, seed=42, and 20 examples per silo. Expected runtime: ~60–90 minutes on RTX 3060 12 GB (requires CUDA).

```bash
bash run_experiments.sh --smoke --exp all 2>&1 | tee experiment_logs/smoke.log
grep -E "completed|Traceback|Error" experiment_logs/smoke.log
```

After the smoke test, verify JSON outputs:

```bash
ls experiment_logs/*.json   # 12 JSON files expected
python3 -c "import json; d=json.load(open('experiment_logs/fl_fedprox_alpha0.5_nodp_bert_seed42.json')); print(d['final_metrics'])"
```

### 4. Full stack with Docker Compose

```bash
# Start infrastructure (FHIR R4 server + fl_server + etl_worker)
make up-infra

# Start AI clients (foreground — live logs)
make up-ai

# Monitor all logs
make logs

# Tear down
make down
```

### 5. Run paper experiments (Experiment A)

```bash
# Paper experiment: 3 seeds × 6 configurations (FedProx/FedAvg × no-DP/σ=0.5/1.0/2.0)
bash run_nodocker.sh --exp A

# Or with Docker
bash run_experiments.sh --exp A
```

See [`docs/deploy.md`](./docs/deploy.md) for GPU cloud deployment instructions.

### 6. Generate plots

```bash
python3 -c "
from evaluation.plots import curva_epsilon_vs_f1, curva_f1_vs_alpha
# (populate with your actual result values)
curva_epsilon_vs_f1([2.1, 4.3, 8.7], [0.71, 0.69, 0.64], output_path='figures/epsilon_vs_f1.pdf')
"
```

---

## Custom Experiment (No Script Needed)

`run_experiments.sh` automates the full 12-config × 3-seed paper reproduction matrix. For any single FL configuration, use env vars directly via `make`:

### 1. Configure `.env`

```bash
cp .env.example .env
# Edit .env — uncomment the preset block for Experiment A (BERT) or Experiment B (LLM)
```

### 2. Start infrastructure

```bash
make up-infra   # starts FHIR R4 server + fl_server + etl_worker; blocks until all healthy
                # auto-triggers ETL reload if FHIR has fewer than 1000 patients
```

### 3. Choose backend and run

```bash
# Experiment A — ICD-10 coding / PubMedBERT, FedProx, no DP, 5 rounds
make run-bert ROUNDS=5 STRATEGY=fedprox NOISE=0.0

# Experiment B — Discharge summary / Llama-3.2, FedProx, moderate DP
make run-llm ROUNDS=5 STRATEGY=fedprox NOISE=1.0

# FedAvg comparison, strong DP, 3 silos
make run-bert ROUNDS=10 STRATEGY=fedavg NOISE=2.0 N_SILOS=3
```

Or with raw `docker compose` (all env vars inline):

```bash
MODEL_BACKEND=bert FL_STRATEGY=fedprox FL_NOISE_MULTIPLIER=0.0 \
FL_NUM_ROUNDS=5 FL_MIN_CLIENTS=5 FL_PROXIMAL_MU=0.01 FL_SEED=42 \
docker compose up fl_server ai_client_silo_0 ai_client_silo_1 \
  ai_client_silo_2 ai_client_silo_3 ai_client_silo_4
```

---

## Environment Variables

### FL Server

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_SERVER_ADDRESS` | `[::]:9091` | gRPC listen address |
| `FL_NUM_ROUNDS` | `5` | Number of FL rounds |
| `FL_MIN_CLIENTS` | `2` | Minimum clients before round starts |
| `FL_STRATEGY` | `fedprox` | Aggregation strategy (`fedprox` or `fedavg`) |
| `FL_PROXIMAL_MU` | `0.01` | FedProx proximal term μ |
| `FL_ROUND_TIMEOUT` | `3600` | Per-round timeout (seconds) |
| `FL_NOISE_MULTIPLIER` | `0.9` | DP-SGD noise multiplier σ — 0.0=no DP, 0.5=weak, 1.0=moderate, 2.0=strong |
| `FL_INITIAL_CLIP_NORM` | `1.0` | Gradient clipping norm C₀ (initial value for adaptive clipping) |
| `FL_CLIP_NORM_TARGET_Q` | `0.5` | Adaptive clipping target quantile |
| `FL_FRACTION_FIT` | `1.0` | Fraction of clients sampled per training round |
| `FL_LEARNING_RATE` | `5e-5` | Local learning rate sent to clients in fit_config |
| `FL_NUM_EPOCHS` | `1` | Local training epochs per FL round |
| `FL_EVAL_ACCURACY` | `false` | Enable per-round ICD-10 accuracy metric |

### AI Client

| Variable | Default | Description |
|----------|---------|-------------|
| `FHIR_SERVER_URL` | `http://localhost:8080/fhir` | FHIR R4 server base URL (Tier 1) |
| `FL_SERVER_ADDRESS` | `fl-server:9091` | Orchestration Tier gRPC address |
| `MODEL_BACKEND` | `bert` | Experiment selector: `bert` (Exp A — paper), `llm` (Exp B — extension), `llm-summarization` |
| `MODEL_NAME` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | Any HuggingFace model ID |
| `HF_TOKEN` | — | HuggingFace token (optional — only for gated Llama models, Exp B) |
| `MAX_SEQ_LEN` | `512` | Max token sequence length |
| `ETL_PARTITION_ID` | `0..4` (per silo) | Dirichlet partition assigned to this silo |
| `FL_BATCH_SIZE` | `8` | Local training batch size |
| `FL_GRADIENT_ACCUM_STEPS` | `0` | Gradient accumulation steps (0 = model default) |
| `FL_MAX_EXAMPLES` | `0` | Cap training examples per round (0 = no limit) |
| `FL_NOISE_MULTIPLIER` | `0.0` | DP noise multiplier (must match FL_SERVER value) |
| `FL_TARGET_DELTA` | `1e-5` | DP δ for RDP accountant (< 1/N recommended) |
| `FL_DP_SUBSAMPLE_RATE` | `0.1` | DP subsampling rate q |
| `FL_PROXIMAL_MU` | `0.01` | FedProx μ — client-side proximal regularization term |
| `FL_SEED` | `42` | Random seed for reproducibility |
| `FL_KEEP_MODEL_IN_VRAM` | `false` | Keep model loaded between rounds (faster dev runs) |
| `FL_PARALLEL_GPU` | `false` | `false` = serialize silos (1 GPU); `true` = parallel (multi-GPU) |
| `GPU_LOCK_PATH` | `/var/gpu_sync/gpu.lock` | Flock file path for GPU serialization |
| `FL_EVAL_ACCURACY` | `false` | Enable per-round ICD-10 accuracy metrics |
| `FL_TOP_K` | `5` | Top-K predictions for ICD-10 evaluation |
| `BERT_BENCHMARK` | `full` | ICD-10 benchmark subset: `top50`, `full`, or `none` |
| `BERT_LABEL_INDEX_PATH` | `/app/data/label_index.json` | ICD-10 label index file |
| `FL_SAVE_CHECKPOINT` | `""` | Directory to save LoRA adapters after the final round |
| `FL_CALIBRATE_GRAD_NORM` | `false` | Gradient norm calibration mode (outputs max_grad_norm) |

### ETL Worker

| Variable | Default | Description |
|----------|---------|-------------|
| `FHIR_SERVER_URL` | `http://localhost:8080/fhir` | FHIR R4 server target URL |
| `ETL_PARTITION_ID` | `-1` | Partition to load (-1 = all) |
| `ETL_DATA_PATH` | `etl_worker/data/clinical_evolutions.csv` | Input CSV path |

### Evaluation

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | OpenRouter API key for LLM-as-judge |
| `METRICS_LOG_DIR` | `evaluation/logs` | Directory for CSV metrics output |

---

## Expected Results

Results from the paper (MIMIC-IV v3.1, top-50 ICD-10 subset, R=20 rounds, K=5 silos, α=0.5, seeds 42/43/44).

### Experiment A — ICD-10 Coding *(paper results)*

| Configuration | Micro-F1@5 | Notes |
|---------------|-----------|-------|
| Centralised (upper bound, no federation) | 0.409 | Non-federated baseline |
| FL FedProx, no DP (σ=0) | 0.315 | 77% of centralised |
| FL FedAvg, no DP (σ=0) | 0.311 | Comparable to FedProx |
| FL FedProx + DP σ=0.5 (ε≈4.38) | 0.115 | Light privacy |
| FL FedProx + DP σ=1.0 (ε≈2.29) | 0.109 | Moderate privacy |
| FL FedProx + DP σ=2.0 (ε≈17.37) | 0.102 | Strong privacy (WOR subsampling) |

> The 63.5–67.6% relative Micro-F1 drop under DP is dominated by per-sample gradient clipping (C₀=1.0), not noise injection.
> All results: mean over 3 seeds; 95% bootstrap CI (B=10,000 resamples).

### Experiment B — Discharge Summary *(extension, not evaluated in the paper)*

| Configuration | ROUGE-1 | ROUGE-L | BERTScore-F1 | LLM-judge |
|---------------|---------|---------|-------------|-----------|
| Centralised (no FL) | ~0.38–0.45 | ~0.30–0.36 | ~0.82–0.87 | ~3.6–4.0 |
| FedProx (α=0.5, no DP) | ~0.36–0.43 | ~0.28–0.34 | ~0.80–0.85 | ~3.4–3.8 |
| FedProx (α=0.5, σ=1.0) | ~0.31–0.38 | ~0.24–0.30 | ~0.76–0.81 | ~3.0–3.5 |

> Experiment B numbers are from preliminary runs on `experiment_results_vastai/` and are not reported in the paper.

---

## Privacy and Differential Privacy

### DP-SGD Mechanism

The system applies **client-side DP-SGD** (Opacus) before any data leaves the edge node:

1. During local training, Opacus clips per-sample gradients of the LoRA adapter layers to L2 norm C₀ (default 1.0, set to a literature-based constant independent of the data — Yu et al., 2022; Anil et al., 2022).
2. Calibrated Gaussian noise N(0, σ²C₀²I) is injected into the aggregated LoRA gradients **before the update leaves the silo**.
3. The FL server receives only privatised LoRA deltas and acts as a clean FedProx aggregator — it adds no noise of its own.
4. Privacy budget (ε, δ) is tracked per-round at the client using the **RDP accountant** (Mironov 2017); the cumulative ε over all rounds is the value reported in the paper.

The base model weights (frozen NF4 parameters) are excluded from Opacus via `requires_grad=False` and never perturbed.

**DP parameters (default):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| σ (noise multiplier) | 0.9 | Noise scale relative to clipping norm |
| C₀ (clipping norm) | 1.0 | Per-sample gradient L2 clipping threshold |
| q (subsampling rate) | 0.1 | Poisson subsampling rate for RDP composition |
| δ (failure probability) | 1e-5 | Should be < 1/N (N = dataset size) |

### Privacy Budget per Configuration

The ε values below are computed by the RDP accountant after R=20 FL rounds with WOR subsampling rate q=batch/n_local (cumulative ε over all rounds):

| σ | Rounds | ε (δ=1e-5) | Interpretation |
|---|--------|-----------|----------------|
| 0.0 | any | ∞ | No privacy |
| 0.5 | 20 | ≈4.38 | Light DP |
| 1.0 | 20 | ≈2.29 | Moderate DP |
| 2.0 | 20 | ≈17.37 | Strong DP (WOR accounting) |

### Gradient Inversion Evaluation

The DLG attack (Zhu et al., 2019) is run in `evaluation/src/evaluation/run_gradient_inversion.py` to quantify how much patient text can be reconstructed from gradients:

```bash
python -m evaluation.run_gradient_inversion \
    --fhir-url http://localhost:8080/fhir \
    --sigma-values 0.0 0.5 1.0 2.0 \
    --n-samples 10 \
    --output experiment_logs/gradient_inversion.json
```

Expected result: ROUGE-1 reconstruction quality drops from ~0.25–0.35 (σ=0, no DP) to < 0.05 (σ≥1.0), demonstrating that DP noise effectively prevents clinical text reconstruction.

---

## Repository Structure

```
federated-fhir-architecture/
├── fl_server/
│   ├── src/fl_server/
│   │   ├── __init__.py          # Entry point: uv run fl-server
│   │   └── server.py            # FedProx + adaptive DP strategy
│   ├── Dockerfile
│   └── pyproject.toml
│
├── ai_client/
│   ├── src/ai_client/
│   │   ├── __init__.py          # Entry point: uv run ai-client
│   │   ├── fl_client.py         # FHIRFederatedClient (NumPyClient)
│   │   ├── model_setup_bert.py  # PubMedBERT + PLM-ICD + LoRA (Exp A — paper)
│   │   ├── model_setup.py       # Llama NF4 4-bit + LoRA (Exp B — extension)
│   │   ├── fhir_consumer_bert.py          # FHIR → BERTInput (Exp A)
│   │   ├── fhir_consumer.py               # FHIR → TrainingExample (Exp B/LLM)
│   │   ├── fhir_consumer_summarization.py # FHIR → SummarizationExample (Exp B)
│   │   └── centralized_baseline.py        # Non-FL baseline for comparison
│   ├── Dockerfile
│   └── pyproject.toml
│
├── etl_worker/
│   ├── src/etl_worker/
│   │   ├── __init__.py          # Entry point: uv run etl-worker
│   │   └── etl_pipeline.py      # CSV → FHIR Transaction Bundles
│   ├── mimic_builder.py         # MIMIC-IV v3.1 → FHIR bundles (production)
│   ├── generate_dataset.py      # Synthetic dataset generator (development)
│   └── pyproject.toml
│
├── evaluation/
│   ├── src/evaluation/
│   │   ├── metrics_logger.py        # FederatedRunLogger + GPUTimer
│   │   ├── icd_metrics.py           # Mullenbach 2018 metrics (micro-F1, AUC-ROC)
│   │   ├── summarization_metrics.py # ROUGE-1/2/L + BERTScore(PubMedBERT)
│   │   ├── llm_judge.py             # LLM-as-judge ensemble (Exp B extension)
│   │   ├── statistical_analysis.py  # Bootstrap CI + Wilcoxon test
│   │   ├── plots.py                 # Publication figures (IEEE style)
│   │   ├── fhir_benchmark.py        # FHIR server latency + completeness benchmark
│   │   └── gradient_inversion.py    # DLG attack implementation (Zhu et al. 2019)
│   └── pyproject.toml
│
├── docs/
│   ├── architecture.md             # System architecture documentation
│   └── deploy.md                   # GPU cloud deployment guide
├── run_experiments.sh           # Full experiment orchestration (12 configs)
├── Makefile                     # make setup / up-infra / up-ai / logs / down / clean
├── docker-compose.yml           # Multi-service stack definition
├── pyproject.toml               # uv workspace root
├── uv.lock                      # Locked dependency versions
└── LICENSE                      # MIT License
```

---

## License

This project is released under the [MIT License](LICENSE).

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for code style,
testing, and the pull-request process. By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Security

This repository must never contain patient data or credentials. If you find a
security or privacy issue (including accidentally committed sensitive data),
please follow the responsible-disclosure process in [SECURITY.md](SECURITY.md).

## Citation

{% raw %}
```bibtex
@inproceedings{rosito2025herald,
  title     = {{HERALD}: Healthcare fEderated leaRning Architecture with {LoRA}
               and Differential-privacy},
  author    = {Rosito, Fernando Barcelos and Franco, Muriel Figueredo and
               Cazella, Silvio César},
  booktitle = {Proceedings of the IEEE International Conference on e-Health Networking,
               Application and Services (Healthcom)},
  year      = {2025},
  note      = {Under review},
}
```
{% endraw %}

---

## References

- McMahan, H. B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Li, T. et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).* MLSys.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Huang, C.-W. et al. (2022). *PLM-ICD: Automatic ICD Coding with Pretrained Language Models.* ACL BioNLP.
- Mullenbach, J. et al. (2018). *Explainable Prediction of Medical Codes from Clinical Text.* NAACL.
- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* FnTCS.
- Mironov, I. (2017). *Rényi Differential Privacy of the Gaussian Mechanism.* CSF.
- Hsu, T.-M. H. et al. (2019). *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.* NeurIPS FedML Workshop.
- Zhu, L. et al. (2019). *Deep Leakage from Gradients.* NeurIPS.
- Johnson, A. E. W. et al. (2023). *MIMIC-IV, a freely accessible electronic health record dataset.* Scientific Data.
- HL7 International. *FHIR R4 Specification.* hl7.org/fhir/R4.
- Bai, X. et al. (2024). *Qwen2.5 Technical Report.* arXiv.
