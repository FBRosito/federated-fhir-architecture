# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Strict Architecture and Engineering Rules

The guidelines below are **non-negotiable**. Every code change must respect them.

1. **Package Management — `uv` only.**
   Use EXCLUSIVELY `uv` for dependency management and virtual environments. Never suggest or use `pip`, `conda`, or `poetry`.

2. **Separation of Concerns (SoC).**
   Maintain strict separation between `etl_worker` (data pipeline), `ai_client` (AI client), and `fl_server` (orchestrator). The `ai_client` **must never** process raw text — it consumes only structured HL7 FHIR resources via the HAPI FHIR server API.

3. **Hardware Constraint — 12 GB VRAM GPU.**
   All AI solutions must run locally without OOM on a single 12 GB VRAM GPU. Prefer lazy loading, 4-bit quantization (NF4 via BitsAndBytes), and PEFT/LoRA. Never load the full model in full precision.

4. **Privacy and Federated Learning — FedProx + Client-Side DP.**
   The framework uses FedProx with client-side Differential Privacy (Opacus, per-sample gradient clipping + Gaussian noise applied ONLY to LoRA adapter layers before weights leave the edge node). The FL server is a clean FedProx aggregator with no DP noise injection. Do not alter this architectural foundation.

## Project Overview

Privacy-Preserving Federated e-Health Architecture for clinical NLP over FHIR-standardized data.
Four canonical tiers:

**Tier 1 — Data Interoperability:** HAPI FHIR (HL7 R4, port 8080) + ETL Worker. Transforms MIMIC-IV CSVs into FHIR Transaction Bundles. Any FHIR-compliant hospital can join without changing AI code.

**Tier 2 — Federated Orchestration:** Flower SuperLink (gRPC, port 9091). Star topology: distributes global LoRA adapter, aggregates privatized client updates via FedProx (μ=0.01). Zero patient data contact.

**Tier 3 — Edge Computation (Silos):** AI Clients (silo_0..4). Foundation model (Llama-3.2-1B NF4 4-bit or PubMedBERT-base) stays in local GPU memory. Only LoRA deltas (~5–32 MB, optionally TurboQuant-compressed) traverse the network.

**Tier 4 — Cross-Cutting Privacy:** Opacus DP-SGD applied client-side to LoRA gradients only (frozen base excluded via `requires_grad=False`). Gradient clipping C₀ fixed after calibration round; RDP accountant tracks ε per round.

Data flows: CSV → ETL → HAPI FHIR → AI Clients ←→ FL Server (gRPC)

## Build & Run Commands

Package manager: **uv** (Python 3.13)

```bash
# Recommended workflow via Makefile
make setup          # uv sync — creates/updates .venv
make up-infra       # starts HAPI FHIR + fl_server + etl_worker; waits for infra to be ready
make up-ai          # starts ai_client_silo_0 and ai_client_silo_1 in foreground (live logs)
make logs           # docker compose logs -f (all containers)
make down           # stops all containers
make clean          # down -v + removes .venv and Python caches (full reset)

# Direct commands (without Makefile)
uv sync --frozen              # Install dependencies from lockfile
uv run fl-server              # Start the Flower FL orchestrator
uv run ai-client              # Start an FL client (requires GPU for training)
uv run etl-worker             # Run ETL pipeline (CSV → FHIR)
uv run evaluation             # Record per-round metrics
docker compose up             # Full stack: HAPI FHIR + all services
```

ETL with custom args:
```bash
uv run python etl_worker/etl_pipeline.py --data etl_worker/data/clinical_evolutions.csv --partition 0 --fhir-url http://localhost:8080/fhir --dry-run
```

No test suite or linter is currently configured.

## Architecture

**Monorepo workspace** with four service packages under `[tool.uv.workspace]`:

- **`fl_server/`** — Flower ServerApp. Implements FedProx strategy (not FedAvg) with weighted aggregation. Listens on gRPC port 9091. Strategy choice is deliberate: FedProx handles Non-IID data by adding a proximal term that prevents client drift. The server is a clean aggregator — DP is applied client-side, not here.

- **`ai_client/`** — Flower NumPyClient. Key modules:
  - `fl_client.py` — `FHIRFederatedClient(NumPyClient)`: orchestrates fit/evaluate rounds, lazy-loads model & data; applies Opacus DP-SGD to LoRA layers; optionally compresses LoRA deltas via TurboQuant before transmission.
  - `model_setup.py` — Loads Llama-3.2-1B (NF4 4-bit, BitsAndBytes double quant, ~900 MB VRAM) or PubMedBERT-base (fp16, ~220 MB), applies LoRA via PEFT. Only LoRA weights are exchanged over the network.
  - `fhir_consumer.py` — Fetches Condition + DocumentReference from HAPI FHIR, joins by patient reference into training examples.
  - `turbocompress.py` — Two-stage TurboQuant compressor: random rotation + Lloyd-Max (Stage 1), QJL 1-bit residual (Stage 2). Reduces LoRA delta transmission 6.4× at 4-bit effective width.

- **`etl_worker/`** — Converts MIMIC-IV CSVs to FHIR Transaction Bundles (Patient + Condition + Composition + DocumentReference). Non-IID Dirichlet(α) partitioning. Chronological 80/10/10 split by `admittime` rank (not absolute year — MIMIC-IV v3.1 applies per-patient date-shifting).

- **`evaluation/`** — `FederatedRunLogger` records per-round metrics (loss, perplexity, ICD-10 F1, ε_spent) to JSON. Uses scikit-learn for P@k/R@k/F1@k and AUC-ROC.

## Data Flow

```
ETL Worker --[HTTP POST FHIR Bundle]--> HAPI FHIR Server (port 8080)
AI Client  --[HTTP GET paginated]-----> HAPI FHIR Server
AI Client  <--[gRPC NDArrays]---------> FL Server (port 9091)
```

All communication is synchronous HTTP/gRPC. No message brokers.

## Key Environment Variables

| Variable | Default | Used By |
|---|---|---|
| `FHIR_SERVER_URL` | `http://localhost:8080/fhir` | etl_worker, ai_client |
| `FL_SERVER_ADDRESS` | `[::]:9091` (server) / `fl_server:9091` (client) | fl_server, ai_client |
| `FL_NUM_ROUNDS` | `5` | fl_server |
| `FL_MIN_CLIENTS` | `2` | fl_server |
| `FL_STRATEGY` | `fedprox` | fl_server |
| `ETL_PARTITION_ID` | `-1` (all) | etl_worker, ai_client |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B` | ai_client |
| `MAX_SEQ_LEN` | `512` | ai_client |
| `FL_ROUND_TIMEOUT` | `3600` | fl_server |
| `MODEL_BASE_PRECISION` | `nf4` | ai_client (LLM only) |
| `FL_LORA_COMPRESS` | `false` | ai_client (LLM only) |
| `FL_COMPRESS_BITS` | `4` | ai_client (LLM only) |
| `OPENROUTER_API_KEY` | *(required for LLM judges)* | evaluation/post_eval (Exp B) |
| `FL_USE_LITERATURE_C0` | `true` | run_experiments.sh |

> **`FL_USE_LITERATURE_C0=true` (default, required for formal DP):** C₀=1.0 based on
> Yu et al. (2022) / Anil et al. (2022). Calibration still runs but its output is
> informational only — it does NOT feed into `FL_MAX_GRAD_NORM` for DP training runs.
> Setting this to `false` makes C₀ data-dependent, invalidating the formal DP guarantee.

## DP Privacy Budget Notes

- **`epsilon_spent`** in experiment JSONs = per-round ε (single round, fresh accountant).
- **`epsilon_cumulative`** = true privacy cost; RDP composition over all rounds so far.
  This is the value to report in the paper. With 5 rounds at σ=1.0, q=0.1, the
  cumulative ε is roughly 2× the per-round value.
- **Neighboring relation**: admission-level DP. Adding/removing one hospital admission
  (all associated DocumentReferences, Conditions, and Patient demographics) constitutes
  a neighboring dataset. Patients with multiple admissions contribute independently.
- **C₀ data-independence**: C₀=1.0 is declared before any data is seen. The calibration
  script measures gradient norms as a sanity check only; its output never modifies C₀
  in a formally-DP run (`FL_USE_LITERATURE_C0=true`).

## Code Conventions

- Each service has `src/<package>/__init__.py` with a `main()` entry point registered in its `pyproject.toml`
- All code, comments, docstrings, and log messages are in English
- **Exception:** Portuguese prompt format strings (`"### Resposta:\n"`, `"### Instrução:\n"`, `"### Contexto clínico:\n"`) in `fhir_consumer.py` and `fhir_consumer_summarization.py` are intentional — they match the prompt format the model was trained on and must not be translated
- **Exception:** ICD10_MAP keys in `etl_pipeline.py` are Portuguese diagnosis names that must match the `raw_diagnosis` column of the input CSV exactly
- Design decisions are documented in `docs/decisoes_de_arquitetura.md`
- Docker builds use workspace root as context to access the root `pyproject.toml`
