# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Regras Estritas de Arquitetura e Engenharia

As diretrizes abaixo são **inegociáveis**. Toda alteração de código deve respeitá-las.

1. **Gerenciamento de Pacotes — somente `uv`.**
   Use EXCLUSIVAMENTE o `uv` para gestão de dependências e ambientes virtuais. Nunca sugira ou use `pip`, `conda` ou `poetry`.

2. **Separação de Contextos (SoC).**
   Mantenha a separação estrita entre `etl_worker` (pipeline de dados), `ai_client` (cliente de IA) e `fl_server` (orquestrador). O `ai_client` **nunca** deve fazer processamento de texto bruto — ele consome apenas recursos HL7 FHIR estruturados via API do servidor HAPI FHIR.

3. **Restrição de Hardware — GPU 12 GB VRAM.**
   Todas as soluções de IA devem rodar localmente sem OOM em uma única GPU de 12 GB de VRAM. Privilegie lazy loading, quantização 4-bit (NF4 via BitsAndBytes) e PEFT/LoRA. Nunca carregue o modelo completo em precisão total.

4. **Privacidade e Federated Learning — FedProx + DP Adaptativa.**
   O framework utiliza a estratégia FedProx com Privacidade Diferencial Adaptativa (server-side adaptive clipping) para lidar com a distribuição Non-IID dos dados clínicos. Não altere essa fundação arquitetural.

## Project Overview

Federated Learning system for clinical NLP over FHIR-standardized healthcare data. Trains Llama-3-8B with LoRA adapters across distributed nodes using Flower, with differential privacy (server-side adaptive clipping). Data flows: CSV → ETL → HAPI FHIR → AI Clients → FL Server aggregation.

## Build & Run Commands

Package manager: **uv** (Python 3.13)

```bash
uv sync --frozen              # Install all dependencies from lockfile
uv run fl-server              # Start Flower FL orchestrator
uv run ai-client              # Start FL client (needs GPU for training)
uv run etl-worker             # Run ETL pipeline (CSV → FHIR)
uv run evaluation             # Run evaluation/metrics logger
docker-compose up             # Full stack: HAPI FHIR + all services
```

ETL with custom args:
```bash
uv run python etl_worker/etl_pipeline.py --data etl_worker/data/clinical_evolutions.csv --partition 0 --fhir-url http://localhost:8080/fhir --dry-run
```

No test suite or linter is currently configured.

## Architecture

**Monorepo workspace** with four service packages under `[tool.uv.workspace]`:

- **`fl_server/`** — Flower ServerApp. Implements FedProx strategy (not FedAvg) with DP-adaptive clipping wrapper. Listens on gRPC port 9091. Strategy choice is deliberate: FedProx handles Non-IID data by adding a proximal term that prevents client drift.

- **`ai_client/`** — Flower NumPyClient. Three key modules:
  - `fl_client.py` — `FHIRFederatedClient(NumPyClient)`: orchestrates fit/evaluate rounds, lazy-loads model & data
  - `model_setup.py` — Loads Llama-3-8B quantized to NF4 4-bit via BitsAndBytes, applies LoRA (rank=16) targeting all attention + FFN projections. Only LoRA weights (~24M params) are exchanged over the network, not the 8B base model.
  - `fhir_consumer.py` — Fetches Condition + DocumentReference from HAPI FHIR, joins by patient reference into training examples (clinical text → ICD-10 code)

- **`etl_worker/`** — Converts clinical CSV to FHIR Transaction Bundles (Patient + Condition + Composition + DocumentReference). Implements Non-IID partitioning by medical specialty (cardiology, pneumology, endocrinology, general).

- **`evaluation/`** — `FederatedRunLogger` records per-round metrics (loss, perplexity, ICD-10 accuracy) to CSV. Uses scikit-learn for precision/recall/F1.

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
| `MODEL_NAME` | `meta-llama/Meta-Llama-3-8B-Instruct` | ai_client |
| `MAX_SEQ_LEN` | `512` | ai_client |

## Code Conventions

- Each service has `src/<package>/__init__.py` with a `main()` entry point registered in its `pyproject.toml`
- Documentation and comments are in Portuguese (Brazilian)
- Design decisions are documented in `docs/decisoes_de_arquitetura.md`
- Docker builds use workspace root as context to access the root `pyproject.toml`
