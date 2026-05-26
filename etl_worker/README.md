# etl_worker — ETL Pipeline

Transforms MIMIC-IV CSV files into HL7 FHIR R4 Transaction Bundles and loads them into the FHIR R4 server. Handles Non-IID Dirichlet partitioning across silos. Runs once at experiment setup — not during FL rounds.

## Entry point

```bash
uv run etl-worker
```

## Key source files

| File | Purpose |
|------|---------|
| `src/etl_worker/etl_pipeline.py` | Loads pre-built FHIR bundles from disk into the FHIR server |
| `mimic_builder.py` | Reads MIMIC-IV CSVs → builds FHIR bundles with Dirichlet partitioning |
| `generate_dataset.py` | Generates synthetic clinical data for smoke tests (not paper experiments) |

## Workflow

1. **Build bundles** (once, from MIMIC-IV CSV): `make build-mimic` → writes `data/bundles/` + `data/label_index.json`
2. **Load into FHIR server**: `uv run etl-worker` → POSTs bundles to `FHIR_SERVER_URL`

## Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FHIR_SERVER_URL` | `http://localhost:8080/fhir` | FHIR R4 server to load data into |
| `ETL_BUNDLES_PATH` | `etl_worker/data/bundles` | Directory containing pre-built FHIR bundles |
| `ETL_DIRICHLET_ALPHA` | `0.5` | Non-IID partitioning α (paper uses 0.5) |
| `MIMIC_HOSP_DIR` | — | Path to MIMIC-IV `hosp/` directory (for `mimic_builder.py`) |
