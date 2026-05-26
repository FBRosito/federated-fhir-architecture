# ai_client — Federated Learning Silo

Each silo runs one instance of this service. It fetches clinical data from the local FHIR R4 server, trains the model locally under DP-SGD, and sends only the privatized LoRA delta back to the FL server.

## Entry point

```bash
uv run ai-client
```

## Key source files

| File | Purpose |
|------|---------|
| `src/ai_client/fl_client.py` | `FHIRFederatedClient` — Flower NumPyClient implementation |
| `src/ai_client/model_setup_bert.py` | PubMedBERT + PLM-ICD + LoRA setup (Experiment A — paper) |
| `src/ai_client/model_setup.py` | Llama NF4 4-bit + LoRA setup (Experiment B — extension) |
| `src/ai_client/fhir_consumer_bert.py` | Fetches Condition + DocumentReference → BERT input tensors |
| `src/ai_client/fhir_consumer.py` | Fetches data for LLM training (Experiment B) |
| `src/ai_client/centralized_baseline.py` | Non-federated baseline (full-data training for comparison) |

## Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `bert` | `bert` = Experiment A (paper); `llm` = Experiment B |
| `FHIR_SERVER_URL` | `http://localhost:8080/fhir` | Local FHIR R4 server |
| `FL_SERVER_ADDRESS` | `localhost:9091` | Flower SuperLink gRPC address |
| `ETL_PARTITION_ID` | `0` | Silo index (0–4 for K=5 silos) |
| `FL_NOISE_MULTIPLIER` | `0.0` | DP-SGD noise σ (0=no DP, 0.5/1.0/2.0 for paper configs) |
| `FL_MAX_GRAD_NORM` | `1.0` | Per-sample gradient clipping norm C₀ |
| `BERT_BENCHMARK` | `top50` | ICD-10 subset: `top50` (paper) or `full` |
