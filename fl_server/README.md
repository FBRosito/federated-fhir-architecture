# fl_server — Federated Learning Orchestrator

Flower SuperLink server managing the star topology. Coordinates FL rounds, distributes global LoRA weights to silos, and aggregates privatized LoRA deltas. Never accesses patient data.

## Entry point

```bash
uv run fl-server
```

## Key source files

| File | Purpose |
|------|---------|
| `src/fl_server/server.py` | FedProx and FedAvg strategy implementations; round lifecycle |

## Aggregation strategies

- **FedProx** (default, `FL_STRATEGY=fedprox`): Proximal regularization μ=0.01 penalizes silo models that deviate too far from the global model — more robust under Non-IID data distribution.
- **FedAvg** (`FL_STRATEGY=fedavg`): Standard weighted average (μ=0).

The server is a **clean aggregator** — it adds no DP noise server-side. All privacy guarantees are enforced at the silo (Tier 3 / Tier 4).

## Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_SERVER_ADDRESS` | `[::]:9091` | gRPC listen address |
| `FL_NUM_ROUNDS` | `20` | Number of FL rounds (paper uses 20) |
| `FL_MIN_CLIENTS` | `5` | Minimum silos before a round starts |
| `FL_STRATEGY` | `fedprox` | Aggregation strategy (`fedprox` or `fedavg`) |
| `FL_PROXIMAL_MU` | `0.01` | FedProx proximal term μ |
