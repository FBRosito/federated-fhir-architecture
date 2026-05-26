# HERALD — System Architecture

**Healthcare fEderated leaRning Architecture with LoRA and Differential-privacy**

This document describes the technical architecture of HERALD, a privacy-preserving federated learning system for clinical ICD-10 coding over HL7 FHIR R4 data.

---

## Four-Tier Architecture

```
CSV (MIMIC-IV)
     │
     ▼  [Tier 1 — Data Interoperability]
┌─────────────────────────────────────┐
│          etl_worker                 │
│  MIMIC-IV → FHIR R4 Transaction     │
│  Bundles (Patient + Condition +     │
│  Composition + DocumentReference)  │
│  Dirichlet(α=0.5) Non-IID split     │
└──────────────────┬──────────────────┘
                   │ HTTP POST /fhir
                   ▼
     ┌─────────────────────────┐
     │     fhir_r4_server      │
     │  HL7 FHIR R4  :8080     │
     │  In-memory (stdlib)     │
     └──────┬──────────────────┘
            │ HTTP GET (paginated)
   [Tier 3 — Edge Computation ─────────────────────────────]
   │                                                        │
   ▼                      ▼                     ▼           ▼
┌──────────┐        ┌──────────┐         ┌──────────┐  ┌──────────┐
│ silo_0   │        │ silo_1   │         │ silo_2   │  │ silo_4   │
│ ai_client│        │ ai_client│         │ ai_client│  │ ai_client│
│ FHIR fetch│       │ FHIR fetch│        │ FHIR fetch│ │ FHIR fetch│
│ LoRA train│       │ LoRA train│        │ LoRA train│ │ LoRA train│
│ DP-SGD   │        │ DP-SGD   │         │ DP-SGD   │  │ DP-SGD   │
└────┬─────┘        └────┬─────┘         └────┬─────┘  └────┬─────┘
     │ [Tier 4 — DP clip + Gaussian noise before leaving silo]
     │ Noisy LoRA delta Δθ̃_k (gRPC)
     └──────────────┬────────────────────────────────────────┘
                    │ gRPC :9091
                    ▼  [Tier 2 — Federated Orchestration]
     ┌──────────────────────────────┐
     │         fl_server            │
     │  FedProx (μ=0.01) / FedAvg   │
     │  Clean aggregator            │
     │  (no server-side DP noise)   │
     │  Broadcasts θ_{t+1}          │
     └──────────────────────────────┘
```

---

## Tier 1 — Data Interoperability

**Components:** `fhir_server.py` + `etl_worker`

The ETL worker transforms MIMIC-IV CSV tables into HL7 FHIR R4 Transaction Bundles and loads them into the in-memory FHIR R4 server via HTTP POST. The FHIR server is a lightweight Python stdlib implementation (no Java, no external dependencies) exposing exactly the endpoints required by the AI clients: `Patient`, `Condition`, and `DocumentReference`.

This tier runs once at experiment setup. During FL rounds, silos query it read-only via paginated GET requests.

**Non-IID partitioning:** The ETL worker applies Dirichlet(α=0.5) partitioning over ICD-10 chapters to simulate realistic cross-hospital label heterogeneity. Each of the 5 silos receives a different speciality-skewed subset of admissions.

---

## Tier 2 — Federated Orchestration

**Components:** Flower SuperLink (`fl_server`)

The FL server manages the star topology via gRPC (port 9091). It never accesses patient data. Each round it:

1. Broadcasts the current global LoRA weights θ_t to all K=5 silos.
2. Waits for privatized LoRA deltas Δθ̃_k from each silo.
3. Computes a FedProx-weighted average to produce θ_{t+1}.
4. Begins the next round.

Two aggregation strategies are supported: **FedProx** (μ=0.01, proximal regularization for Non-IID robustness) and **FedAvg** (μ=0, standard averaging).

---

## Tier 3 — Edge Computation (Silos)

**Components:** `ai_client` (one process per silo)

Each silo runs inside a security boundary (simulated by process isolation). It:

1. Fetches its assigned FHIR partition via paginated REST calls (`/fhir/Condition`, `/fhir/DocumentReference`).
2. Builds the training dataset in memory (no disk writes of patient data).
3. Loads PubMedBERT with frozen base weights and LoRA adapters (r=8, α=16) targeting `query` and `value` attention layers.
4. Trains for E=1 local epoch under DP-SGD.
5. Sends only the LoRA delta Δθ̃_k to the FL server.

**Base model weights never leave the silo.** Raw patient data never leaves the silo.

---

## Tier 4 — Privacy Layer (Cross-Cutting)

**Components:** Opacus DP-SGD within each silo

DP-SGD is applied before any gradient information leaves Tier 3:

1. **Per-sample gradient clipping:** Each sample's gradient is clipped to L2 norm ≤ C₀=1.0. This bounds the maximum influence of any single patient on the model update.
2. **Gaussian noise injection:** Calibrated noise N(0, σ²C₀²I) is added to the clipped gradients.
3. **RDP accounting:** The privacy budget (ε, δ) is tracked per-round using the Rényi DP accountant. Cumulative ε is reported after R=20 rounds.
4. **WOR subsampling:** Without-replacement subsampling is used for tight RDP composition.

The FL server is a **clean aggregator** — it adds no server-side noise and sees only the already-privatized LoRA deltas.

The FedProx proximal term operates exclusively on trainable parameters (LoRA adapters + classification head); frozen base weights W₀ are unaffected.

---

## Communication Round (One FL Round t)

```
Server                         Silos (×5)
  │                               │
  │── broadcast θ_t ──────────────▶│
  │                               │
  │                     FHIR fetch (paginated GET)
  │                     local train E=1 epoch
  │                     DP-SGD: clip(C₀=1.0) + noise(σ)
  │                               │
  │◀── noisy LoRA delta Δθ̃_k ─────│
  │                               │
  FedProx/FedAvg aggregation
  θ_{t+1} = Σ_k w_k · (θ_t + Δθ̃_k)
  │
  (next round)
```

---

## Privacy Budget

Cumulative ε after R=20 rounds (δ=1e-5, C₀=1.0, WOR subsampling):

| σ | ε | Interpretation |
|---|---|----------------|
| 0.0 | ∞ | No privacy |
| 0.5 | ≈4.38 | Light DP |
| 1.0 | ≈2.29 | Moderate DP |
| 2.0 | ≈17.37 | Strong DP (WOR accounting) |

The dominant cost in Micro-F1 under DP comes from per-sample clipping (which truncates gradients even without noise) rather than from noise injection. This is confirmed by ablation: σ=0.5 and σ=2.0 produce similar F1 degradation (~63–68% relative drop) despite very different ε.

---

## Security Boundary

| Data | Leaves silo? |
|------|-------------|
| Raw patient text (DocumentReference) | **Never** |
| ICD-10 labels (Condition) | **Never** |
| Base model weights W₀ (PubMedBERT) | **Never** |
| LoRA adapter delta Δθ̃_k (privatized) | Yes — gRPC to FL server |
| Privacy budget (ε, δ) | Logged locally per silo |

No secure aggregation (e.g., SMPC or homomorphic encryption) is used in this prototype. The FL server sees each silo's noised delta individually, which is acknowledged as a limitation in the paper.

---

## Key Design Decisions

- **FHIR as the exclusive data access layer:** No silo reads data from CSV or any pre-processed format. All data flows through standardized FHIR R4 REST queries, making the system interoperable with any real hospital FHIR server.
- **Client-side DP (not server-side):** Noise is injected before gradients leave the silo. Server-side adaptive clipping (Flower's `DifferentialPrivacyServerSideAdaptiveClipping`) is available in the codebase but is not used in the paper — the rationale is explained in `fl_server/src/fl_server/server.py`.
- **LoRA-only weight exchange:** The frozen base model (110M parameters) is initialized identically on all silos. Only the LoRA adapter tensors (~15 MB) are exchanged per round, achieving a 99.9% bandwidth reduction.
- **No TurboQuant in paper experiments:** The `turbocompress.py` module implements an optional two-stage Lloyd-Max + QJL compression scheme available via `FL_LORA_COMPRESS=true`. All paper results use `FL_LORA_COMPRESS=false` (no compression).
