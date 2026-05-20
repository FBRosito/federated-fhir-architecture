# Architecture Decision Records — FL-FHIR Architecture

**Project:** Federated Learning over Clinical Data in the FHIR Standard
**Repository:** `federated-fhir-architecture`
**Document version:** 3.0
**Date:** 2026-05-20

---

## Table of Contents

1. [FedProx + Client-Side Differential Privacy](#1-fedprox--client-side-differential-privacy)
   - 1.1 [The problem: Non-IID data in hospital environments](#11-the-problem-non-iid-data-in-hospital-environments)
   - 1.2 [Why FedProx and not FedAvg](#12-why-fedprox-and-not-fedavg)
   - 1.3 [Why client-side DP-SGD (Opacus)](#13-why-client-side-dp-sgd-opacus)
   - 1.4 [Threat model addressed](#14-threat-model-addressed)
   - 1.5 [Known limitations](#15-known-limitations)
2. [4-bit Quantised Llama-3 + LoRA](#2-4-bit-quantised-llama-3--lora)
   - 2.1 [Hardware constraint as a design requirement](#21-hardware-constraint-as-a-design-requirement)
   - 2.2 [NF4 quantisation via BitsAndBytes](#22-nf4-quantisation-via-bitsandbytes)
   - 2.3 [Why LoRA and not full fine-tuning](#23-why-lora-and-not-full-fine-tuning)
   - 2.4 [LoRA adapter configuration](#24-lora-adapter-configuration)
   - 2.5 [Communication efficiency in the federated setting](#25-communication-efficiency-in-the-federated-setting)
3. [References](#3-references)

---

## 1. FedProx + Client-Side Differential Privacy

### 1.1 The problem: Non-IID data in hospital environments

Real-world clinical data does not follow i.i.d. distributions across healthcare institutions. The diagnostic profile of a cardiology hospital is radically different from that of a pulmonology centre or a general emergency unit. In this system, that phenomenon is explicitly modelled by the `etl_worker`, which partitions data by medical speciality:

| Partition | Unit profile | Diagnostic prevalence | Typical ICD-10 codes |
|---|---|---|---|
| 0 | Hospital A — Cardiology | Cardiovascular (70%) | I10, I50.0, I20.0, I63.9 |
| 1 | Hospital B — Pulmonology | Respiratory (70%) | J44.1, J45.9, J18.9, I26.9 |
| 2 | Centre C — Endocrinology | Metabolic/Endocrine (70%) | E11.9, E03.9, E28.2, E21.0 |
| 3 | Unit D — General | Uniform distribution | M54.5, N39.0, F32.9, N18.3 |

This heterogeneity is not a simulation artefact — it reflects the reality of any healthcare network and is the primary technical challenge of Federated Learning applied to the medical domain.

### 1.2 Why FedProx and not FedAvg

**FedAvg** (McMahan et al., 2017) is the reference algorithm for FL. It assumes that client gradients converge, on average, to the gradient of the global objective. This assumption holds for i.i.d. data, but **breaks under Non-IID distributions**: each client optimises a different local loss function, and the simple average of gradients may point in conflicting directions — a phenomenon known as *client drift*.

**FedProx** (Li et al., 2020b) addresses this by adding a **proximal term** to each client's local loss:

```
L_FedProx(w) = L_local(w) + (μ/2) · ‖w − w_global‖²
```

The practical effect is that each client is penalised for deviating excessively from the global model. The hyperparameter μ controls the strength of this constraint:

- μ → 0: degrades to FedAvg (no constraint)
- μ → ∞: clients do not update (static global model)
- μ = 0.01 (adopted value): allows local adaptation with convergence stability

The choice of μ=0.01 is conservative and appropriate for the moderate-to-high degree of heterogeneity in the dataset. The value is forwarded to clients via `fit_config` at every round, ensuring the proximal term is correctly applied by the `ai_client` without requiring knowledge of the global server configuration.

**Code reference:**

```python
# fl_server/server.py — build_base_strategy()
return FedProx(proximal_mu=proximal_mu, **common_kwargs)
```

### 1.3 Why client-side DP-SGD (Opacus)

Differential Privacy is applied **at the client**, before any gradient or weight delta leaves the edge node. The mechanism uses Opacus (Yousefpour et al., 2021), which implements DP-SGD with RDP accounting:

**1. Per-sample clipping:** during the backward pass, Opacus intercepts each individual sample's gradient and projects it to maximum L2 norm `C₀ = 1.0`. The value of `C₀` is fixed before any contact with the data, based on the empirical results of Yu et al. (2022) and Anil et al. (2022) for LoRA-based language models. This ensures the **data independence** required for the formal `(ε, δ)`-DP guarantee.

**2. Gaussian noise:** after per-sample clipping and gradient aggregation over the batch, calibrated noise `N(0, σ²C₀²I)` with `σ = noise_multiplier` is injected into the LoRA layer gradients **before the optimisation step**.

**3. Base model exclusion:** the base model weights (Llama-3 in NF4 4-bit or PubMedBERT) have `requires_grad=False` and are fully excluded from the DP mechanism — only LoRA layers are protected, keeping the computational overhead proportional to the adapter size, not the full model.

**4. RDP accounting:** the RDP accountant (Mironov, 2017) tracks the accumulated privacy budget. Poisson subsampling at rate `q = 0.1` per round enables privacy amplification. The `epsilon_cumulative` value reported in each result JSON is the figure to cite in the paper.

```python
# ai_client/fl_client.py — Opacus DP-SGD application
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=lora_model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=n_epochs,
    target_epsilon=target_epsilon,
    target_delta=FL_TARGET_DELTA,
    max_grad_norm=C0,   # C₀ = 1.0 (literature-based, data-independent)
)
```

**Why client-side and not server-side?**

Server-side DP (added *after* aggregation) does not protect individual client gradients while they are in transit or visible to the server — a compromised server can inspect the deltas before applying noise. With client-side DP, noise is already embedded in the delta leaving the silo: even if the server is adversarial, it receives only privatised gradients. This is the strongest privacy guarantee under the federated threat model (Geyer et al., 2017; Wei et al., 2020).

The `fl_server` is a clean FedProx aggregator — it adds no noise and does not need to be trusted for the DP guarantee to hold.

### 1.4 Threat model addressed

**Gradient/model inversion attacks:** given access to the LoRA deltas sent by a client, an adversary may attempt to reconstruct the clinical text that produced them (Zhu et al., 2019). The Gaussian noise injected client-side corrupts the gradients before they leave the silo, making reconstruction infeasible for σ ≥ 1.0 — as demonstrated by the inversion experiments in `evaluation/src/evaluation/run_gradient_inversion.py`.

**Membership inference:** an adversary attempts to determine whether a specific patient record was used in training. The `(ε, δ)`-DP guarantee formally bounds the adversary's advantage. With σ=0.9, 5 rounds, and q=0.1, the cumulative ε is approximately 4–5 (δ=1e-5), providing moderate-to-strong protection — appropriate for de-identified clinical data.

**Adversarial server:** because noise is applied client-side, a compromised server inspecting the received deltas sees only already-privatised gradients — the DP guarantee does not depend on the server's integrity.

### 1.5 Known limitations

- **Secure Aggregation:** each client's LoRA delta arrives at the server in plaintext (protected only by noise). To prevent the server from reconstructing individual contributions by differencing across rounds, cryptographic Secure Aggregation would be required (Bonawitz et al., 2017). This extension significantly increases operational complexity and is outside the current scope.
- **Neighbouring relation — admission-level:** the DP guarantee covers the addition or removal of a complete hospital admission (all associated DocumentReference, Condition, and Patient resources). Patients with multiple admissions contribute independently — which is conservative and formally correct, but may underestimate the exposure of patients with extensive clinical histories.
- **C₀ literature vs. data:** `C₀ = 1.0` is declared before seeing any data (formal guarantee). The calibration script (`FL_CALIBRATE_GRAD_NORM=true`) measures real gradient norms as a sanity check; its output does not alter `C₀` in formally DP runs.

---

## 2. 4-bit Quantised Llama-3 + LoRA

### 2.1 Hardware constraint as a design requirement

The system is designed to run **locally on a single 12 GB VRAM GPU**, without relying on cloud infrastructure. This constraint is not arbitrary: it reflects the hardware available in mid-sized healthcare institutions and is what makes the system deployable in real-world scenarios.

Llama-3-8B in full precision (float32) requires ~32 GB of VRAM — infeasible. The combination of 4-bit quantisation + LoRA reduces this requirement to **~6–8 GB during training**, within the margin of a 12 GB GPU.

Estimated VRAM usage:

| Component | Precision | Approximate VRAM |
|---|---|---|
| Llama-3-8B (base weights, NF4 4-bit) | 4-bit NF4 | ~4.5 GB |
| Activations + KV cache (seq=512) | bfloat16 | ~1.5 GB |
| Trainable LoRA weights (~24M params) | bfloat16 | ~0.2 GB |
| LoRA gradients + AdamW states | float32 | ~1.0 GB |
| **Estimated total** | | **~7.2 GB** |

### 2.2 NF4 quantisation via BitsAndBytes

**NF4 (NormalFloat 4-bit)** quantisation represents each base model parameter in 4 bits using a normalised floating-point code optimised for neural network weight distributions (approximately Gaussian). Computation at inference and training time occurs in `bfloat16` — weights are dequantised on-the-fly per block before each matrix operation.

**Double quantisation** (`bnb_4bit_use_double_quant=True`): also quantises the per-block quantisation constants (which would otherwise remain in float32), saving an additional ~0.4 bits per parameter.

```python
# ai_client/model_setup.py — QuantizationConfig
BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",       # NormalFloat 4-bit
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)
```

The base model is loaded with `device_map="auto"`, letting BitsAndBytes allocate layers to the available GPU. Base weights are **fully frozen** (`requires_grad=False`) — only the LoRA adapters are trainable.

### 2.3 Why LoRA and not full fine-tuning

**Full fine-tuning** of an 8B-parameter LLM would require:
- ~32 GB VRAM for weights in float32
- ~64–96 GB for gradients + optimiser states (Adam: 2 moments × 32-bit)
- ~30 GB of data transmission per round per client in the federated setting

**LoRA** (Hu et al., 2022) decomposes the update of each weight matrix into a product of two low-rank matrices:

```
ΔW = A · B    where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), r ≪ min(d, k)
```

With `r = 16` (adopted rank), the number of trainable parameters drops from **8B to ~24M** (~0.3% of the total). Only matrices A and B are trained; the rest of the model remains frozen in NF4.

The effective scale of the adaptation is controlled by `lora_alpha`:

```
W_eff = W_base + (lora_alpha / r) · A · B = W_base + 2.0 · A · B
```

With `alpha = 32` and `r = 16`, the scaling factor is 2.0 — a standard value that balances training stability and the expressive capacity of the adaptation.

### 2.4 LoRA adapter configuration

```python
# ai_client/model_setup.py
LoraConfig(
    r            = 16,       # decomposition rank
    lora_alpha   = 32,       # effective scale = 32/16 = 2.0
    lora_dropout = 0.05,     # regularisation during training
    task_type    = TaskType.CAUSAL_LM,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # FFN (SwiGLU)
    ],
    modules_to_save = ["embed_tokens", "lm_head"], # fully adapted
)
```

**Why all 7 modules?** Covering only `q_proj` and `v_proj` (the minimal common configuration) suffices for style adaptation, but is insufficient for a new structured extraction task like ICD-10 coding. Including the FFN projections (`gate_proj`, `up_proj`, `down_proj`) increases the capacity to store factual medical domain knowledge in the intermediate layers.

**`modules_to_save`:** `embed_tokens` and `lm_head` are trained fully (not via LoRA) so that the model can learn the specific vocabulary of ICD-10 codes at the output layer.

### 2.5 Communication efficiency in the federated setting

The primary scalability bottleneck in FL is not computation — it is **communication**. Each round involves uploading and downloading weights between client and server. The choice of LoRA radically transforms that cost:

| Training mode | Parameters transmitted | Approximate size/round |
|---|---|---|
| Full fine-tuning (bfloat16) | 8B | ~16 GB |
| LoRA rank=16 (bfloat16) | ~24M | ~48 MB |
| **Reduction** | | **~333×** |

In practice, only the LoRA tensors are serialised as `NDArrays` and sent to the `fl_server` via gRPC. The Llama-3-8B base model is never transmitted — it is loaded independently on each node from the HuggingFace cache (`model_cache` Docker volume).

This design is fundamental for making the federated protocol viable on hospital networks with limited bandwidth.

---

## 3. References

- McMahan, H. B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Li, T. et al. (2020a). *Federated Learning on Non-IID Data Silos: An Experimental Study.* arXiv:2102.02079.
- Li, T. et al. (2020b). *Federated Optimization in Heterogeneous Networks.* MLSys 2020.
- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in Theoretical Computer Science.
- Mironov, I. (2017). *Rényi Differential Privacy of the Gaussian Mechanism.* CSF 2017.
- Geyer, R. C. et al. (2017). *Differentially Private Federated Learning: A Client Level Perspective.* NeurIPS Workshop.
- Wei, K. et al. (2020). *Federated Learning with Differential Privacy: Algorithms and Performance Analysis.* IEEE TIFS.
- Yousefpour, A. et al. (2021). *Opacus: User-Friendly Differential Privacy Library in PyTorch.* arXiv:2109.12298.
- Yu, D. et al. (2022). *Differentially Private Fine-Tuning of Language Models.* ICLR 2022.
- Anil, R. et al. (2022). *Large-Scale Differentially Private BERT.* EMNLP 2022.
- Bonawitz, K. et al. (2017). *Practical Secure Aggregation for Privacy-Preserving Machine Learning.* CCS 2017.
- Zhu, L. et al. (2019). *Deep Leakage from Gradients.* NeurIPS 2019.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- HL7 International. *FHIR R4 Specification.* https://hl7.org/fhir/R4.
