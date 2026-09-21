# Experimental Readiness Audit — DP Budget-Equalization Paper

- **Date:** 2026-09-19 · **Branch:** `agent/repo-audit-equalization` · **Scope:** read-only audit (no training, no GPU jobs, no src/test/config modifications).
- **Method:** repo-wide grep/read + structural parsing of log JSONs + CPU-only imports. Key citations spot-verified by direct read.
- **Measured environment:** opacus 1.5.4, torch 2.6.0+cu124, flwr 1.27.0 (root `.venv`, `python -c "import …"`).
- **Incident during audit:** at 19:57 local, mid-audit, the untracked test files in `ai_client/tests/`, `etl_worker/tests/`, `evaluation/tests/`, `fl_server/tests/` were deleted by a process external to this audit (a `claude` session has been running on this machine since 19:11; dir mtimes 19:57; only `__pycache__/*.pyc` remain). Section 5 reflects what was measured before the deletion plus what remains.

---

## 1. Experiment inventory

### 1.1 Main pipeline (`experiment_logs/*.json`)

All 15 logs share schema `{tag, completed_at, config, per_round_eval, final_metrics, n_rounds_completed}`; `config = {backend, strategy, noise_multiplier, n_silos, seed, num_rounds, proximal_mu, max_grad_norm}`; `per_round_eval[i]` carries `micro_f1, macro_f1, auc_roc_micro/macro, P@8/15, R@8/15, F1@8/15, eval_loss, train_loss, n_samples=397.72, n_labels=50.0, epsilon_spent` (verified by parsing every file).

Launcher config (not stored in logs): BERT batch=8 / accum=8, `FL_NUM_EPOCHS=1`, FL rounds=20, central epochs=10, seeds `(42 43 44)` — `run_nodocker.sh:58-68`; code defaults q=0.1 (`ai_client/src/ai_client/fl_client.py:88`), C0=1.0 (`fl_client.py:92`), δ=1e-5 (`fl_client.py:86`). Dirichlet α=0.5 from `ETL_DIRICHLET_ALPHA=0.5` in experiment scripts.

| Config group | Algo | σ | C0 | q | R | E | batch/accum | Seeds | Slice | Labels | Results |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `fl_fedavg_alpha0.5_nodp_bert_seed{42,43,44}.json` | FedAvg | 0.0 | 1.0 | not in log (default 0.1) | 20 | 1 | 8/8 | 42,43,44 | MIMIC-IV α=0.5, 5 silos | 50 | Y — per-round + final |
| `fl_fedprox_alpha0.5_nodp_bert_seed{42,43,44}.json` | FedProx μ=0.01 | 0.0 | 1.0 | " | 20 | 1 | 8/8 | 42,43,44 | " | 50 | Y |
| `fl_fedprox_alpha0.5_dp0.5_bert_seed{42,43,44}.json` | FedProx μ=0.01 | 0.5 | 1.0 | " | 20 | 1 | 8/8 (DP path forces 1/1, `model_setup.py:473-481`) | 42,43,44 | " | 50 | Y (ε_spent≈43.09/round) |
| `fl_fedprox_alpha0.5_dp1.0_bert_seed{42,43,44}.json` | FedProx μ=0.01 | 1.0 | 1.0 | " | 20 | 1 | 8/8 | 42,43,44 | " | 50 | Y (ε_spent≈9.43/round, `experiment_results_vastai/.../dp1.0_seed42.json:32`) |
| `fl_fedprox_alpha0.5_dp2.0_bert_seed{42,43,44}.json` | FedProx μ=0.01 | 2.0 | 1.0 | " | 20 | 1 | 8/8 | 42,43,44 | " | 50 | Y |
| `centralizado_bert_seed{42,43,44}.json` | centralized (n_silos=1) | 0.0 | not set (no `max_grad_norm` key) | — | 10 (epochs) | 10 | 8/8 | 42,43,44 | ≤10000 ex. | 50 | Y (metrics final-only) |

Derived assets: `experiment_logs/statistical_summary.json` (n=3 means/CI, Wilcoxon), `experiment_logs/figures/{convergence,epsilon_vs_f1,f1_vs_alpha}.pdf`.
Copies/variants: `experiment_results_vastai/experiment_logs/` (21 logs: duplicates + `centralizado_llm_seed{42,43,44}.json` R=10); `experiment_logs/archive_smoke_20260505_111814/` (12 logs, seed 42 only, R=2, incl. `llm` backend; superseded smoke).

**FedAvg+DP configs in the main pipeline: NOT FOUND** (FedAvg logs are no-DP only). **FL runs at R=10: NOT FOUND** (R=10 occurs only in centralized logs).

### 1.2 `experiments/adaptive-clipping` (Article 2 — clipping strategies)

All runs: FedProx μ=0.01, δ=1e-5, q=0.1, LR=2e-4, batch=8/accum=8, 5 silos, α=0.5, top-50, R=20 (`scripts/run_baseline.sh:31-53`). Logs: 70 run dirs, each 5 silos × `round_001..020.jsonl`, per-round record `{round, silo_id, sigma, clipping_strategy, global_c0, per_layer_thresholds, micro_f1, epsilon, delta, wall_clock_seconds}`.

| Grid | σ / C0 | Seeds | Results |
|---|---|---|---|
| baseline (`run_baseline.sh:31-49`, `run_matrix.sh:14-16`) | σ∈{1.0, 2.0} | 0–9 | Y — 40 runs, `logs/baseline_sigma{1.0,2.0}_seed{0..9}` |
| ε-parity rerun (`docs/artigo2_errata_ci.md:670-681`) | σ=0.4508 | 0–9 | Y — `logs/baseline_sigma0.4508_seed{0..9}` (original σ=1.0 n=10 raw data lost, `scripts/run_baseline_sigma1_n10.sh:5-10`) |
| C0 sweep (`run_c0_sweep_matrix.sh:13-15`) | C0∈{1.0, 0.989, 0.22, 0.05}, σ=1.0 | 0–4 | Y — `logs/c0_sweep_*` (20 runs) |
| per-layer adaptive (`run_per_layer.sh:33-37`) | σ∈{1.0, 2.0}, percentile-75, clamp [0.001, 2.0] | 0–9 | Y — `logs/per_layer_sigma{1.0,2.0}_seed{0..9}` (20 runs) |
| σ sweep (`run_sigma_sweep_matrix.sh:15-16`) | σ∈{0.1, 0.3, 0.5, 2.0}, C0=1.0 | 0–4 | **N — `logs/sigma_sweep_*` NOT FOUND (never executed)** |

### 1.3 `experiments/article3` (dual LoRA / HERALD-PFL)

| Config | σ | q | R | Seeds | Results |
|---|---|---|---|---|---|
| `configs/dual_lora.yaml:6-11` (dual LoRA r=8+r=4, target_ε=2.5) | 1.0 | 0.01 | 100 | 0–9 | Y — `logs/a3_dual_lora_sigma1.0_seed{0..9}` complete (100 rounds/silo) + `cross_silo_eval.jsonl`; per-seed table `docs/article3_dual_lora_seeds.md:27-50` |
| `configs/dp_lora_baseline.yaml`, `configs/ffa_lora.yaml` (scripts `run_dp_lora.sh`, `run_ffa_lora.sh`) | 1.0 | 0.01 | 100 | 0–9 planned | **N — `logs/a3_dp_lora_*`, `logs/a3_ffa_lora_*` NOT FOUND**, despite `run_matrix_a3.sh:12-14` claiming "already complete" |
| `configs/rolo_ra_placeholder.yaml` | — | — | — | — | placeholder, self-declared "NOT YET IMPLEMENTED", unreferenced |

### 1.4 `experiments/client-level-dp` (Gate 1)

FedProx μ=0.01, batch=**32**, E=1, δ=1e-5, 5 silos, α=0.5, top-50, R=20 (`scripts/run_client_level_dp.sh:33-50`). Grid: σ∈{0.5, 1.0, 2.0} × seeds{0,1,2} + no-DP control × seeds{0,1,2} = 12 runs, all R=20 (e.g. `logs/p1_client_level_dp_sigma0.5_seed0_gate_server.log:350`). Results: silo dirs contain **only `final_global_params.npz`** — no per-round metric files; metrics tabulated in `docs/fase1_resultados.md:139-143,165-188`. Calibrated C_silo=0.576683 (`docs/fase1_resultados.md:35`). σ=26.16 dedup scenario: calculated only, never run (`docs/fase1_resultados.md:84,144`).

### 1.5 `experiments/herald-pate` (PATE phase)

No seeds in logs (m02 records `"seed":0). T=5 teachers on silo 0, no DP in teacher loop (`scripts/run_teachers.sh:20-28`). Confident-GNMax sweep σ_voto∈{0.3,0.5,1.0} × τ∈{2.5,3.0,3.5} + ε=∞ bound, 12,500 queries, 300-note proxy corpus (`scripts/run_m0.sh:22-30`; verified in `logs/m0/m0_summary.json`). Results present: `logs/teachers/teachers_summary.json` (val F1 0.100–0.133), `logs/m0/m0_summary.json`, `logs/m01/m01_ac_summary.json`, `logs/m01_b/m01_b_summary.json`, `logs/m02/m02_summary.json` (`gate_passed: false`); doc tables `docs/fase3_m0*_resultados.md`.

### 1.6 Checkpoints / wandb / tensorboard / CSV

- Checkpoints: `experiments/herald-pate/logs/teachers/teacher_{0..4}.pt`, `teachers_t3/teacher_{0..2}.pt`; 50× `article3/logs/a3_dual_lora_*/{0..4}/local_adapter.pt`; 110× `final_global_params.npz` (article3 + client-level-dp).
- **wandb / tensorboard / CSV outputs: NOT FOUND** (repo-wide find).
- Raw grad norms: `experiments/logs/grad_norms_*.jsonl` (92 files; 3 orphan synthetic runs unreferenced by any script).

### 1.7 Seed census

| Line | Seeds | n |
|---|---|---|
| Main pipeline (all groups §1.1) | 42, 43, 44 | 3 |
| adaptive-clipping baseline / per-layer | 0–9 | 10 |
| adaptive-clipping C0 sweep | 0–4 | 5 |
| article3 dual LoRA | 0–9 | 10 |
| client-level-dp (+ control) | 0, 1, 2 | 3 |
| herald-pate | not recorded | 1 |

### 1.8 Discrepancies (reproducibility flags)

1. **R=10 leg missing:** premise R∈{10,20}; all FL logs are R=20, R=10 exists only for centralized runs (§1.1).
2. **"Batch size 1 + gradient accumulation" applies to the LLM backend** (`run_nodocker.sh:65-66` batch=1/accum=64; `ai_client/src/ai_client/centralized_baseline.py:526-527`); BERT runs use batch=8/accum=8 (`run_nodocker.sh:67-68`). The DP training path further forces batch=1/accum=1 per optimizer step (`ai_client/src/ai_client/model_setup.py:473-481`, dataloader `batch_size=1` at `:547`).
3. **Default drift, dead path:** server `FL_NOISE_MULTIPLIER` default **0.9** (`fl_server/src/fl_server/server.py:94`) vs client default **0.0** (`fl_client.py:84`); the server-side adaptive-DP wrapper `wrap_with_dp` (`server.py:280-335`) is only called from tests (`fl_server/tests/test_dropout_and_nan_safety.py:124,162`) — `build_strategy` never wraps.
4. **Config drift:** `adaptive-clipping/configs/baseline.yaml:2` lists σ∈{1.0,2.0} but a third value (0.4508) was actually run; all 10 YAMLs there are loaded by no code (no yaml import in `src/adaptive_clipping`); σ-sweep configs+scripts exist but were never executed (§1.2); `per_layer.yaml:12-13` clamp [0.001, 2.0] vs code defaults [0.1, 10.0] (`clipping.py:107-108`).
5. **False completion claim:** `run_matrix_a3.sh:12-14` asserts dp_lora/ffa_lora (20 runs) complete; logs NOT FOUND (§1.3).
6. **Sampling/accounting mismatch:** actual sampling is WOR fixed-subset (`model_setup.py:541` `replace=False`), accountant charges Poisson-style subsampled RDP (`model_setup.py:670-674`); `opacus.accountants.get_noise_multiplier`: **NOT FOUND** anywhere.
7. **Archived logs:** `experiment_results_vastai/*.json` carry constant `epsilon_spent` per round but **no `epsilon_cumulative`**; per-round subsampling RNG is unseeded (`model_setup.py:540` `np.random.default_rng()` fresh OS entropy each round).
8. **client-level-dp uses batch=32** vs batch=8 elsewhere (§1.4) — cross-line comparability caveat.

---

## 2. Privacy accounting artifacts

- **Callable unit (YES):**
  ```python
  # ai_client/src/ai_client/fl_client.py:521-539
  def _compute_cumulative_epsilon(noise_multiplier: float, sample_rate: float,
                                  total_steps: int, delta: float) -> float:
  ```
  Instantiates a fresh Opacus `RDPAccountant`, steps it `total_steps` times, returns `get_epsilon(delta)` (`:531-536`). Pure bookkeeping — **ε at arbitrary round r is queryable without retraining** (callers pass cumulative step counts, e.g. `fl_client.py:1078-1083`). Guards: `inf` if steps≤0 or σ≤0 (`:528-529`).
- **Other accountant callables:** `group_composed_epsilon(q, noise_multiplier, steps, delta, n_groups=1, alphas=None) -> (eps, alpha)` (`experiments/adaptive-clipping/analysis/renyi_group_composition.py:32-52`; G-group Rényi composition over Opacus `compute_rdp`); `client_level_epsilon(sigma, total_rounds, delta, alphas=None)` + `ClientLevelAccountant` (`experiments/client-level-dp/src/client_level_dp/accountant.py:44-89`); PATE `total_epsilon(detection_probs, sigma_voto, delta, alphas=None)` (`experiments/herald-pate/src/herald_pate/gnmax_accountant.py:118-133`); approximate closed-form `estimate_privacy_budget(..., delta=1e-5)` (`fl_server/src/fl_server/server.py:109-141`, self-described conservative bound).
- **Per-round ε logging (YES):** client-side `epsilon_spent`/`epsilon_cumulative` (`fl_client.py:1062-1091`); merged into run JSONs (`run_experiments.sh:211-212`); per-round JSONL `epsilon`+`delta` in adaptive-clipping (`src/adaptive_clipping/server.py:97-117`, `logging_utils.py:28-30`) and article3 (`src/article3/server.py:79-85`). For archived main-pipeline logs only `epsilon_spent` (constant/round) is present; composed ε is analytically reconstructable offline via the callable above since (σ, q, R, δ) are known.
- **q computation:** `q = dp_subsample_rate` (default 0.0→active 0.1) → `k = max(1, int(n*q))` → `dp_sample_rate = k/n` — `model_setup.py:537-539` (BERT twin `model_setup_bert.py:446-461`); WOR draw `rng_sub.choice(n, size=k, replace=False)` (`:540-541`). Charged as Poisson-subsampled Gaussian: `acc.step(noise_multiplier=σ, sample_rate=q)` (`model_setup.py:670-674, 721-725`). q in use: 0.1 main + adaptive-clipping; **0.01 article3** (`experiments/article3/scripts/run_dual_lora.sh:68`).
- **δ = 1e-5 everywhere** — every occurrence found agrees: `fl_client.py:86,601,1077`; `model_setup.py:145`; `model_setup_bert.py:84`; `server.py:114,483`; `docker-compose.yml:46`; `run_nodocker.sh:330`; adaptive-clipping/article3/client-level-dp/pate scripts and code (`server.py:53`, `client.py:385`, `run_baseline.sh:52`, `run_dual_lora.sh:77`, `calibrate_c_silo.sh:28`, `voting.py:32`). δ≠1e-5: **NOT FOUND** (repo-wide grep).

---

## 3. DP method variants and FL baselines

| Mode | Evidence | Status |
|---|---|---|
| Manual record-level DP-SGD (clip + Gaussian noise `σ·C0`, no PrivacyEngine) | `model_setup.py:651-674`; BERT twin `model_setup_bert.py:557-626` | production path; activation `noise_multiplier>0` (`model_setup.py:476`) |
| No-DP ablation | `FL_NOISE_MULTIPLIER` default 0.0 (`fl_client.py:84`); nodp logs | runs |
| C0 calibration mode (noise-free norm logging) | `FL_CALIBRATE_GRAD_NORM` (`fl_client.py:90`; `model_setup.py:483-489,744-775`) | runs |
| Custom per-layer adaptive clipping (per-attention-layer percentile thresholds, per-group noise) | `experiments/adaptive-clipping/src/adaptive_clipping/clipping.py:88` (`PerLayerClipper`); applied `training.py:70-108` | runs (σ∈{1,2} × 10 seeds); **in-code ε disclaimer** `training.py:22-26` — scalar `epsilon_cumulative` not a valid guarantee for this path; corrected ε via `group_composed_epsilon` only in `docs/per_layer_epsilon_impact.md:159-162` |
| Server-side adaptive clipping (Flower `DifferentialPrivacyServerSideAdaptiveClipping`) | `fl_server/src/fl_server/server.py:62-63,189-240,280-335` | **dead code** — wired only in tests |
| Ghost clipping (Opacus `make_private`) | `experiments/article3-poc/ghost_clipping_poc.py:83-107` | PoC only |
| Opacus `clip_per_layer`/`DPLayerNoise` | — | **NOT FOUND** |
| Client-level DP (server clips silo Δ, noise on aggregate, 1 release/round) | `experiments/client-level-dp/src/client_level_dp/server.py:93-170` | runs (§1.4) |
| PATE Confident-GNMax | `experiments/herald-pate/src/herald_pate/gnmax_accountant.py:81-133`, `voting.py:87-112` | runs (§1.5) |
| Dual-LoRA (local adapter no-DP + global adapter DP) | `experiments/article3/src/article3/dual_training.py:116`; `configs/dual_lora.yaml` | runs (§1.3) |

- **FL algorithms: FedAvg + FedProx only.** `fl_server/src/fl_server/server.py:64,181-186,272-277`; μ default 0.01, env `FL_PROXIMAL_MU` (`server.py:93`; `docker-compose.yml:51`; forwarded per round `server.py:359-363`; applied to LoRA params only `model_setup.py:148-152,610-620`). FedAdam/FedOpt: **NOT FOUND in code** (prose citation only, `papers/artigo3/manuscript.md:75,299`).
- **Named-method search:** DP-FedAvg vanilla — **NOT FOUND** (as a label); FedMentor — **NOT FOUND**; PriFFT — **NOT FOUND**; secret sharing — **NOT FOUND**; secure aggregation — explicitly excluded, prose only (`docs/architecture.md:162`, `papers/artigo3/manuscript.md:211,217`); distillation — prose citations only (`papers/artigo2/manuscript.md:287`, `papers/artigo3/manuscript.md:211,215,316`); PATE — implemented (above).
- **C0 swept beyond 1.0: YES** — {1.0, 0.989, 0.22, 0.05} at σ=1.0 (`experiments/adaptive-clipping/scripts/run_c0_sweep_matrix.sh:13`; `configs/c0_sweep_*.yaml:11`; rationale+ε-parity note in headers). Per-layer clamp bounds are not global C0 (§1.8.4). C_silo=0.576683 data-dependent (`docs/fase1_resultados.md:35`).

---

## 4. Metrics and evaluation assets

| Metric | Computed | Logged/persisted |
|---|---|---|
| micro-F1 | `evaluation/src/evaluation/icd_metrics.py:174` (threshold grid-search `:120-137`) | flat `:60`; per-round via `fl_client.py:1203-1206`; server aggregation `server.py:414-435`; run JSON `run_experiments.sh:222-234` |
| macro-F1 | `icd_metrics.py:175` | same path |
| AUC-ROC micro/macro | `icd_metrics.py:184-193` (zero-variance filter `:180`) | same path |
| P@k / R@k / F1@k, k∈{8,15} | `icd_metrics.py:92-114,206`; k list `:159-160` | flat `:67-72` |
| LLM extraction acc / P@k / R@k / F1@k (beam k=5) | `fl_client.py:354-358` | per-round LLM logs |
| train/eval loss, perplexity | `model_setup_bert.py:160-161,645-646`; `icd_metrics.py:275,283`; `fl_client.py:279-280` | per-round |
| ROUGE-1/2/L, BERTScore (summarization line) | `evaluation/src/evaluation/summarization_metrics.py:75-100,40-55` | `fl_client.py:404-473` |
| LLM-judge (post-eval, optional) | `evaluation/src/evaluation/llm_judge.py:66,144-146` | appends to run JSON (`post_eval.py:283`) |

- **Centralized upper bound: reproducible.** `run_experiments.sh:11` ("Config 1. Centralised — 3 seeds (upper bound without FL)"); runner `run_experiments.sh:362-404` (docker) / `run_nodocker.sh` twin; implementation `ai_client/src/ai_client/centralized_baseline.py:366-483` (BERT); config: `BERT_BENCHMARK=top50`, seq 512, lr=2e-4, batch 8, 10 epochs, ≤10000 examples, seeds 42/43/44 (`run_experiments.sh:371-375,112-115`); output `experiment_logs/centralizado_bert_seed{N}.json` (`centralized_baseline.py:515`). Centralized logs are **final-metrics-only** for ICD metrics (per-epoch train_loss only in `per_round_eval`, `run_experiments.sh:283-295`).
- **Top-50 restriction: enforced in code.** Selection `ai_client/src/ai_client/fhir_consumer_bert.py:68-69` (`counter.most_common(50)`); head size `fl_client.py:631-633` (`num_labels=50`); ETL-side `etl_worker/mimic_builder.py:434-436`; env switch `BERT_BENCHMARK` default top50 (`centralized_baseline.py:512`, `.env.example:12`); shared artifact `etl_worker/data/label_index.json` (50 codes, in-repo).

---

## 5. Test suite health

- pytest is **not declared in any pyproject** (root `pyproject.toml:34-35` has only `addopts="-q"`); pytest 9.1.1 + pytest-cov were installed ad hoc into the venvs for this audit.
- **Main workspace packages — partial, blocked by external deletion.** `pytest -x -q` collected `ai_client/tests` and failed at 5.8 s on **1 real bug**: `ai_client/tests/test_fl_client_dp_accounting.py::TestExtractIcdCodesTruncatesLongerSubcodes::test_should_extract_full_three_digit_subcode_not_truncate_it` — `ICD10_PATTERN`'s `\.\d{1,2}` truncates 3-digit subcodes (`J45.909` → `J45`). Immediately after, the test sources in `ai_client/tests/` (2 files), `etl_worker/tests/` (3), `evaluation/tests/` (2), `fl_server/tests/` (3) were deleted by an external process (audit incident, header); only `__pycache__/*.pyc` remain. **Full pass/fail counts and coverage for the main packages: NOT OBTAINABLE in this state.** Test files were untracked — unrecoverable via git.
- **Experiment packages (intact) — all green:**

| Suite | Result | Runtime |
|---|---|---|
| `experiments/adaptive-clipping` | **12 passed, 0 failed** | ~6.0 s |
| `experiments/client-level-dp` | **25 passed, 0 failed** | ~3.3 s |
| `experiments/herald-pate` | **7 passed, 0 failed** | ~8.6 s |

- **Coverage (measured, `--cov=src --cov-report=term`):**

| Package | TOTAL | 0%-covered modules (risky to touch) |
|---|---|---|
| adaptive-clipping | **10%** | `client.py` 0%, `server.py` 0%, `training.py` 0%, `logging_utils.py` 0% (only `clipping.py` 82%) |
| client-level-dp | **35%** | `client.py` 0%, `cross_silo_eval.py` 0%, `guards.py` 0% (`accountant.py` 87%, `server.py` 80%) |
| herald-pate | **6%** | `voting.py`, `teachers.py`, `silo_teachers.py`, `run_m0.py`, `run_m01*.py`, `run_m02.py`, `m01_diagnostics.py` all 0% (only `gnmax_accountant.py` tested) |

- Main-pipeline modules with **no test files targeting them** (before deletion; inferred from test-file names): `model_setup.py`, `model_setup_bert.py`, `centralized_baseline.py`, `fhir_consumer_bert.py`, `turbocompress.py`, `evaluation/metrics_logger.py`, `summarization_metrics.py`, `fl_server/server.py` (only partially via the now-deleted tests).

---

## 6. Rerun feasibility

- **Runtime data — main pipeline: NOT FOUND.** No wall-clock field in `experiment_logs/*.json` (`FIT_KEYS_TO_MERGE` excludes timing, `run_experiments.sh:211`); `GPUTimer` exists but unused in production (`evaluation/src/evaluation/metrics_logger.py:200-227`). No runtime data in `docs/fase*.md` (`docs/fase1_resultados.md:5-7` states the full sweep was never executed).
- **Runtime data — research lines: YES, per-round.** adaptive-clipping JSONL `wall_clock_seconds` (`src/adaptive_clipping/client.py:272`): ≈7.44–7.71 s/round (`logs/baseline_sigma0.4508_seed0/0/round_001.jsonl` and server log rounds 1–5). article3: ≈33.96 s/round (`logs/a3_dual_lora_sigma1.0_seed3/3/round_018.jsonl`; timer `src/article3/client.py:142`).
- **GPU-hour estimate (lower bound; single GPU, silos sequential, GPU model unknown — treat as rough):** main grid = 5 FL configs × 3 seeds = 15 runs. Proxy: same BERT model, R=20, q=0.1, 5 silos at ≈7.5 s/round/silo → ≈20×7.5×5 ≈ 750 s ≈ **0.21 GPU-h per config-seed → ≈3.2 GPU-h for the 15-run FL grid**; centralized upper bound: no runtime data ("NOT FOUND"). article3 grid (30 runs @ R=100, ≈34 s/round) ≈ 4.7 GPU-h/run → ≈141 GPU-h if rerun. **Estimate quality: LOW** — one measured proxy, unknown hardware, eval time excluded.
- **Determinism controls present:** `torch.manual_seed`/`cuda.manual_seed_all`/`np.random.seed` centralized only (`centralized_baseline.py:294-296,397-492`); split RNG `np.random.default_rng(seed)` `fl_client.py:157-171`; shuffle `_random.seed(42+partition_id)` `fl_client.py:768`; experiment clients seed torch/np/random (`experiments/article3/src/article3/client.py:325-327`; `experiments/adaptive-clipping/src/adaptive_clipping/client.py:481-483`).
- **Determinism gaps:** `cudnn.deterministic`/`benchmark` flags **NOT FOUND**; Flower server-side sampling seed **NOT FOUND** (`server.py`, `docker-compose.yml`); **DP subsampling RNG unseeded** — `np.random.default_rng()` fresh OS entropy per round (`model_setup.py:540`) → exact DP-subset reproduction impossible; `transformers.set_seed`/DataLoader `generator=` **NOT FOUND**.
- **Hardcoded paths:** no absolute `/home/...` in code (only placeholder `.env.example:75`); GPU lock `/var/gpu_sync/gpu.lock` (`fl_client.py:95`, needs writable path outside Docker); container data paths overridable (`run_experiments.sh:396-398`); `logs_root` CWD-relative in experiment `logging_utils.py`.
- **Dataset blockers:** MIMIC-IV v3.1 + MIMIC-IV-Note present in-repo (`physionet.org/files/…`, 12 GB); fresh machine needs **credentialed PhysioNet download** — no downloader script exists (`mimic_builder.py:823,1119` error out); ETL re-runs automatically when patient count <1000 (`run_experiments.sh:454-495`); HAPI FHIR container at `localhost:8080` required. `HF_TOKEN` needed for gated Llama only (`.env.example:61-66`); no PhysioNet token key exists in `.env.example`. Env: Python 3.13 (`.python-version`, `pyproject.toml:6`), uv workspace (`pyproject.toml:9-15`), torch CUDA 12.4 index (`pyproject.toml:17-26`), NVIDIA Container Toolkit for `ai_client/Dockerfile`.

---

## 7. Answers + recommendations

**A. Complete logs for σ×R sweep incl. per-round ε? — PARTIAL.**
σ∈{0.5,1.0,2.0} × R=20 × 3 seeds: complete, with per-round metrics + `epsilon_spent` (`experiment_logs/fl_fedprox_alpha0.5_dp{0.5,1.0,2.0}_bert_seed{42,43,44}.json`). R=10 FL: **NOT FOUND**. Composed per-round ε: reconstructable analytically via `_compute_cumulative_epsilon` (§2); `epsilon_cumulative` absent from archived JSONs. FedAvg+DP leg: NOT FOUND (FedAvg logs are no-DP only).

**B. Utility-ε curves from existing data alone? — YES (offline, analytic ε).**
Per-round utility (micro/macro-F1, AUC, P@k/R@k) exists for every logged round; ε(r) is a pure function of (σ, q, δ, steps) computable with the existing accountant callables — no retraining needed. Missing for a full σ×R curve plot: the R=10 leg, and a decision on the WOR-vs-Poisson accounting semantics (§1.8.6) before quoting ε values.

**C. Comparable DP baselines with zero new implementations:**
1. Record-level DP-SGD FedProx, σ∈{0.5,1,2}, R=20, 3 seeds — logs + code.
2. No-DP controls (FedAvg, FedProx) — logs + code.
3. C0 ablation {0.05,0.22,0.989,1.0} @ σ=1.0, 5 seeds — logs + code.
4. Per-layer adaptive clipping, σ∈{1,2}, 10 seeds — logs + code (+ corrected ε via `group_composed_epsilon`).
5. Client-level DP, σ∈{0.5,1,2}, 3 seeds — logs (final-only metrics) + code + custom accountant.
6. PATE Confident-GNMax, σ_voto×τ grid — logs + code + accountant.
7. Dual-LoRA DP @ σ=1.0, R=100, 10 seeds — logs + code.
8. Vanilla **DP-FedAvg** = `FL_STRATEGY=fedavg` + `FL_NOISE_MULTIPLIER>0` — no new code, just a rerun (untested combination; flag for smoke validation).

**D. Baselines requiring new code:**
1. FedMentor-style distillation — **L** (no distillation/teacher-student training infra; only prose citations).
2. PriFFT-style secret sharing / secure aggregation — **L** (explicitly excluded by design, `docs/architecture.md:162`).
3. Opacus-native `PrivacyEngine` + `clip_per_layer` — **M** (PoC only, `article3-poc/ghost_clipping_poc.py:83-107`).
4. Equalization operator E as a batch analysis tool (ε-target → σ/C0/R reparameterization per method, incl. client-level and group-composed accountants) — **S** (all accountant primitives exist; needs an inversion/driver script).

**E. Accountant reusable as library call for E? — YES.**
`_compute_cumulative_epsilon(noise_multiplier: float, sample_rate: float, total_steps: int, delta: float) -> float` (`ai_client/src/ai_client/fl_client.py:521-539`); plus `group_composed_epsilon(q, noise_multiplier, steps, delta, n_groups=1, alphas=None) -> tuple[float,float]` (`experiments/adaptive-clipping/analysis/renyi_group_composition.py:32-52`) and `client_level_epsilon(sigma, total_rounds, delta, alphas=None)` (`experiments/client-level-dp/src/client_level_dp/accountant.py:44-55`). Caveats: O(steps) loop per query (fine for R≤100, steps≈580); WOR-vs-Poisson semantics to be pinned down (§1.8.6); per-layer path needs `n_groups` composition, not the scalar accountant (in-code disclaimer `training.py:22-26`).

**F. Top-3 blockers (severity-ordered):**
1. **Missing R=10 FL leg + missing FedAvg+DP leg** — the headline σ×R matrix is only half present; reruns required (≈3 GPU-h estimated, LOW confidence).
2. **Accounting semantics unresolved** — WOR sampling charged as Poisson (`model_setup.py:541` vs `:670-674`), unseeded subsampling RNG (`:540`), per-layer ε only corrected offline; the paper's E operator must fix one convention or carry both.
3. **Repo instability + missing instrumentation** — concurrent process deleted the main test suite mid-audit (untracked, unrecoverable); zero runtime logging in the main pipeline; pytest not declared in any pyproject. A clean rerun protocol needs the tests restored/committed, timing fields added, and seeds pinned (cudnn/Flower server/DP subsampling).

---

## 8. Proposed experimental matrix (draft — existing repo assets only)

Legend: **[E]** data exists · **[R]** rerun needed (code exists) · **[N]** new code needed

| Method | Budgets (σ / C0 / R / q) | Seeds | Flag |
|---|---|---|---|
| FedProx DP-SGD (record-level) | σ∈{0.5,1.0,2.0}, C0=1.0, R=20, q=0.1 | 42,43,44 | **[E]** |
| FedProx DP-SGD, R=10 leg | same σ, R=10 | 42,43,44 | **[R]** |
| DP-FedAvg vanilla | σ∈{0.5,1.0,2.0}, R∈{10,20}, q=0.1 | 42,43,44 | **[R]** (env-only combo; smoke-test first) |
| FedProx / FedAvg no-DP | R=20 | 42,43,44 | **[E]** |
| C0 ablation | C0∈{0.05,0.22,0.989,1.0}, σ=1.0, R=20 | 0–4 | **[E]** (note: seed set ≠ main line) |
| Per-layer adaptive clipping | σ∈{1.0,2.0}, R=20 | 0–9 | **[E]** (use `group_composed_epsilon` for ε) |
| Client-level DP | σ∈{0.5,1.0,2.0}, C_silo=0.5767, R=20 | 0–2 | **[E]** (final-only metrics; per-round instrumentation = small **[N]**) |
| PATE Confident-GNMax | σ_voto∈{0.3,0.5,1.0} × τ∈{2.5,3.0,3.5} | n/a (single run) | **[E]** (add seeds = **[R]**) |
| Dual-LoRA DP | σ=1.0, q=0.01, R=100 | 0–9 | **[E]** |
| Equalization operator E (ε-target reparameterization per method) | analytic | n/a | **[N]** (S — compose existing accountants) |
| FedMentor-style distillation | — | — | **[N]** (L — out of scope for minimal matrix) |
| PriFFT / secure aggregation | — | — | **[N]** (L — out of scope) |

**Recommended minimal core:** rows 1–3 (σ × R × algo grid, 3 seeds) + no-DP row + E operator = a defensible fair-budget comparison entirely on existing training code, with ~9–12 rerun jobs and one small analysis script. Rows 6–8 strengthen the "methods" axis if the paper claims cross-mechanism equalization.
