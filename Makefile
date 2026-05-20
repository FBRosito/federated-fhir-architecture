# ==============================================================================
# Task Orchestrator — Federated FHIR Architecture
# Requires: uv (package management) and Docker with modern Compose plugin
# ==============================================================================

.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------
COMPOSE         := docker compose
UV              := uv
STRATEGY        ?= fedprox      # options: fedprox | fedavg
N_SILOS         ?= 5            # number of Dirichlet silos
DIRICHLET_ALPHA ?= 0.5          # α=0.1 (highly Non-IID) | 0.5 (moderate) | 1.0 (near-IID)
ICD_VERSION     ?= icd10        # icd10 = filter admittime >= 2015-10-01
BENCHMARK       ?= full         # top50 | full (Mullenbach 2018) | none
ROUNDS          ?= 5            # number of FL rounds
NOISE           ?= 0.0          # DP noise multiplier: 0.0=off | 0.5=weak | 1.0=moderate | 2.0=strong
MU              ?= 0.01         # FedProx proximal μ (use 0.0 for FedAvg)
COMPRESS        ?= false        # TurboQuant LoRA delta compression: true | false
COMPRESS_BITS   ?= 4            # TurboQuant bit-width: 2 | 4 | 8

SILO_SERVICES   = fl_server $(shell seq 0 $$(( $(N_SILOS) - 1 )) | xargs -I{} echo ai_client_silo_{})

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "  Federated FHIR Architecture — available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

.PHONY: setup
setup: ## Create/update virtual environment and install dependencies via uv
	$(UV) sync

.PHONY: up-infra
up-infra: ## Start HAPI FHIR, FL Server, and ETL Worker; blocks until infra is 100% ready (STRATEGY=fedprox|fedavg)
	FL_STRATEGY=$(STRATEGY) $(COMPOSE) up -d hapi_fhir fl_server etl_worker
	@echo ">>> [1/3] Waiting for HAPI FHIR at http://localhost:8080/fhir/metadata ..."
	@until curl -sf http://localhost:8080/fhir/metadata > /dev/null 2>&1; do \
		printf '.'; sleep 5; \
	done
	@echo " HAPI FHIR ready."
	@echo ">>> [2/3] Waiting for ETL Worker to finish FHIR loading (max 10 min)..."
	@i=0; \
	while [ $$i -lt 120 ]; do \
		s=$$(docker inspect --format='{{.State.Status}}' etl_worker 2>/dev/null || echo "gone"); \
		[ "$$s" != "running" ] && break; \
		printf '.'; sleep 5; i=$$((i+1)); \
	done; \
	final=$$(docker inspect --format='{{.State.Status}} (exit={{.State.ExitCode}})' etl_worker 2>/dev/null || echo "gone"); \
	echo " ETL Worker: $$final."
	@echo ">>> [3/3] Waiting for fl_server to become healthy (port 9091) ..."
	@until [ "$$(docker inspect --format='{{.State.Health.Status}}' fl_server 2>/dev/null)" = "healthy" ]; do \
		printf '.'; sleep 5; \
	done
	@echo " fl_server healthy."
	@echo ""
	@echo "Infra 100% ready — run 'make up-ai'."

.PHONY: up-ai
up-ai: ## Start 5 AI silos in foreground with live logs
	$(COMPOSE) up ai_client_silo_0 ai_client_silo_1 ai_client_silo_2 ai_client_silo_3 ai_client_silo_4

.PHONY: up-ai-fast
up-ai-fast: ## Quick mode: 1 silo, 20 examples, seq=512, ~2-3 min end-to-end (Llama-3.2-1B)
	@echo ">>> [1/3] Ensuring HAPI FHIR and ETL Worker are running..."
	FL_STRATEGY=$(STRATEGY) $(COMPOSE) up -d hapi_fhir etl_worker
	@until curl -sf http://localhost:8080/fhir/metadata > /dev/null 2>&1; do \
		printf '.'; sleep 5; \
	done
	@echo " HAPI FHIR ready."
	@i=0; \
	while [ $$i -lt 120 ]; do \
		s=$$(docker inspect --format='{{.State.Status}}' etl_worker 2>/dev/null || echo "gone"); \
		[ "$$s" != "running" ] && break; \
		printf '.'; sleep 5; i=$$((i+1)); \
	done; \
	echo " ETL Worker: $$(docker inspect --format='{{.State.Status}} (exit={{.State.ExitCode}})' etl_worker 2>/dev/null || echo gone)."
	@echo ">>> [2/3] Restarting fl_server (2 rounds, 1 client)..."
	FL_NUM_ROUNDS=2 FL_MIN_CLIENTS=1 FL_STRATEGY=$(STRATEGY) FL_LEARNING_RATE=5e-6 FL_NOISE_MULTIPLIER=0 \
		$(COMPOSE) up -d --force-recreate fl_server
	@until [ "$$(docker inspect --format='{{.State.Health.Status}}' fl_server 2>/dev/null)" = "healthy" ]; \
		do printf '.'; sleep 3; done
	@echo " fl_server ready."
	@echo ">>> [3/3] Starting AI silo..."
	FL_NUM_ROUNDS=2 FL_MIN_CLIENTS=1 FL_STRATEGY=$(STRATEGY) FL_LEARNING_RATE=5e-6 \
	FL_KEEP_MODEL_IN_VRAM=true FL_MAX_EXAMPLES=20 MAX_SEQ_LEN=512 \
	FL_GRADIENT_ACCUM_STEPS=2 FL_BATCH_SIZE=8 FL_NOISE_MULTIPLIER=0 \
		$(COMPOSE) up ai_client_silo_0

.PHONY: smoke
smoke: ## Run full smoke test (all experiments, 1 seed, 20 examples) — MUST pass before cloud
	@echo ">>> Smoke test: all experiments (A+B), 1 seed, 20 examples, 2 rounds"
	bash run_experiments.sh --smoke --exp all
	@echo ">>> Smoke test done. Check experiment_logs/run_*_smoke.log"

.PHONY: smoke-A
smoke-A: ## Run smoke test for Experiment A only (ICD-10 / PubMedBERT)
	bash run_experiments.sh --smoke --exp A

.PHONY: smoke-B
smoke-B: ## Run smoke test for Experiment B only (discharge summary / Llama)
	bash run_experiments.sh --smoke --exp B

.PHONY: logs
logs: ## Follow logs of all containers in real time
	$(COMPOSE) logs -f

.PHONY: logs-master
logs-master: ## Tail the most recent master log file (experiment_logs/run_*.log)
	@latest=$$(ls -t experiment_logs/run_*.log 2>/dev/null | head -1); \
	[ -n "$$latest" ] || { echo "No master log found in experiment_logs/."; exit 1; }; \
	echo ">>> Tailing: $$latest"; \
	tail -f "$$latest"

.PHONY: logs-pack
logs-pack: ## Compress all experiment_logs into a tarball for sharing (experiment_logs.tar.gz)
	tar -czf experiment_logs.tar.gz experiment_logs/
	@echo ">>> Created: experiment_logs.tar.gz ($(du -sh experiment_logs.tar.gz | cut -f1))"

.PHONY: down
down: ## Stop all containers
	$(COMPOSE) down

.PHONY: build-mimic
build-mimic: ## Generate FHIR bundles from MIMIC-IV + real notes (N_SILOS=5 DIRICHLET_ALPHA=0.5 BENCHMARK=full ICD_VERSION=icd10)
	$(UV) run --package etl-worker python etl_worker/mimic_builder.py \
		--mimic-dir physionet.org/files/mimiciv/3.1 \
		--note-dir  physionet.org/files/mimic-iv-note/2.2/note \
		--bundles-dir etl_worker/data/bundles \
		--max-admissions $(or $(MAX_ADMISSIONS),2000) \
		--n-silos $(N_SILOS) \
		--dirichlet-alpha $(DIRICHLET_ALPHA) \
		--icd-version $(ICD_VERSION) \
		--benchmark $(BENCHMARK) \
		--split-output etl_worker/data/temporal_split.json \
		--label-index-output etl_worker/data/label_index.json \
		--skip-vitals

.PHONY: up-mimic
up-mimic: build-mimic up-infra ## Generate MIMIC-IV bundles, start infra and load into HAPI FHIR

# ------------------------------------------------------------------------------
# Single-experiment targets (no script needed)
# Override on command line: make run-bert ROUNDS=10 STRATEGY=fedavg NOISE=1.0
# ------------------------------------------------------------------------------

.PHONY: run-bert
run-bert: ## Experiment A — PubMedBERT + ICD-10 multi-label. Args: ROUNDS STRATEGY NOISE N_SILOS MU COMPRESS
	MODEL_BACKEND=bert BERT_BENCHMARK=top50 FL_BATCH_SIZE=8 FL_GRADIENT_ACCUM_STEPS=8 \
	MAX_SEQ_LEN=512 FL_NUM_ROUNDS=$(ROUNDS) FL_STRATEGY=$(STRATEGY) \
	FL_NOISE_MULTIPLIER=$(NOISE) FL_PROXIMAL_MU=$(MU) FL_MIN_CLIENTS=$(N_SILOS) \
	ETL_DIRICHLET_ALPHA=$(DIRICHLET_ALPHA) FL_LORA_COMPRESS=$(COMPRESS) FL_COMPRESS_BITS=$(COMPRESS_BITS) \
	$(COMPOSE) up $(SILO_SERVICES)

.PHONY: run-llm
run-llm: ## Experiment B — Llama-3.2 + discharge summary. Args: ROUNDS STRATEGY NOISE N_SILOS MU COMPRESS
	MODEL_BACKEND=llm FL_BATCH_SIZE=1 FL_GRADIENT_ACCUM_STEPS=64 MAX_SEQ_LEN=1024 \
	FL_NUM_ROUNDS=$(ROUNDS) FL_STRATEGY=$(STRATEGY) \
	FL_NOISE_MULTIPLIER=$(NOISE) FL_PROXIMAL_MU=$(MU) FL_MIN_CLIENTS=$(N_SILOS) \
	ETL_DIRICHLET_ALPHA=$(DIRICHLET_ALPHA) FL_LORA_COMPRESS=$(COMPRESS) FL_COMPRESS_BITS=$(COMPRESS_BITS) \
	$(COMPOSE) up $(SILO_SERVICES)

.PHONY: clean
clean: ## Stop containers, remove volumes, .venv and Python caches (full reset)
	$(COMPOSE) down -v
	rm -rf .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
