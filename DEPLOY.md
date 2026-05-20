# Deploy Guide — Federated FHIR Architecture

Step-by-step: GitHub commit → cloud VM → MIMIC-IV data → full experiments.

---

## Prerequisites checklist

- [ ] Smoke test passes locally (see below)
- [ ] HuggingFace token with Llama-3 access
- [ ] PhysioNet account with MIMIC-IV and MIMIC-IV-Note approved access
- [ ] SSH key pair for the cloud VM
- [ ] ~200 GB disk on the cloud VM (MIMIC ~80 GB + Docker images ~20 GB + model cache ~15 GB)

---

## Step 0 — Run the local smoke test first

**The smoke test MUST pass before going to the cloud.** It covers every code path in ~2–3 hours on a local GPU (or longer without one).

```bash
# Runs all experiments: calibration, centralised, FL+FedProx, FL+FedAvg, FL+DP (both backends)
# Output: live on screen AND saved to experiment_logs/run_YYYYMMDD_HHMMSS_smoke.log
make smoke

# Quick check: should show no Traceback / Error
grep -E "Traceback|Error|EXIT_CODE" experiment_logs/run_*_smoke.log

# All JSON results produced?
ls experiment_logs/*.json | wc -l   # expect 12+ files (smoke: 1 seed)
```

If any error appears, share the log:
```bash
make logs-pack   # creates experiment_logs.tar.gz for sharing
```

---

## Step 1 — Push to GitHub

The remote is already configured: `git@github.com:FBRosito/federated-fhir-architecture.git`

### 1.1 Verify what will be committed (security check)

```bash
git status
# Confirm these are NOT staged (they are in .gitignore):
#   physionet.org/   ← MIMIC-IV data (never commit)
#   .env             ← HF token (never commit)
#   experiment_logs/*.log  ← raw logs
#   experiment_logs/*.json ← result JSONs (commit after experiments)
```

### 1.2 Commit all changes

```bash
cd /home/fbrosito/workspace/federated-fhir-architecture

git add \
  CLAUDE.md README.md DEPLOY.md Makefile run_experiments.sh \
  ai_client/ etl_worker/ fl_server/ evaluation/ \
  docker-compose.yml pyproject.toml uv.lock \
  .gitignore

git status   # review — no physionet.org/, no .env, no *.log

git commit -m "feat: complete architecture — translation, gap fixes, eval pipeline, README"

git push origin main
```

### 1.3 Verify on GitHub

Open `https://github.com/FBRosito/federated-fhir-architecture` and confirm:
- All source files are present
- No `physionet.org/`, `.env`, or large binary files
- `DEPLOY.md`, `README.md`, `Makefile`, `run_experiments.sh` look correct

---

## Step 2 — Provision the cloud VM

### 2.1 Recommended platforms

| Platform | GPU | VRAM | $/h | Notes |
|----------|-----|------|-----|-------|
| **Lambda Labs** | A100 SXM4 | 40 GB | ~$1.29 | Best for long runs (24h+), stable |
| Vast.ai | A100 | 40–80 GB | $0.80–2.00 | Marketplace — check host reputation |
| RunPod | A100 SXM | 80 GB | ~$2.49 | Simple UI |
| GCP | A100 | 40 GB | ~$3.67 | Most reliable, most expensive |

**Recommended: Lambda Labs A100 SXM4 40 GB** — most stable for long FL runs.

### 2.2 Instance configuration

- **OS:** Ubuntu 22.04 LTS
- **Disk:** 200 GB minimum (500 GB preferred for full MIMIC-IV)
- **SSH key:** upload your public key during instance creation

### 2.3 Connect

```bash
ssh ubuntu@<INSTANCE_IP>
# or if using Lambda:
ssh -i ~/.ssh/your_key ubuntu@<INSTANCE_IP>
```

---

## Step 3 — Set up the instance

### 3.1 Install Docker + NVIDIA Container Toolkit + uv

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Docker Compose plugin (modern)
sudo apt-get install -y docker-compose-plugin
docker compose version   # should show v2.x

# NVIDIA Container Toolkit (if not pre-installed by Lambda)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or re-login
uv --version
```

### 3.2 Clone the repository

```bash
git clone git@github.com:FBRosito/federated-fhir-architecture.git
cd federated-fhir-architecture
```

If using HTTPS:
```bash
git clone https://github.com/FBRosito/federated-fhir-architecture.git
cd federated-fhir-architecture
```

### 3.3 Configure credentials

```bash
# Create .env from template
cat > .env <<'EOF'
HF_TOKEN=hf_YOUR_TOKEN_HERE
PHYSIONET_DIR=/home/ubuntu/physionet.org/files
EOF

# Verify token works
docker run --rm -e HF_TOKEN=$HF_TOKEN \
  huggingface/transformers-pytorch-gpu \
  python -c "from huggingface_hub import login; login(token='$HF_TOKEN'); print('OK')"
```

### 3.4 Install Python dependencies

```bash
uv sync --frozen
```

---

## Step 4 — Download MIMIC-IV

You need a PhysioNet account with approved access to both datasets:
- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- MIMIC-IV-Note v2.2: https://physionet.org/content/mimic-iv-note/2.2/

### 4.1 Create directory structure

```bash
mkdir -p ~/physionet.org/files/mimiciv/3.1/hosp
mkdir -p ~/physionet.org/files/mimiciv/3.1/icu
mkdir -p ~/physionet.org/files/mimic-iv-note/2.2/note
```

### 4.2 Download required files

```bash
PUSER="YOUR_PHYSIONET_USERNAME"
HOSP="https://physionet.org/files/mimiciv/3.1/hosp"
NOTE="https://physionet.org/files/mimic-iv-note/2.2/note"

cd ~/physionet.org/files/mimiciv/3.1/hosp

# Core tables (required for Experiment A — ICD-10 coding)
for f in diagnoses_icd.csv.gz d_icd_diagnoses.csv.gz admissions.csv.gz \
          patients.csv.gz services.csv.gz procedures_icd.csv.gz \
          d_icd_procedures.csv.gz prescriptions.csv.gz; do
  wget -N --user "$PUSER" --ask-password "$HOSP/$f"
done

# Discharge notes (required for Experiment B — discharge summary)
cd ~/physionet.org/files/mimic-iv-note/2.2/note
wget -N --user "$PUSER" --ask-password "$NOTE/discharge.csv.gz"

# Check sizes
du -sh ~/physionet.org/files/mimiciv/3.1/hosp/
du -sh ~/physionet.org/files/mimic-iv-note/2.2/note/
```

> **Optional — ICU vitals** (adds ~50 GB, used only for synthetic narratives):
> ```bash
> cd ~/physionet.org/files/mimiciv/3.1/icu
> wget -N --user "$PUSER" --ask-password \
>   https://physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz \
>   https://physionet.org/files/mimiciv/3.1/icu/chartevents.csv.gz
> ```

### 4.3 Update .env with the correct path

```bash
# The path must match what you created above
echo "PHYSIONET_DIR=/home/ubuntu/physionet.org/files" >> .env
```

---

## Step 5 — Build FHIR bundles from MIMIC-IV

This step reads MIMIC-IV, applies Dirichlet partitioning, and writes pre-assembled FHIR bundles to `etl_worker/data/bundles/`. Runtime: ~15–30 min for 10,000 admissions.

```bash
# 10,000 admissions × 5 silos × α=0.5, full ICD-10 benchmark
# Writes: etl_worker/data/bundles/bundle_000000.json ... bundle_009999.json
#         etl_worker/data/temporal_split.json
#         etl_worker/data/label_index.json
make build-mimic \
  MAX_ADMISSIONS=10000 \
  N_SILOS=5 \
  DIRICHLET_ALPHA=0.5 \
  BENCHMARK=full \
  ICD_VERSION=icd10

# Verify output
ls etl_worker/data/bundles/ | wc -l   # expect ~10000
cat etl_worker/data/temporal_split.json | python3 -m json.tool | head -20
```

---

## Step 6 — Start infrastructure

```bash
# Builds Docker images (first time: 10–20 min depending on bandwidth)
docker compose build

# Starts HAPI FHIR + FL Server + ETL Worker (loads bundles into HAPI)
# Blocks until all three are healthy — do not proceed until this completes
make up-infra

# Verify HAPI FHIR has data
curl -s http://localhost:8080/fhir/Patient?_summary=count | python3 -m json.tool
# "total" should be > 0

curl -s http://localhost:8080/fhir/DocumentReference?_summary=count | python3 -m json.tool
# "total" should match number of bundles
```

---

## Step 7 — Run the full experiment matrix

### 7.1 Configure hardware for A100

```bash
# A100 40 GB: larger batch, parallel silos
export FL_BATCH_SIZE=8
export FL_GRADIENT_ACCUM_STEPS=8
export FL_PARALLEL_GPU=true

# For A100 80 GB, can push further:
# export FL_BATCH_SIZE=16
# export FL_GRADIENT_ACCUM_STEPS=4
```

### 7.2 Launch experiments (survives terminal disconnect)

```bash
# screen or tmux session (IMPORTANT — experiment takes 15-20h)
screen -S flexp
# or: tmux new -s flexp

# Run inside screen/tmux:
# All output goes to screen AND to experiment_logs/run_YYYYMMDD_HHMMSS_full.log
bash run_experiments.sh --exp all

# Detach (keep running): Ctrl+A, D (screen) or Ctrl+B, D (tmux)
# Reattach: screen -r flexp or tmux attach -t flexp
```

Alternatively, with `nohup` (less preferred — harder to reattach):
```bash
nohup bash run_experiments.sh --exp all &
echo $! > experiment_logs/run.pid
# The master log is written inside run_experiments.sh automatically
```

### 7.3 Monitor progress

```bash
# Follow the master log (all output, all experiments)
make logs-master

# Quick progress check — completed runs
grep "Run .* completed\|COMPLETED" experiment_logs/run_*_full.log | wc -l
# 38 total runs (2 calibrations + 36 experiments)

# Show only key metrics as they arrive
tail -f experiment_logs/run_*_full.log | grep -E "CONFIG|completed|loss=|rouge|f1|ERROR"

# Check JSON files produced so far
ls experiment_logs/*.json 2>/dev/null | wc -l
```

### 7.4 Estimated runtime on A100 40 GB

| Phase | Runs | Est. time |
|-------|------|-----------|
| Calibration (bert + llm) | 2 | ~25 min |
| Centralised baseline (bert + llm, 3 seeds each) | 6 | ~3 h |
| FL no-DP (FedProx + FedAvg, both backends, 3 seeds) | 12 | ~5 h |
| FL with DP (σ=0.5, 1.0, 2.0, both backends, 3 seeds) | 18 | ~8 h |
| **Total** | **38** | **~16 h** |

---

## Step 8 — Retrieve results

### 8.1 Download all results to local machine

```bash
# From your LOCAL machine:
REMOTE="ubuntu@<INSTANCE_IP>"
REPO="federated-fhir-architecture"

# Pack on remote first
ssh $REMOTE "cd $REPO && make logs-pack"

# Download
scp $REMOTE:~/$REPO/experiment_logs.tar.gz ./
tar -xzf experiment_logs.tar.gz
ls experiment_logs/*.json | wc -l   # expect 38
```

### 8.2 Inspect results

```bash
# Summary of final metrics for all runs
python3 - <<'EOF'
import json, glob, os
rows = []
for path in sorted(glob.glob("experiment_logs/*.json")):
    try:
        d = json.load(open(path))
        tag = d.get("tag", os.path.basename(path))
        fm  = d.get("final_metrics", {})
        rows.append((tag, fm))
    except Exception:
        pass
for tag, fm in rows:
    print(f"{tag:60s}  {fm}")
EOF
```

### 8.3 Commit results to git

```bash
# On the cloud VM:
git add experiment_logs/*.json
git commit -m "results: full experimental matrix — bert+llm, 3 seeds, all DP configs"
git push origin main

# Pull locally:
git pull origin main
```

---

## Step 9 — Post-evaluation (Experiment B only)

After the FL run with `MODEL_BACKEND=llm-summarization` and `FL_SAVE_CHECKPOINT` set:

```bash
# On the cloud VM, after the LLM run completes:
python -m evaluation.post_eval \
  --run-json experiment_logs/fl_fedprox_alpha0.5_nodp_llm_seed42.json \
  --checkpoint /tmp/checkpoints/silo_0 \
  --fhir-url http://localhost:8080/fhir \
  --max-samples 50

# LLM-judge requires OPENROUTER_API_KEY:
export OPENROUTER_API_KEY=sk-or-your_key_here
```

---

## Step 10 — Generate publication figures

On your local machine after downloading results:

```bash
# Statistical summary (bootstrap CI, Wilcoxon tests)
python3 - <<'EOF'
import json, glob
from evaluation.statistical_analysis import summarize_runs, compare_all_vs_baseline

metric = "micro_f1"  # or rouge_l for Exp B
data = {}
for path in glob.glob("experiment_logs/*.json"):
    d = json.load(open(path))
    val = d.get("final_metrics", {}).get(metric)
    if val is not None:
        data.setdefault(d["config"]["strategy"] + "_" + d["config"]["backend"], []).append(val)

summary = summarize_runs(data)
for cfg, ci in summary.items():
    print(f"{cfg}: {ci}")
EOF

# Plots
python3 - <<'EOF'
from evaluation.plots import curva_epsilon_vs_f1, curva_f1_vs_alpha, convergence_curves

# Privacy-utility tradeoff (fill with your actual values)
curva_epsilon_vs_f1(
    epsilon_values=[2.1, 4.3, 8.7],
    f1_values=[0.71, 0.69, 0.64],
    baseline_f1=0.75,
    output_path="figures/epsilon_vs_f1.pdf",
)

curva_f1_vs_alpha(
    alpha_values=[0.1, 0.5, 1.0],
    f1_fedprox=[0.58, 0.63, 0.66],
    f1_fedavg=[0.52, 0.57, 0.61],
    f1_centralizado=0.75,
    output_path="figures/f1_vs_alpha.pdf",
)
EOF
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | Batch too large for GPU | Reduce `FL_BATCH_SIZE` to 4 or 2 |
| `fl_server not healthy` after 2 min | gRPC startup timeout | `docker compose restart fl_server` |
| FL silos exit without metrics | HAPI FHIR has no data | Re-run `make up-infra` after `make build-mimic` |
| `label_index not found` | BERT_LABEL_INDEX_PATH wrong | Ensure `etl_worker/data/label_index.json` exists |
| `HF token invalid` | Token expired or wrong scope | Regenerate at huggingface.co/settings/tokens |
| Silo hangs after training | Flower `start_client` deprecation warning | Expected/harmless — watch for actual error lines |
| Empty ROUGE scores | `reference_summary` missing | Ensure MIMIC-IV-Note `discharge.csv.gz` was loaded |
| `experiment_logs/*.json` missing | `save_run_json` couldn't parse server log | Check `grep "Aggregated eval" experiment_logs/*_server.log` |

### Sharing logs for debugging

```bash
# Pack all logs into a single compressed file
make logs-pack
# Produces: experiment_logs.tar.gz

# Or just the master log for a specific run
gzip -k experiment_logs/run_20260502_143000_full.log
# Share: experiment_logs/run_20260502_143000_full.log.gz
```
