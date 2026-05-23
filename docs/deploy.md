# Deploy Guide — Federated FHIR Architecture

Step-by-step: GitHub → Vast.ai A100 → MIMIC-IV → full experiment matrix.

> **No Docker required.** Services run directly as Python processes + HAPI FHIR CLI (Java).
> Vast.ai container instances block Docker-in-Docker at the kernel level. This guide avoids it entirely.

---

## Prerequisites checklist

- [x] Smoke test passes locally (`make smoke`)
- [x] HuggingFace token with Llama-3.2 access (huggingface.co/settings/tokens)
- [x] PhysioNet account with MIMIC-IV and MIMIC-IV-Note approved access
- [ ] SSH key pair (see Step 2)
- [ ] ~40 GB disk on the cloud VM
- [ ] Vast.ai account with credit balance (vastai.com)

---

## Cost estimate

| Phase | Est. time | Cost (~$1.00/h A100 PCIe) |
|-------|-----------|--------------------------|
| Setup + MIMIC scp + build-mimic | ~1.5 h | ~$1.50 |
| `uv sync` + HAPI start + ETL load | ~30 min | ~$0.50 |
| Mini-validation (Step 6.5) | ~15 min | ~$0.25 |
| `run_nodocker.sh --exp all` | ~16 h | ~$16 |
| **Total cloud** | **~18 h** | **~$18** |

> **Shut down the instance immediately after experiments.** Post-processing runs locally — no GPU needed.

---

## Step 0 — Local smoke test (MUST pass before going to the cloud)

```bash
make smoke
grep -E "Traceback|Error" experiment_logs/run_*_smoke.log | wc -l   # expect 0
ls experiment_logs/*.json | wc -l   # expect 12+ files
```

---

## Step 1 — Push to GitHub

### 1.1 Security check

```bash
git status
# Must NOT be staged: physionet.org/  .env  experiment_logs/*.log
```

### 1.2 Commit and push

```bash
git add README.md Makefile run_experiments.sh run_nodocker.sh \
        ai_client/ etl_worker/ fl_server/ evaluation/ \
        docker-compose.yml pyproject.toml uv.lock .gitignore docs/

git status   # review once more

git commit -m "feat: complete architecture — eval pipeline, README"
git push origin main
```

---

## Step 2 — Provision a Vast.ai instance

### 2.1 Account setup

1. Go to **vastai.com** → Create account → **Billing** → Add credit card
2. Add balance (e.g. $30) — charges are deducted as the instance runs

### 2.2 SSH key

```bash
# On your LOCAL machine:
ssh-keygen -t ed25519 -f ~/.ssh/vastai_key -N ""
cat ~/.ssh/vastai_key.pub   # copy this line
```

Go to Vast.ai → **Account** → **SSH Keys** → paste → Save.

### 2.3 Finding an instance

1. Go to **Search** → type `A100` in the GPU box
2. Filter requirements:

   | Field | Minimum | Reason |
   |-------|---------|--------|
   | GPU RAM | ≥ 40 GB | Model + LoRA + activations |
   | Disk Space | ≥ 40 GB | MIMIC + venv + model cache |
   | Max Duration | ≥ 20 h | Full run takes ~16–18 h |
   | Reliability | ≥ 99% | Risk of host dropping mid-run |
   | CUDA | ≥ 12.1 | Required for BitsAndBytes NF4 |

3. Sort by **Reliability** descending, then price
4. **A100 PCIe is as good as SXM4 for this workload** — the bottleneck is LoRA compute, not memory bandwidth

5. Select **`PyTorch (Vast)`** template (`vastai/pytorch`, CUDA 12.x)
6. Set **Disk (GB)** to `40` (minimum) or `60` (recommended)
7. Leave Docker Options blank — Docker is not used
8. Click **Rent**

> **How to read a listing:**
> ```
> Reliability 99.95%   ← must be ≥ 99%
> Max Duration 12 days ← must be ≥ 20 h
> DLPerf 127.5         ← higher = faster training
> $0.703/hr            ← billed per hour
> ```

### 2.4 Connecting

After the instance starts (status = Running), click it → **Connect** tab:

```bash
ssh -p <PORT> root@<HOST_IP> -i ~/.ssh/vastai_key
```

The port (`-p`) is unique to each instance and is never 22.

---

## Step 3 — Set up the instance

### 3.1 Verify GPU

```bash
nvidia-smi
# Expected: A100 with VRAM listed. If blank, wait 30s and retry.
```

### 3.2 Install Java (for HAPI FHIR)

```bash
apt-get update -q && apt-get install -y openjdk-17-jre-headless
java -version   # expected: openjdk 17.x
```

### 3.3 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version
```

### 3.4 Clone the repository

```bash
git clone https://github.com/FBRosito/federated-fhir-architecture.git
cd federated-fhir-architecture
```

### 3.5 Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your HuggingFace token:

```
HF_TOKEN=hf_YOUR_TOKEN_HERE
```

To edit without a GUI:
```bash
nano .env   # Ctrl+O to save, Ctrl+X to exit
```

### 3.6 Install Python dependencies

This downloads PyTorch CUDA + all packages into `.venv/` (~10 GB, ~15–20 min):

```bash
uv sync --frozen
```

---

## Step 4 — Transfer MIMIC-IV

MIMIC-IV must be placed **inside the project directory** at `physionet.org/files/...` — this path is already in `.gitignore`.

**From your LOCAL machine** (replace PORT and HOST_IP with your instance values):

```bash
# Create directories inside the project:
ssh -p <PORT> root@<HOST_IP> -i ~/.ssh/vastai_key \
  "mkdir -p ~/federated-fhir-architecture/physionet.org/files/mimiciv/3.1/hosp \
             ~/federated-fhir-architecture/physionet.org/files/mimic-iv-note/2.2/note"

# Copy MIMIC-IV hosp files (~4 GB, ~5–10 min):
scp -P <PORT> -i ~/.ssh/vastai_key \
    physionet.org/files/mimiciv/3.1/hosp/{diagnoses_icd,d_icd_diagnoses,admissions,patients,services,procedures_icd,d_icd_procedures,prescriptions,microbiologyevents,labevents}.csv.gz \
    root@<HOST_IP>:~/federated-fhir-architecture/physionet.org/files/mimiciv/3.1/hosp/

# Copy MIMIC-IV-Note (~500 MB):
scp -P <PORT> -i ~/.ssh/vastai_key \
    physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz \
    root@<HOST_IP>:~/federated-fhir-architecture/physionet.org/files/mimic-iv-note/2.2/note/
```

**Alternative — download directly from PhysioNet** (if you don't have the data locally):

```bash
# On the cloud instance, inside ~/federated-fhir-architecture/:
PUSER="YOUR_PHYSIONET_USERNAME"
HOSP="https://physionet.org/files/mimiciv/3.1/hosp"
NOTE="https://physionet.org/files/mimic-iv-note/2.2/note"

mkdir -p physionet.org/files/mimiciv/3.1/hosp
mkdir -p physionet.org/files/mimic-iv-note/2.2/note

cd physionet.org/files/mimiciv/3.1/hosp
for f in diagnoses_icd.csv.gz d_icd_diagnoses.csv.gz admissions.csv.gz \
          patients.csv.gz services.csv.gz procedures_icd.csv.gz \
          d_icd_procedures.csv.gz prescriptions.csv.gz \
          microbiologyevents.csv.gz labevents.csv.gz; do
  wget -N --user "$PUSER" --ask-password "$HOSP/$f"
done

cd ~/federated-fhir-architecture/physionet.org/files/mimic-iv-note/2.2/note
wget -N --user "$PUSER" --ask-password "$NOTE/discharge.csv.gz"
```

---

## Step 5 — Build FHIR bundles from MIMIC-IV

Reads MIMIC-IV CSVs and writes pre-assembled FHIR bundles to `etl_worker/data/bundles/`. Runtime: ~15–30 min.

```bash
cd ~/federated-fhir-architecture

make build-mimic MAX_ADMISSIONS=10000 N_SILOS=5 DIRICHLET_ALPHA=0.5 BENCHMARK=top50 ICD_VERSION=icd10

# Verify:
ls etl_worker/data/bundles/ | wc -l          # expect ~10000
ls etl_worker/data/label_index.json          # must exist for BERT experiment
```

---

## Step 6 — Start HAPI FHIR and load data

The `run_nodocker.sh` script handles this automatically at startup. To verify manually:

```bash
# Download HAPI FHIR CLI and start server (first time: ~1 min download)
curl -fsSL "https://github.com/hapifhir/hapi-fhir/releases/download/v8.10.0/hapi-fhir-8.10.0-cli.zip" \
     -o /tmp/hapi-cli.zip
unzip -q /tmp/hapi-cli.zip -d /tmp/hapi-cli/
find /tmp/hapi-cli -name "hapi*.jar" | head -1 | xargs -I{} cp {} hapi-fhir-cli.jar

nohup java -jar hapi-fhir-cli.jar run-server --fhir-version R4 --port 8080 \
  > experiment_logs/hapi_fhir.log 2>&1 &

# Wait ~2 min for HAPI FHIR to boot, then verify:
curl -s http://localhost:8080/fhir/metadata | python3 -m json.tool | head -5
# Expected: {"resourceType": "CapabilityStatement", ...}

# Load FHIR bundles into HAPI FHIR:
FHIR_SERVER_URL=http://localhost:8080/fhir \
ETL_BUNDLES_PATH=$(pwd)/etl_worker/data/bundles \
uv run etl-worker

# Verify data loaded:
curl -s "http://localhost:8080/fhir/Patient?_summary=count" | python3 -c \
  "import sys,json; print('patients:', json.load(sys.stdin)['total'])"
# Expected: patients: ~10000
```

---

## Step 6.5 — Mini-validation (REQUIRED before Step 7)

Run this before the full matrix. If something is wrong, you lose ~$0.05 — not $10.

```bash
cd ~/federated-fhir-architecture

export FL_BATCH_SIZE=8 FL_GRADIENT_ACCUM_STEPS=8 FL_PARALLEL_GPU=true

bash run_nodocker.sh --exp A --smoke
```

**Expected output (last lines):**
```
ALL EXPERIMENTS COMPLETED — exp=A | mode=smoke | seeds=42
JSON files: 7 runs saved
```

**Check the JSON was written:**
```bash
ls experiment_logs/*.json | wc -l   # expect 7
python3 -c "
import json, glob
for p in sorted(glob.glob('experiment_logs/*.json'))[:3]:
    d = json.load(open(p))
    print(d['tag'], d.get('final_metrics',{}))
"
```

If the smoke test fails, read the master log to diagnose:
```bash
tail -100 experiment_logs/run_*_smoke.log
```

---

## Step 7 — Run the full experiment matrix

### 7.1 Configure hardware for RTX 4090 (recommended) or A100

```bash
# RTX 4090 (24 GB) — optimal for Experiment A (BERT only):
export FL_BATCH_SIZE=8
export FL_GRADIENT_ACCUM_STEPS=8
export FL_PARALLEL_GPU=true

# A100 PCIe 40 GB — can push further:
# export FL_BATCH_SIZE=16
# export FL_GRADIENT_ACCUM_STEPS=4
# export FL_PARALLEL_GPU=true
```

### 7.2 Launch in a screen session (IMPORTANT — survives SSH disconnect)

```bash
screen -S flexp

# Inside screen:
cd ~/federated-fhir-architecture
bash run_nodocker.sh --exp A

# Detach (keep running): Ctrl+A, D
# Reattach later:        screen -r flexp
```

### 7.3 Monitor progress (from a second SSH terminal)

```bash
# Tail the master log:
tail -f experiment_logs/run_*_full.log

# Count completed runs:
grep "Run .* completed\|COMPLETED" experiment_logs/run_*_full.log | wc -l
# 19 total (1 calibration + 3 centralised + 15 FL runs)

# Check JSONs produced so far:
ls experiment_logs/*.json 2>/dev/null | wc -l

# Watch for errors:
grep -c "Traceback\|CUDA out of memory\|killed" experiment_logs/run_*_full.log
# If > 0, investigate immediately
```

### 7.4 Estimated runtime on RTX 4090 (Experiment A only — BERT top-50)

| Phase | Runs | Est. time |
|-------|------|-----------|
| Calibration (bert) | 1 | ~10 min |
| Centralised baseline (bert, 3 seeds) | 3 | ~1.5 h |
| FL no-DP (FedProx + FedAvg, 3 seeds) | 6 | ~3 h |
| FL with DP (σ=0.5, 1.0, 2.0, 3 seeds) | 9 | ~5 h |
| **Total** | **19** | **~10 h** |

---

## Step 8 — Retrieve results and destroy the instance

### 8.1 Pack and download results

```bash
# On the cloud instance:
cd ~/federated-fhir-architecture
tar -czf experiment_logs.tar.gz experiment_logs/*.json experiment_logs/*.log

# Optional git backup:
git add experiment_logs/*.json
git commit -m "results: full experimental matrix — bert+llm, 3 seeds, all DP configs"
git push origin main
```

```bash
# From your LOCAL machine:
scp -P <PORT> -i ~/.ssh/vastai_key \
    root@<HOST_IP>:~/federated-fhir-architecture/experiment_logs.tar.gz ./

tar -xzf experiment_logs.tar.gz
ls experiment_logs/*.json | wc -l   # expect ~38
```

### 8.2 Destroy the instance

Vast.ai dashboard → My Instances → **Destroy** (not Stop — Destroy stops billing).

---

## Step 9 — Post-processing (entirely local — no GPU needed)

### 9.1 Statistical analysis + plots

```bash
cd /path/to/federated-fhir-architecture
uv sync --frozen
bash run_postprocessing.sh
# Produces: experiment_logs/statistical_summary.json + experiment_logs/figures/*.pdf
```

### 9.2 LLM-as-a-Judge (Experiment B — requires OpenRouter API key)

```bash
export OPENROUTER_API_KEY=sk-or-your_key_here

uv run python -m evaluation.llm_judge \
  --results-dir experiment_logs/ \
  --backend llm \
  --max-samples 50
```

Get a key at openrouter.ai/keys. Cost: ~$0.01–0.05 per sample.

### 9.3 Commit results

```bash
git add experiment_logs/*.json experiment_logs/figures/
git commit -m "results: statistical summary + publication figures"
git push origin main
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | Batch too large | Reduce `FL_BATCH_SIZE` to 4 or 2 |
| `fl_server failed to start` after 2 min | Port 9091 in use | `pkill -f fl-server; sleep 2` then re-run |
| HAPI FHIR not ready after 6 min | Java OOM | `tail experiment_logs/hapi_fhir.log` — add `-Xmx4g` if heap error |
| FL silos exit without metrics | HAPI FHIR has no data | Re-run ETL: `FHIR_SERVER_URL=http://localhost:8080/fhir uv run etl-worker` |
| `label_index not found` | build-mimic not run | Run Step 5 first |
| `HF token invalid` | Token expired | Regenerate at huggingface.co/settings/tokens, update `.env` |
| Silo hangs after training | Flower deprecation warning | Expected/harmless — watch for actual error lines |
| Empty ROUGE scores | MIMIC-IV-Note not loaded | Ensure `discharge.csv.gz` was transferred and ETL ran |
| `experiment_logs/*.json` missing after run | `save_run_json` couldn't parse log | `grep "Aggregated eval" experiment_logs/*_server.log` |
| HAPI FHIR loses data between runs | In-memory server restarted | `run_nodocker.sh` detects this and re-runs ETL automatically |

### Sharing logs for debugging

```bash
tar -czf experiment_logs.tar.gz experiment_logs/run_*.log experiment_logs/*.json
# Share: experiment_logs.tar.gz
```
