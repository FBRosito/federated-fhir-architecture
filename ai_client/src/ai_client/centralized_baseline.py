"""
centralized_baseline.py
-----------------------
Centralised baseline for scientific comparison with the FL system.

Routed by MODEL_BACKEND:
  'bert' → PubMedBERT + per-label attention + BCEWithLogitsLoss (Experiment A)
  other  → Llama + text-generation cross-entropy (Experiment B)

Both backends read MIMIC-IV data directly from CSVs (bypassing HAPI FHIR),
apply the same filters as mimic_builder.py, and train with a continuous
optimiser across epochs — no federated communication overhead.

Methodological differences from FL:
  - Data source: direct CSV (not FHIR)
  - Optimiser: state preserved across epochs
  - Communication: zero overhead (no gRPC rounds)
  - Partitioning: none (trains on all data at once)

Environment variables:
  MODEL_BACKEND        'bert' or 'llm' (default: 'llm')
  FL_SEED              reproducibility seed (default: 42)
  FL_MAX_EXAMPLES      example limit; 0 = no limit (default: 5000)
  FL_NUM_ROUNDS        training epochs (default: 5)
  FL_LEARNING_RATE     learning rate
  FL_BATCH_SIZE        batch size
  FL_GRADIENT_ACCUM_STEPS gradient accumulation steps
  MIMIC_HOSP_DIR       directory with MIMIC-IV/hosp CSVs
  MIMIC_NOTE_DIR       directory with discharge.csv.gz
  BERT_BENCHMARK       'top50' or 'full' (default: 'top50')
  BERT_LABEL_INDEX_PATH path to pre-built label_index.json
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai_client.fhir_consumer import TrainingExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ai_client.centralized_baseline")

# Leakage pattern identical to mimic_builder.load_notes
_LEAKAGE_RE = re.compile(
    r"(Discharge\s+(Diagnosis|Condition|Instructions)"
    r"|Assessment\s+and\s+Plan\s*:"
    r"|Impression\s*:)",
    re.IGNORECASE,
)


# ── Stratified split ───────────────────────────────────────────────────────────

def _stratified_split(
    examples: list[TrainingExample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """80/20 split stratified by primary ICD-10 code."""
    rng = np.random.default_rng(seed)
    by_code: dict[str, list[TrainingExample]] = collections.defaultdict(list)
    for ex in examples:
        by_code[ex.icd10_code].append(ex)

    train, val = [], []
    for _, group in by_code.items():
        shuffled = list(rng.permutation(group))  # type: ignore[arg-type]
        n_train  = max(1, math.floor(len(shuffled) * train_ratio))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:])

    if not val and train:
        rng.shuffle(train)
        n_move = max(1, math.ceil(len(train) * (1 - train_ratio)))
        val    = train[:n_move]
        train  = train[n_move:]

    log.info(
        "Train/eval split: %d/%d examples | %d unique ICD-10 codes",
        len(train), len(val), len(by_code),
    )
    return train, val


# ── MIMIC-IV loading (primary diagnosis — for LLM) ────────────────────────────

def load_mimic_direct(
    hosp_dir: Path,
    note_dir: Path,
    max_examples: int = 0,
    max_chars: int = 2048,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Loads MIMIC-IV for the LLM backend (primary ICD-10 diagnosis).

    Applies the same filters as mimic_builder.py:
      - seq_num == 1, icd_version == 10
      - truncation to max_chars characters
      - discard notes with a diagnosis section (leakage)
    """
    log.info("Loading primary ICD-10 diagnoses from %s...", hosp_dir)
    diag = pd.read_csv(
        hosp_dir / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    diag = diag[(diag["seq_num"] == 1) & (diag["icd_version"] == 10)].copy()
    log.info("  %d primary ICD-10 diagnoses", len(diag))

    icd_dict = pd.read_csv(
        hosp_dir / "d_icd_diagnoses.csv.gz",
        usecols=["icd_code", "icd_version", "long_title"],
    )
    icd_dict = icd_dict[icd_dict["icd_version"] == 10][["icd_code", "long_title"]]
    df = diag.merge(icd_dict, on="icd_code", how="left")

    hadm_ids = set(df["hadm_id"].unique())
    log.info("Loading discharge notes for %d admissions from %s...", len(hadm_ids), note_dir)
    notes = pd.read_csv(note_dir / "discharge.csv.gz", usecols=["hadm_id", "text"])
    notes = notes[notes["hadm_id"].isin(hadm_ids)]
    notes = notes.groupby("hadm_id", as_index=False).first()
    log.info("  %d discharge notes loaded", len(notes))

    df = df.merge(notes, on="hadm_id", how="inner")
    log.info("  %d examples after diagnosis × note join", len(df))

    examples: list[TrainingExample] = []
    skipped_leakage = 0
    for _, row in df.iterrows():
        snippet = str(row["text"])[:max_chars]
        if _LEAKAGE_RE.search(snippet):
            skipped_leakage += 1
            continue
        examples.append(TrainingExample(
            patient_ref    = f"Patient/M{int(row['subject_id'])}",
            clinical_text  = snippet,
            icd10_code     = str(row["icd_code"]),
            icd10_display  = str(row.get("long_title", "")),
            condition_id   = "",
            doc_ref_id     = "",
            partition_note = "partition_id=-1",
        ))

    log.info(
        "Valid examples: %d | discarded for leakage: %d",
        len(examples), skipped_leakage,
    )
    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]
        log.info("Limited to %d examples (FL_MAX_EXAMPLES).", max_examples)

    return _stratified_split(examples, train_ratio=0.8, seed=seed)


# ── MIMIC-IV loading (all codes per admission — for BERT) ─────────────────────

def load_mimic_direct_multilabel(
    hosp_dir: Path,
    note_dir: Path,
    max_examples: int = 0,
    max_chars: int = 2048,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Loads MIMIC-IV for the BERT backend with multi-label ground truth.

    Difference from load_mimic_direct():
      - Loads ALL ICD-10 diagnoses per admission and populates all_icd10_codes.
      - icd10_code remains the primary diagnosis (seq_num=1) as an anchor.
    """
    log.info("Loading all ICD-10 diagnoses from %s...", hosp_dir)
    diag_all = pd.read_csv(
        hosp_dir / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    diag_all = diag_all[diag_all["icd_version"] == 10].copy()
    diag_all["icd_code"] = diag_all["icd_code"].str.upper()

    # All codes per admission (multi-label ground truth)
    all_codes_map: dict[int, list[str]] = (
        diag_all.groupby("hadm_id")["icd_code"]
        .apply(list)
        .to_dict()
    )

    # Primary diagnosis as the example anchor
    primary = diag_all[diag_all["seq_num"] == 1][
        ["subject_id", "hadm_id", "icd_code"]
    ].copy()
    log.info("  %d admissions with primary ICD-10 diagnosis", len(primary))

    icd_dict = pd.read_csv(
        hosp_dir / "d_icd_diagnoses.csv.gz",
        usecols=["icd_code", "icd_version", "long_title"],
    )
    icd_dict = icd_dict[icd_dict["icd_version"] == 10][["icd_code", "long_title"]]
    icd_dict["icd_code"] = icd_dict["icd_code"].str.upper()
    df = primary.merge(icd_dict, on="icd_code", how="left")

    hadm_ids = set(df["hadm_id"].unique())
    log.info("Loading discharge notes for %d admissions from %s...", len(hadm_ids), note_dir)
    notes = pd.read_csv(note_dir / "discharge.csv.gz", usecols=["hadm_id", "text"])
    notes = notes[notes["hadm_id"].isin(hadm_ids)]
    notes = notes.groupby("hadm_id", as_index=False).first()
    log.info("  %d notes loaded", len(notes))

    df = df.merge(notes, on="hadm_id", how="inner")
    log.info("  %d examples after join", len(df))

    examples: list[TrainingExample] = []
    skipped_leakage = 0
    for _, row in df.iterrows():
        snippet = str(row["text"])[:max_chars]
        if _LEAKAGE_RE.search(snippet):
            skipped_leakage += 1
            continue
        hadm_id   = int(row["hadm_id"])
        all_codes = all_codes_map.get(hadm_id, [str(row["icd_code"]).upper()])
        examples.append(TrainingExample(
            patient_ref     = f"Patient/M{int(row['subject_id'])}",
            clinical_text   = snippet,
            icd10_code      = str(row["icd_code"]).upper(),
            icd10_display   = str(row.get("long_title", "")),
            condition_id    = "",
            doc_ref_id      = "",
            partition_note  = "partition_id=-1",
            all_icd10_codes = all_codes,
        ))

    log.info(
        "Valid examples: %d | leakage discarded: %d",
        len(examples), skipped_leakage,
    )
    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]
        log.info("Limited to %d examples.", max_examples)

    return _stratified_split(examples, train_ratio=0.8, seed=seed)


# ── Experiment B: LLM baseline (Llama + text generation) ──────────────────────

def run_centralized(
    hosp_dir: Path,
    note_dir: Path,
    model_name: str,
    max_examples: int,
    max_seq_len: int,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accum_steps: int,
    eval_accuracy: bool,
    seed: int,
    output_log: Path,
) -> None:
    """Centralised baseline for Experiment B: Llama text generation."""
    from ai_client.model_setup import apply_lora, load_quantized_model, train_continuous
    from ai_client.fl_client import _evaluate_local

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_ex, eval_ex = load_mimic_direct(
        hosp_dir, note_dir, max_examples=max_examples, max_chars=2048, seed=seed,
    )
    log.info(
        "Centralised LLM: %d train | %d eval | %d unique ICD-10 codes",
        len(train_ex), len(eval_ex),
        len({e.icd10_code for e in train_ex}),
    )

    model, tokenizer = load_quantized_model(model_name)
    model = apply_lora(model)

    log.info(
        "Starting centralised LLM training: %d epochs × %d examples.",
        num_epochs, len(train_ex),
    )
    per_epoch = train_continuous(
        model                = model,
        tokenizer            = tokenizer,
        examples             = train_ex,
        num_epochs           = num_epochs,
        learning_rate        = learning_rate,
        batch_size           = batch_size,
        gradient_accum_steps = gradient_accum_steps,
        max_length           = max_seq_len,
    )

    log.info("Final LLM evaluation on %d examples...", len(eval_ex))
    loss, n_eval, metrics = _evaluate_local(
        model            = model,
        tokenizer        = tokenizer,
        examples         = eval_ex,
        max_length       = max_seq_len,
        compute_accuracy = eval_accuracy,
    )
    log.info(
        "Final evaluation: loss=%.4f | ppl=%.2f%s",
        loss,
        metrics.get("eval_perplexity", 0.0),
        f" | acc={metrics['eval_icd10_accuracy']:.2%}" if eval_accuracy else "",
    )

    results = {
        "experiment":        "centralizado_llm",
        "model_name":        model_name,
        "seed":              seed,
        "num_epochs":        num_epochs,
        "max_examples":      max_examples,
        "max_seq_len":       max_seq_len,
        "n_train":           len(train_ex),
        "n_eval":            n_eval,
        "per_epoch_metrics": per_epoch,
        "final_eval":        {"eval_loss": loss, **metrics},
    }
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text(json.dumps(results, indent=2))
    log.info("LLM results saved to %s", output_log)


# ── Experiment A: BERT baseline (PubMedBERT + ICD-10 multi-label) ─────────────

def run_centralized_bert(
    hosp_dir: Path,
    note_dir: Path,
    max_examples: int,
    max_seq_len: int,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accum_steps: int,
    benchmark: str,
    label_index_path: str,
    seed: int,
    output_log: Path,
) -> None:
    """
    Centralised baseline for Experiment A: PubMedBERT + ICD-10 multi-label.

    Uses the same PLM-ICD architecture (Huang et al., 2022) as the FL silos,
    but trains on all data without partitioning or privacy constraints.
    """
    from torch.utils.data import DataLoader
    from ai_client.model_setup_bert import (
        BertTrainingConfig,
        build_bert_dataset,
        load_bert_model,
        train_bert_one_round,
    )
    from ai_client.fhir_consumer_bert import build_label_index, save_label_index
    from evaluation.icd_metrics import evaluate_bert_model

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_ex, eval_ex = load_mimic_direct_multilabel(
        hosp_dir, note_dir, max_examples=max_examples, max_chars=2048, seed=seed,
    )
    log.info("Centralised BERT: %d train | %d eval", len(train_ex), len(eval_ex))

    # Label index: prefer the pre-built file from mimic_builder
    # (ensures index consistency with the FL silos)
    label_index = build_label_index(
        train_ex + eval_ex, benchmark=benchmark, label_index_path=label_index_path,
    )
    num_labels = len(label_index)
    log.info("Label index: %d labels (benchmark=%s).", num_labels, benchmark)

    if label_index_path and not Path(label_index_path).exists() and label_index:
        save_label_index(label_index, label_index_path)

    # Model: PubMedBERT + LoRA (r=8) + per-label attention
    model, tokenizer = load_bert_model(num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("BERT model on %s | num_labels=%d", device, num_labels)

    # Centralised training with multiple epochs (train_bert_one_round accepts num_epochs>1)
    train_cfg = BertTrainingConfig(
        num_epochs           = num_epochs,
        learning_rate        = learning_rate,
        batch_size           = batch_size,
        gradient_accum_steps = gradient_accum_steps,
    )
    log.info(
        "Starting centralised BERT training: %d epochs × %d examples.",
        num_epochs, len(train_ex),
    )
    _, n_train, train_metrics = train_bert_one_round(
        model, tokenizer, train_ex, label_index, train_cfg, max_seq_len,
    )

    # Evaluation with Mullenbach 2018 metrics (ICD10Metrics)
    log.info("Final BERT evaluation on %d examples...", len(eval_ex))
    eval_dataset = build_bert_dataset(
        eval_ex, label_index, tokenizer, max_seq_len, num_labels=num_labels,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size = max(batch_size, 8),
        shuffle    = False,
        pin_memory = torch.cuda.is_available(),
    )
    icd_metrics = evaluate_bert_model(model, eval_loader, device, k_list=[8, 15])

    results = {
        "experiment":    "centralizado_bert",
        "model_name":    "PubMedBERT",
        "benchmark":     benchmark,
        "seed":          seed,
        "num_epochs":    num_epochs,
        "max_examples":  max_examples,
        "max_seq_len":   max_seq_len,
        "n_train":       n_train,
        "n_eval":        icd_metrics.n_samples,
        "train_metrics": train_metrics,
        "eval_metrics":  icd_metrics.to_flat_dict(),
    }
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text(json.dumps(results, indent=2))
    log.info("BERT results saved to %s", output_log)
    log.info("%s", icd_metrics)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    backend  = os.getenv("MODEL_BACKEND", "llm").lower()
    seed     = int(os.getenv("FL_SEED", "42"))
    hosp_dir = Path(os.getenv("MIMIC_HOSP_DIR", "physionet.org/files/mimiciv/3.1/hosp"))
    note_dir = Path(os.getenv("MIMIC_NOTE_DIR", "physionet.org/files/mimic-iv-note/2.2/note"))

    # FL_MAX_EXAMPLES=0 means "no limit" (can be >100k — VERY slow).
    # run_experiments.sh passes CENTRAL_MAX_EXAMPLES (default 5000) for the centralised run.
    max_examples = int(os.getenv("FL_MAX_EXAMPLES", "5000"))

    if backend == "bert":
        run_centralized_bert(
            hosp_dir             = hosp_dir,
            note_dir             = note_dir,
            max_examples         = max_examples,
            max_seq_len          = int(os.getenv("MAX_SEQ_LEN",             "512")),
            num_epochs           = int(os.getenv("FL_NUM_ROUNDS",           "5")),
            learning_rate        = float(os.getenv("FL_LEARNING_RATE",      "2e-4")),
            batch_size           = int(os.getenv("FL_BATCH_SIZE",           "8")),
            gradient_accum_steps = int(os.getenv("FL_GRADIENT_ACCUM_STEPS", "8")),
            benchmark            = os.getenv("BERT_BENCHMARK",              "top50"),
            label_index_path     = os.getenv("BERT_LABEL_INDEX_PATH",       ""),
            seed                 = seed,
            output_log           = Path(f"experiment_logs/centralizado_bert_seed{seed}.json"),
        )
    else:
        run_centralized(
            hosp_dir             = hosp_dir,
            note_dir             = note_dir,
            model_name           = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B"),
            max_examples         = max_examples,
            max_seq_len          = int(os.getenv("MAX_SEQ_LEN",             "1024")),
            num_epochs           = int(os.getenv("FL_NUM_ROUNDS",           "5")),
            learning_rate        = float(os.getenv("FL_LEARNING_RATE",      "5e-5")),
            batch_size           = int(os.getenv("FL_BATCH_SIZE",           "1")),
            gradient_accum_steps = int(os.getenv("FL_GRADIENT_ACCUM_STEPS", "64")),
            eval_accuracy        = os.getenv("FL_EVAL_ACCURACY", "false").lower() == "true",
            seed                 = seed,
            output_log           = Path(f"experiment_logs/centralizado_llm_seed{seed}.json"),
        )


if __name__ == "__main__":
    main()
