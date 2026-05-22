#!/usr/bin/env python3
"""
mimic_builder.py — Builds FHIR bundles from MIMIC-IV v3.1.

Reads the `hosp/` (and optionally `icu/`) modules of MIMIC-IV and generates a
directory of pre-assembled FHIR Transaction Bundle JSON files
(bundle_NNNNNN.json), ready for direct upload to HAPI FHIR without any further
reprocessing.

This decouples heavy MIMIC data extraction (reading ~10 GB, ~5-15 min) from
loading into HAPI FHIR (POST of JSONs in ~1-2 min), which can be repeated on
each `make clean` / `make up-infra` without additional cost.

Operation mode (selected automatically):

  1. Real notes (--note-dir provided, default):
     Uses MIMIC-IV-Note discharge notes (discharge.csv.gz) as clinical
     narrative. The first `--note-chars` characters (default 2048 ≈ 512 tokens)
     are used, capturing Chief Complaint + HPI without reaching the
     Assessment/Discharge Diagnosis section (median at 5669 chars).

  2. Synthetic narratives (fallback when --note-dir is unavailable):
     Builds narrative from structured data: demographics, labs, microbiology,
     ICD-10 procedures.

Non-IID partitioning (for simulating 2 federated silos):
  - Partition 0 ("cardiorespiratory"): primary ICD-10 diagnosis chapter I or J
  - Partition 1 ("general"):           all other chapters

Usage with real notes:
    uv run --package etl-worker python etl_worker/mimic_builder.py \\
        --mimic-dir  physionet.org/files/mimiciv/3.1 \\
        --note-dir   physionet.org/files/mimic-iv-note/2.2/note \\
        --bundles-dir etl_worker/data/bundles \\
        --max-admissions 8000

Usage with synthetic narratives (legacy):
    uv run --package etl-worker python etl_worker/mimic_builder.py \\
        --mimic-dir  physionet.org/files/mimiciv/3.1 \\
        --bundles-dir etl_worker/data/bundles \\
        --max-admissions 8000 --skip-vitals
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mimic_builder")

# ── Key lab items of interest (itemid → readable label) ───────────────────────
# Selected by clinical relevance and frequency in MIMIC-IV
KEY_LAB_ITEMS: dict[int, str] = {
    50931: "Glucose",
    50912: "Creatinine",
    51301: "White Blood Cells",
    51222: "Hemoglobin",
    50983: "Sodium",
    50971: "Potassium",
    51006: "Urea Nitrogen",
    50882: "Bicarbonate",
    50893: "Calcium",
    51265: "Platelet Count",
    50889: "C-Reactive Protein",
    50902: "Chloride",
    50813: "Lactate",
    51275: "PTT",
    50820: "pH",
}

# ── Key ICU vital sign items (chartevents) ────────────────────────────────────
KEY_VITAL_ITEMS: dict[int, str] = {
    220045: "Heart Rate",
    220050: "Arterial BP systolic",
    220051: "Arterial BP diastolic",
    220052: "Arterial BP mean",
    220179: "Non-invasive BP systolic",
    220180: "Non-invasive BP diastolic",
    220210: "Respiratory Rate",
    220277: "SpO2",
    223762: "Temperature Fahrenheit",
    223761: "Temperature Celsius",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _gz(path: Path, filename: str) -> Path:
    """Returns the full path for a .csv.gz file in the MIMIC module."""
    p = path / filename
    if not p.exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            f"Ensure --mimic-dir points to the MIMIC-IV v3.1 root directory "
            f"(must contain 'hosp/' and 'icu/' subdirectories)."
        )
    return p


def _icd10_chapter(code: str) -> str:
    """Returns the first character of the ICD-10 code (chapter)."""
    return code[0].upper() if code else "?"


def _dirichlet_partition(
    df: pd.DataFrame,
    n_silos: int,
    alpha: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assigns partition_id to each row via Dirichlet(α) distribution over ICD-10 chapters.

    Low α (0.1) → highly Non-IID: each silo is dominated by 1-2 chapters.
    High α (1.0) → near-IID: uniform distribution across silos.
    Standard in FL papers since LEAF (Caldas et al., 2019) and Hsu et al., 2019.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["partition_id"] = -1

    for chapter in df["icd_chapter"].unique():
        idx = list(df.index[df["icd_chapter"] == chapter])
        n = len(idx)
        if n == 0:
            continue

        proportions = rng.dirichlet([alpha] * n_silos)
        # Convert proportions to integer counts without loss
        counts = np.floor(proportions * n).astype(int)
        remainder = n - counts.sum()
        # Distribute remainder to silos with largest residual fraction
        residuals = (proportions * n) - counts
        top_idx = np.argsort(residuals)[::-1][:remainder]
        counts[top_idx] += 1

        rng.shuffle(idx)
        start = 0
        for silo_id, count in enumerate(counts):
            end = start + max(count, 0)
            for row_idx in idx[start:end]:
                df.at[row_idx, "partition_id"] = silo_id
            start = end

        # Residual cases from rounding: assign to the dominant silo
        unassigned = [i for i in idx if df.at[i, "partition_id"] == -1]
        dominant_silo = int(np.argmax(proportions))
        for row_idx in unassigned:
            df.at[row_idx, "partition_id"] = dominant_silo

    return df


def _calc_age(anchor_age: int, anchor_year: int, admit_year: int) -> int:
    """Approximates patient age at admission using MIMIC anchor fields."""
    return max(0, anchor_age + (admit_year - anchor_year))


def _format_gender(g: str) -> str:
    return "male" if str(g).upper().startswith("M") else "female"


def _los_hours(admittime: Any, dischtime: Any) -> float:
    """Calculates length of stay in hours."""
    try:
        a = pd.to_datetime(admittime)
        d = pd.to_datetime(dischtime)
        return max(0.0, (d - a).total_seconds() / 3600)
    except Exception:
        return 0.0


def _safe_str(val: object, fallback: str = "unknown") -> str:
    s = str(val).strip()
    return fallback if (not s or s.lower() == "nan") else s


def _build_narrative(
    row: pd.Series,
    labs: dict[int, tuple[str, str, str]],       # itemid → (label, value, unit)
    microbiology: list[str],
    procedures: list[str],
    vitals: dict[int, tuple[str, str, str]],      # itemid → (label, value, unit)
    medications: list[str] | None = None,
) -> str:
    """
    Builds a clinical narrative in English from structured data.

    Fields are organized as close as possible to a clinical progress note so
    the LLM can learn to associate the clinical context with the target ICD-10
    code.
    """
    gender_str = "male" if row["gender"] == "male" else "female"
    age = row["age"]
    admission_type     = _safe_str(row.get("admission_type"),     "unknown").title()
    admission_source   = _safe_str(row.get("admission_location"), "unknown").title()
    discharge_location = _safe_str(row.get("discharge_location"), "unknown").title()
    los = row.get("los_hours", 0.0)
    service = _safe_str(row.get("service"), "not recorded").upper()
    admit_date = str(row.get("admittime", ""))[:10]

    lines: list[str] = []

    # ── Paragraph 1: General admission data ──
    lines.append(
        f"Patient, {gender_str}, {age} years old. "
        f"Admitted via {admission_type} ({admission_source}) on {admit_date}. "
        f"Discharged to {discharge_location} after {los:.0f} hours. "
        f"Clinical service: {service}."
    )

    # ── Paragraph 2: Laboratory results ──
    if labs:
        lab_parts = []
        for itemid, (label, value, unit) in sorted(labs.items(), key=lambda x: x[1][0]):
            unit_str = f" {unit}" if unit and unit not in ("", "nan") else ""
            lab_parts.append(f"{label}: {value}{unit_str}")
        lines.append("Laboratory results: " + "; ".join(lab_parts) + ".")

    # ── Paragraph 3: Vital signs (ICU) ──
    if vitals:
        vital_parts = []
        for itemid, (label, value, unit) in sorted(vitals.items(), key=lambda x: x[1][0]):
            unit_str = f" {unit}" if unit and unit not in ("", "nan") else ""
            vital_parts.append(f"{label}: {value}{unit_str}")
        lines.append("Vital signs (ICU): " + "; ".join(vital_parts) + ".")

    # ── Paragraph 4: Microbiology ──
    if microbiology:
        lines.append("Microbiology findings: " + " ".join(microbiology))

    # ── Paragraph 5: Procedures ──
    if procedures:
        lines.append("Procedures performed: " + "; ".join(procedures) + ".")

    # ── Paragraph 6: Medications ──
    if medications:
        lines.append("Medications administered: " + "; ".join(medications) + ".")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_icd_codes(hosp_dir: Path, hadm_ids: set[int]) -> dict[int, list[str]]:
    """
    Loads ALL ICD-10 codes (all seq_num values) for the given admissions.

    Returns:
        {hadm_id: [primary_code, secondary1, secondary2, ...]} ordered by seq_num.
    """
    log.info("Loading all ICD-10 codes per admission (multi-label)...")
    diag_all = pd.read_csv(
        _gz(hosp_dir, "diagnoses_icd.csv.gz"),
        usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    diag_all = diag_all[
        (diag_all["icd_version"] == 10) & diag_all["hadm_id"].isin(hadm_ids)
    ].copy()
    diag_all = diag_all.sort_values(["hadm_id", "seq_num"])

    result: dict[int, list[str]] = {}
    for _, row in diag_all.iterrows():
        hid = int(row["hadm_id"])
        result.setdefault(hid, []).append(str(row["icd_code"]))

    log.info("  All ICD-10 codes loaded: %d admissions, mean %.1f codes/admission.",
             len(result), sum(len(v) for v in result.values()) / max(len(result), 1))
    return result


def _assign_temporal_split_percentile(df: "pd.DataFrame") -> "pd.Series":
    """Assigns temporal split (train/val/test) by chronological rank percentile.

    MIMIC-IV v3.1 applies per-patient date-shifting (real 2008-2022 dates →
    approx. 2105-2214), making absolute year cutoffs unreliable.
    Instead, sort by admittime and split by row-index position: 80% train,
    10% val, 10% test. This produces a reproducible temporal holdout regardless
    of the shifted date range.
    """
    import numpy as np
    n = len(df)
    train_end = int(0.80 * n)
    val_end   = int(0.90 * n)
    idx = df.sort_values("admittime").index
    splits = pd.Series("test", index=df.index)
    splits.loc[idx[:train_end]] = "train"
    splits.loc[idx[train_end:val_end]] = "val"
    return splits


def load_base_data(
    hosp_dir: Path,
    max_admissions: int,
    seed: int,
    n_silos: int = 2,
    dirichlet_alpha: float = 0.0,
    icd_version: str = "all",
    benchmark: str = "full",
    split_output: str | None = None,
    label_index_output: str | None = None,
) -> pd.DataFrame:
    """
    Loads and joins base tables: ICD-10 diagnoses, admissions, patients, and
    clinical service. Returns a DataFrame with one row per selected admission,
    with Non-IID partitioning applied (Dirichlet or legacy by specialty).

    Args:
        hosp_dir:            MIMIC-IV hosp/ directory.
        max_admissions:      Total number of admissions to select.
        seed:                Seed for reproducibility.
        n_silos:             Number of federated silos (default 2 = legacy behaviour).
        dirichlet_alpha:     α for Dirichlet partitioning (0.0 = legacy by specialty).
        icd_version:         "icd10" filters admissions with ICD-10-CM (admittime >= 2015-10-01);
                             "all" includes all versions.
        benchmark:           "top50" keeps only the 50 most frequent ICD-10 codes;
                             "full" keeps all codes with ≥10 occurrences (Mullenbach 2018
                             default); "none" applies no filter.
        split_output:        Path to save train/val/test split metadata as JSON.
                             None = do not save.
        label_index_output:  Path to save the global label index (code → index) as JSON.
    """
    log.info("Loading primary ICD-10 diagnoses...")
    diag = pd.read_csv(
        _gz(hosp_dir, "diagnoses_icd.csv.gz"),
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    # Primary ICD-10 diagnosis only for the base line
    diag = diag[(diag["icd_version"] == 10) & (diag["seq_num"] == 1)].copy()
    log.info("  %d admissions with primary ICD-10 diagnosis.", len(diag))

    log.info("Loading ICD-10 dictionary...")
    icd_dict = pd.read_csv(
        _gz(hosp_dir, "d_icd_diagnoses.csv.gz"),
        usecols=["icd_code", "icd_version", "long_title"],
        dtype={"icd_code": str},
    )
    icd_dict = icd_dict[icd_dict["icd_version"] == 10][["icd_code", "long_title"]]

    log.info("Loading admissions...")
    adm = pd.read_csv(
        _gz(hosp_dir, "admissions.csv.gz"),
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "admission_location", "discharge_location",
            "language", "insurance", "admit_provider_id",
        ],
        parse_dates=["admittime", "dischtime"],
    )

    log.info("Loading patient data...")
    pts = pd.read_csv(
        _gz(hosp_dir, "patients.csv.gz"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )

    log.info("Loading clinical services...")
    svc = pd.read_csv(
        _gz(hosp_dir, "services.csv.gz"),
        usecols=["subject_id", "hadm_id", "transfertime", "curr_service"],
        parse_dates=["transfertime"],
    )
    # Keep only the first service for each admission
    svc = (
        svc.sort_values("transfertime")
        .groupby("hadm_id", as_index=False)
        .first()[["hadm_id", "curr_service"]]
    )

    # ── Main join ──
    df = (
        diag
        .merge(icd_dict, on="icd_code", how="left")
        .merge(adm, on=["subject_id", "hadm_id"], how="inner")
        .merge(pts, on="subject_id", how="inner")
        .merge(svc, on="hadm_id", how="left")
    )
    df["long_title"] = df["long_title"].fillna(df["icd_code"])

    # ── ICD version filter — icd10 uses version code, not absolute date ──
    # MIMIC-IV v3.1 applies per-patient date-shifting; absolute date cutoffs
    # (e.g. 2015-10-01) become wrong after shifting. Filter by the icd_version
    # column instead: value 10 = ICD-10-CM, value 9 = ICD-9-CM.
    if icd_version == "icd10":
        before = len(df)
        df = df.copy()  # already filtered to icd_version==10 at diag loading step
        log.info(
            "ICD-10 filter: %d admissions (filtered by icd_version==10 at load, no date cutoff).",
            before,
        )

    # ── Temporal split: chronological 80/10/10 by admittime rank ──
    # Percentile-based split is robust to MIMIC-IV v3.1 per-patient date-shifting.
    df["temporal_split"] = _assign_temporal_split_percentile(df)
    for split_name in ("train", "val", "test"):
        log.info("Split %s: %d admissions.", split_name, (df["temporal_split"] == split_name).sum())

    # ── Benchmark filter (Mullenbach 2018 methodology) ──
    if benchmark in ("top50", "full"):
        code_counts = df["icd_code"].value_counts()
        if benchmark == "top50":
            top50_series = code_counts.head(50)
            valid_codes = set(top50_series.index)
            log.info("Benchmark top50: %d most frequent ICD-10 codes.", len(valid_codes))
            # Export global label index (code → index) for cross-silo consistency
            if label_index_output:
                import json as _json
                label_index = {code: idx for idx, code in enumerate(top50_series.index)}
                out_li = Path(label_index_output)
                out_li.parent.mkdir(parents=True, exist_ok=True)
                out_li.write_text(_json.dumps(label_index, indent=2))
                log.info("Top50 label index saved to: %s", out_li)
        else:  # full
            full_series = code_counts[code_counts >= 10]
            valid_codes = set(full_series.index)
            log.info("Benchmark full: %d ICD-10 codes with ≥10 occurrences.", len(valid_codes))
            if label_index_output:
                import json as _json
                label_index = {code: idx for idx, code in enumerate(full_series.index)}
                out_li = Path(label_index_output)
                out_li.parent.mkdir(parents=True, exist_ok=True)
                out_li.write_text(_json.dumps(label_index, indent=2))
                log.info("Full label index saved to: %s", out_li)
        before = len(df)
        df = df[df["icd_code"].isin(valid_codes)].copy()
        log.info("After benchmark '%s' filter: %d → %d admissions.", benchmark, before, len(df))

    # ── Compute admission age and length of stay ──
    df["admit_year"] = df["admittime"].dt.year
    df["age"] = df.apply(
        lambda r: _calc_age(r["anchor_age"], r["anchor_year"], r["admit_year"]),
        axis=1,
    )
    df["los_hours"] = df.apply(
        lambda r: _los_hours(r["admittime"], r["dischtime"]), axis=1
    )
    df["gender"] = df["gender"].apply(_format_gender)
    df["icd_chapter"] = df["icd_code"].apply(_icd10_chapter)

    # ── Non-IID partitioning ──
    if dirichlet_alpha > 0.0:
        log.info(
            "Dirichlet partitioning: α=%.2f | %d silos", dirichlet_alpha, n_silos
        )
        df = _dirichlet_partition(df, n_silos=n_silos, alpha=dirichlet_alpha, seed=seed)
        df["partition_label"] = df["partition_id"].apply(lambda i: f"silo_{i}")
    else:
        # Legacy: 2 silos by ICD-10 chapter (cardiorespiratory vs general)
        log.info("Legacy partitioning: 2 silos by specialty.")
        df["partition_id"] = df["icd_chapter"].apply(
            lambda c: 0 if c in ("I", "J") else 1
        )
        df["partition_label"] = df["partition_id"].map(
            {0: "cardiorespiratory", 1: "general"}
        )
        n_silos = 2

    # ── Balanced sampling across silos ──
    rng = random.Random(seed)
    per_silo = max(1, max_admissions // n_silos)
    groups = []
    for pid in range(n_silos):
        subset = df[df["partition_id"] == pid]
        n = min(per_silo, len(subset))
        if n < per_silo:
            log.warning(
                "Silo %d has only %d admissions (requested %d).", pid, n, per_silo
            )
        if n > 0:
            sampled_idx = rng.sample(list(subset.index), n)
            groups.append(subset.loc[sampled_idx])

    selected = pd.concat(groups).reset_index(drop=True)

    # ── Load all ICD-10 codes per admission (multi-label ground truth) ──
    hadm_ids = set(selected["hadm_id"].astype(int))
    all_codes_map = load_all_icd_codes(hosp_dir, hadm_ids)
    selected["all_icd_codes"] = selected["hadm_id"].astype(int).map(
        lambda hid: ",".join(all_codes_map.get(hid, []))
    )

    dist_str = " | ".join(
        f"silo {pid}: {(selected['partition_id'] == pid).sum()}"
        for pid in range(n_silos)
    )
    log.info("Selected admissions: %d (%s)", len(selected), dist_str)

    # ── Export temporal split metadata (optional) ──
    if split_output:
        import json as _json
        split_meta: dict = {
            "benchmark":  benchmark,
            "icd_version": icd_version,
            "total":       int(len(selected)),
            "n_train":     int((selected["temporal_split"] == "train").sum()),
            "n_val":       int((selected["temporal_split"] == "val").sum()),
            "n_test":      int((selected["temporal_split"] == "test").sum()),
            "hadm_splits": {
                str(int(row["hadm_id"])): row["temporal_split"]
                for _, row in selected.iterrows()
            },
        }
        out_path = Path(split_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(split_meta, indent=2), encoding="utf-8")
        log.info("Temporal split metadata saved to: %s", out_path)

    return selected


def load_labs(hosp_dir: Path, hadm_ids: set[int]) -> dict[int, dict[int, tuple[str, str, str]]]:
    """
    Reads labevents.csv.gz in chunks and returns the latest lab per (hadm_id, itemid).

    Returns:
        {hadm_id: {itemid: (label, value, unit)}}
    """
    log.info("Loading lab events (file ~2.5 GB, processing in chunks)...")
    result: dict[int, dict[int, tuple[str, str, str]]] = {}
    key_items = set(KEY_LAB_ITEMS)
    chunk_size = 500_000
    chunks_read = 0

    for chunk in pd.read_csv(
        _gz(hosp_dir, "labevents.csv.gz"),
        usecols=["hadm_id", "itemid", "charttime", "value", "valueuom"],
        chunksize=chunk_size,
        parse_dates=["charttime"],
        low_memory=False,
    ):
        chunks_read += 1
        if chunks_read % 50 == 0:
            log.info("  ... processed %d M lab rows.", chunks_read // 2)

        filtered = chunk[
            chunk["hadm_id"].isin(hadm_ids) & chunk["itemid"].isin(key_items)
        ]
        if filtered.empty:
            continue

        for _, row in filtered.iterrows():
            hid = int(row["hadm_id"])
            iid = int(row["itemid"])
            val = str(row["value"]) if pd.notna(row["value"]) else ""
            unit = str(row["valueuom"]) if pd.notna(row["valueuom"]) else ""
            label = KEY_LAB_ITEMS[iid]

            if hid not in result:
                result[hid] = {}
            # Keep the most recent measurement (last in time order)
            # chunks are in ascending temporal order — replacing always keeps the latest
            result[hid][iid] = (label, val, unit)

    log.info("  Labs loaded for %d admissions.", len(result))
    return result


def load_microbiology(hosp_dir: Path, hadm_ids: set[int]) -> dict[int, list[str]]:
    """
    Reads microbiologyevents.csv.gz and returns relevant findings per admission.

    Returns:
        {hadm_id: [finding_text, ...]}
    """
    log.info("Loading microbiology events...")
    df = pd.read_csv(
        _gz(hosp_dir, "microbiologyevents.csv.gz"),
        usecols=["hadm_id", "spec_type_desc", "test_name", "org_name", "interpretation"],
        low_memory=False,
    )
    df = df[df["hadm_id"].isin(hadm_ids)]
    result: dict[int, list[str]] = {}

    for _, row in df.iterrows():
        hid = int(row["hadm_id"])
        spec = str(row["spec_type_desc"]) if pd.notna(row["spec_type_desc"]) else ""
        test = str(row["test_name"]) if pd.notna(row["test_name"]) else ""
        org  = str(row["org_name"]) if pd.notna(row["org_name"]) else ""
        interp = str(row["interpretation"]) if pd.notna(row["interpretation"]) else ""

        if org and org.lower() not in ("", "nan", "no growth"):
            text = f"{spec} — {test}: {org}"
            if interp and interp.lower() not in ("", "nan"):
                text += f" ({interp})"
        elif interp and interp.lower() not in ("", "nan"):
            text = f"{spec} — {test}: {interp}"
        else:
            continue  # no relevant finding

        result.setdefault(hid, []).append(text)

    log.info("  Microbiology loaded for %d admissions.", len(result))
    return result


def load_procedures(hosp_dir: Path, hadm_ids: set[int]) -> dict[int, list[str]]:
    """
    Reads procedures_icd.csv.gz (ICD-10) and returns descriptions per admission.

    Returns:
        {hadm_id: [procedure_desc, ...]}
    """
    log.info("Loading ICD-10 procedures...")
    proc = pd.read_csv(
        _gz(hosp_dir, "procedures_icd.csv.gz"),
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    proc = proc[(proc["icd_version"] == 10) & proc["hadm_id"].isin(hadm_ids)]

    d_proc = pd.read_csv(
        _gz(hosp_dir, "d_icd_procedures.csv.gz"),
        usecols=["icd_code", "icd_version", "long_title"],
        dtype={"icd_code": str},
    )
    d_proc = d_proc[d_proc["icd_version"] == 10][["icd_code", "long_title"]]

    merged = proc.merge(d_proc, on="icd_code", how="left")
    merged["long_title"] = merged["long_title"].fillna(merged["icd_code"])

    result: dict[int, list[str]] = {}
    for _, row in merged.iterrows():
        result.setdefault(int(row["hadm_id"]), []).append(str(row["long_title"]))

    log.info("  Procedures loaded for %d admissions.", len(result))
    return result


def load_medications(hosp_dir: Path, hadm_ids: set[int], max_per_admission: int = 15) -> dict[int, list[str]]:
    """
    Reads prescriptions.csv.gz and returns up to `max_per_admission` distinct
    medications per admission (brand and generic names, no duplicates).

    Used by Experiment B (discharge summary) to enrich the structured prompt.

    Returns:
        {hadm_id: [drug_name_1, drug_name_2, ...]}
    """
    log.info("Loading prescriptions (prescriptions.csv.gz)...")
    try:
        rx = pd.read_csv(
            _gz(hosp_dir, "prescriptions.csv.gz"),
            usecols=["hadm_id", "drug", "route", "dose_val_rx", "dose_unit_rx"],
            low_memory=False,
        )
    except FileNotFoundError:
        log.warning("prescriptions.csv.gz not found — medications skipped.")
        return {}

    rx = rx[rx["hadm_id"].isin(hadm_ids)].copy()
    rx["drug"] = rx["drug"].fillna("").astype(str).str.strip()
    rx = rx[rx["drug"] != ""]

    result: dict[int, list[str]] = {}
    for hid, group in rx.groupby("hadm_id"):
        # Deduplicate by drug name, keeping dose when available
        seen: set[str] = set()
        meds: list[str] = []
        for _, r in group.iterrows():
            name = str(r["drug"]).strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            dose = str(r["dose_val_rx"]).strip()
            unit = str(r["dose_unit_rx"]).strip()
            route = str(r["route"]).strip()
            parts = [name]
            if dose and dose not in ("", "nan"):
                parts.append(dose + (" " + unit if unit and unit != "nan" else ""))
            if route and route not in ("", "nan"):
                parts.append(f"({route})")
            meds.append(" ".join(p for p in parts if p))
            if len(meds) >= max_per_admission:
                break
        result[int(hid)] = meds

    log.info("  Medications loaded for %d admissions.", len(result))
    return result


def load_vitals(icu_dir: Path, hadm_ids: set[int]) -> dict[int, dict[int, tuple[str, str, str]]]:
    """
    Reads icu/icustays.csv.gz to map hadm_id → stay_id, then reads
    icu/chartevents.csv.gz in chunks to extract recent vital signs.

    Returns:
        {hadm_id: {itemid: (label, value, unit)}}
    """
    log.info("Loading ICU stays...")
    try:
        stays = pd.read_csv(
            _gz(icu_dir, "icustays.csv.gz"),
            usecols=["subject_id", "hadm_id", "stay_id"],
        )
    except FileNotFoundError:
        log.warning("icu/icustays.csv.gz not found — ICU vital signs skipped.")
        return {}

    stays = stays[stays["hadm_id"].isin(hadm_ids)]
    if stays.empty:
        return {}

    stay_ids = set(stays["stay_id"])
    hadm_by_stay = dict(zip(stays["stay_id"], stays["hadm_id"]))
    key_items = set(KEY_VITAL_ITEMS)

    log.info("Loading ICU vital signs (%d stays)...", len(stay_ids))
    result: dict[int, dict[int, tuple[str, str, str]]] = {}

    try:
        for chunk in pd.read_csv(
            _gz(icu_dir, "chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "charttime", "value", "valueuom"],
            chunksize=500_000,
            parse_dates=["charttime"],
            low_memory=False,
        ):
            filtered = chunk[
                chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(key_items)
            ]
            for _, row in filtered.iterrows():
                sid = int(row["stay_id"])
                iid = int(row["itemid"])
                hid = hadm_by_stay.get(sid)
                if hid is None:
                    continue
                val = str(row["value"]) if pd.notna(row["value"]) else ""
                unit = str(row["valueuom"]) if pd.notna(row["valueuom"]) else ""
                label = KEY_VITAL_ITEMS[iid]
                result.setdefault(hid, {})[iid] = (label, val, unit)
    except FileNotFoundError:
        log.warning("icu/chartevents.csv.gz not found — ICU vital signs skipped.")

    log.info("  Vital signs loaded for %d admissions.", len(result))
    return result


def load_notes(
    note_dir: Path,
    hadm_ids: set[int],
    max_chars: int,
    skip_leakage_filter: bool = False,
) -> dict[int, str]:
    """
    Loads discharge notes from MIMIC-IV-Note and returns the first `max_chars`
    characters of each note.

    skip_leakage_filter=False (default): discards notes where the
    "Assessment/Discharge Diagnosis" section appears before `max_chars` chars.
    Use for input notes (avoid ICD-10 leakage during training).

    skip_leakage_filter=True: loads the full note WITHOUT leakage filter.
    Use for the ROUGE/BERTScore reference in Experiment B, where the full
    discharge note including the diagnosis section is needed.

    Returns:
        {hadm_id: text}
    """
    path = note_dir / "discharge.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Discharge notes not found at: {path}\n"
            f"Check --note-dir and that MIMIC-IV-Note has been downloaded."
        )

    log.info("Loading discharge notes from %s ...", path)
    df = pd.read_csv(path, usecols=["hadm_id", "text"])
    df = df[df["hadm_id"].isin(hadm_ids)].copy()

    # One note per admission (discharge.csv already has 1:1 hadm_id:note)
    df = df.groupby("hadm_id", as_index=False).first()

    _LEAKAGE_RE = re.compile(
        r"(Discharge\s+(Diagnosis|Condition|Instructions)|Assessment\s+and\s+Plan\s*:|Impression\s*:)",
        re.IGNORECASE,
    )

    result: dict[int, str] = {}
    skipped = 0
    for _, row in df.iterrows():
        text = str(row["text"])
        snippet = text[:max_chars]
        if not skip_leakage_filter and _LEAKAGE_RE.search(snippet):
            skipped += 1
            continue
        result[int(row["hadm_id"])] = snippet

    if skipped:
        log.warning(
            "%d notes discarded — diagnosis section appears before %d chars.",
            skipped, max_chars,
        )
    log.info("  Notes loaded for %d admissions.", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FHIR bundle assembly and serialization
# ─────────────────────────────────────────────────────────────────────────────

def _build_row_series(
    row: pd.Series,
    labs: dict[int, dict[int, tuple]],
    microbiology: dict[int, list[str]],
    procedures: dict[int, list[str]],
    vitals: dict[int, dict[int, tuple]],
    notes: dict[int, str] | None = None,
    medications: dict[int, list[str]] | None = None,
    full_notes: dict[int, str] | None = None,
) -> pd.Series:
    """
    Builds a pd.Series with the schema expected by etl_pipeline builders.

    Additional fields for Experiment B (discharge summary):
      - medications_text: admission medications (for structured prompt)
      - discharge_summary: full discharge note (ROUGE/BERTScore reference)
    """
    hid = int(row["hadm_id"])
    sid = int(row["subject_id"])

    hid_medications = medications.get(hid, []) if medications else []

    if notes and hid in notes:
        # Real notes mode: narrative = truncated note (ICD coder input)
        narrative = notes[hid]
    else:
        narrative = _build_narrative(
            row,
            labs=labs.get(hid, {}),
            microbiology=microbiology.get(hid, []),
            procedures=procedures.get(hid, []),
            vitals=vitals.get(hid, {}),
            medications=hid_medications,
        )

    discharge_summary = ""
    if full_notes and hid in full_notes:
        discharge_summary = full_notes[hid]

    birth_year = int(row["anchor_year"]) - int(row["anchor_age"])
    try:
        record_date = pd.Timestamp(row["admittime"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        record_date = "2000-01-01T00:00:00Z"

    return pd.Series({
        "patient_id":         f"M{sid}",
        "patient_name":       f"Patient {sid}",
        "gender":             row["gender"],
        "birth_date":         f"{birth_year}-01-01",
        "record_date":        record_date,
        "raw_diagnosis":      str(row["long_title"]),
        "clinical_text":      narrative,
        "practitioner":       str(row.get("admit_provider_id", "")) or "unknown",
        "partition_id":       int(row["partition_id"]),
        "partition_label":    str(row["partition_label"]),
        "icd_code":           str(row["icd_code"]),
        "all_icd_codes":      str(row.get("all_icd_codes", "")),
        # Experiment B fields (discharge summary)
        "medications_text":   "; ".join(hid_medications) if hid_medications else "",
        "discharge_summary":  discharge_summary,
    })


def write_fhir_bundles(
    base: pd.DataFrame,
    labs: dict[int, dict[int, tuple]],
    microbiology: dict[int, list[str]],
    procedures: dict[int, list[str]],
    vitals: dict[int, dict[int, tuple]],
    bundles_dir: Path,
    notes: dict[int, str] | None = None,
    medications: dict[int, list[str]] | None = None,
    full_notes: dict[int, str] | None = None,
) -> int:
    """
    Builds FHIR Transaction Bundles from MIMIC data and serializes them to disk.

    Each admission generates a `bundle_NNNNNN.json` file with 4 resources:
    Patient, Condition, Composition, and DocumentReference.

    Returns:
        Number of bundles generated.
    """
    from etl_worker.etl_pipeline import (
        build_patient,
        build_condition,
        build_composition,
        build_document_reference,
        build_discharge_summary_doc_ref,
        build_transaction_bundle,
    )

    bundles_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(bundles_dir.glob("bundle_*.json"))
    if existing:
        log.info("Removing %d existing bundles from %s...", len(existing), bundles_dir)
        for f in existing:
            f.unlink()

    total = len(base)
    by_partition: dict[str, int] = {}

    for i, (_, row) in enumerate(base.iterrows()):
        series = _build_row_series(row, labs, microbiology, procedures, vitals, notes, medications, full_notes)

        patient_urn, patient      = build_patient(series)
        cond_urn,    condition    = build_condition(series, patient_urn)
        comp_urn,    composition  = build_composition(series, patient_urn, cond_urn)
        doc_urn,     doc_ref      = build_document_reference(series, patient_urn, comp_urn)

        # Discharge summary DocumentReference (LOINC 18842-5) for Experiment B.
        # Present only when the full discharge note is available.
        discharge_urn = discharge_doc = None
        if series.get("discharge_summary", ""):
            discharge_urn, discharge_doc = build_discharge_summary_doc_ref(
                series, patient_urn
            )

        bundle = build_transaction_bundle(
            patient,      patient_urn,
            condition,    cond_urn,
            composition,  comp_urn,
            doc_ref,      doc_urn,
            discharge_doc=discharge_doc,
            discharge_doc_urn=discharge_urn,
        )

        out_path = bundles_dir / f"bundle_{i:06d}.json"
        out_path.write_text(bundle.model_dump_json(exclude_none=True), encoding="utf-8")

        label = str(series["partition_label"])
        by_partition[label] = by_partition.get(label, 0) + 1

        if (i + 1) % 200 == 0:
            log.info("  %d/%d bundles generated...", i + 1, total)

    log.info("=== FHIR bundles generated: %d files in %s ===", total, bundles_dir)
    for label, count in sorted(by_partition.items()):
        log.info("  %s: %d bundles", label, count)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generates pre-assembled FHIR bundles from MIMIC-IV v3.1."
    )
    parser.add_argument(
        "--mimic-dir",
        default="physionet.org/files/mimiciv/3.1",
        help="MIMIC-IV v3.1 root directory (must contain hosp/ and icu/ subdirectories).",
    )
    parser.add_argument(
        "--note-dir",
        default="physionet.org/files/mimic-iv-note/2.2/note",
        help="MIMIC-IV-Note directory containing discharge.csv.gz. "
             "If present, real notes replace synthetic narratives.",
    )
    parser.add_argument(
        "--note-chars",
        type=int,
        default=2048,
        help="Number of initial note characters to use (~512 tokens). "
             "2048 chars captures Chief Complaint + HPI without reaching "
             "Assessment/Discharge Diagnosis.",
    )
    parser.add_argument(
        "--bundles-dir",
        default="etl_worker/data/bundles",
        help="Output directory for bundle_NNNNNN.json files.",
    )
    parser.add_argument(
        "--max-admissions",
        type=int,
        default=2000,
        help="Maximum number of admissions to include (split evenly across partitions).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-vitals",
        action="store_true",
        help="Skip loading ICU vital signs (ignored in real notes mode).",
    )
    parser.add_argument(
        "--n-silos",
        type=int,
        default=2,
        help="Number of federated silos for partitioning (default: 2).",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.0,
        help="Dirichlet partitioning α parameter (0.0 = legacy by specialty). "
             "Typical values: 0.1 (highly Non-IID), 0.5 (moderate), 1.0 (near-IID).",
    )
    parser.add_argument(
        "--icd-version",
        choices=["icd10", "all"],
        default="icd10",
        help="'icd10' filters admissions with ICD-10-CM (admittime >= 2015-10-01), "
             "following MIMIC-IV transition. 'all' includes all versions.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["top50", "full", "none"],
        default="full",
        help="ICD-10 code filter: 'top50' = 50 most frequent (MIMIC-IV-50); "
             "'full' = all with ≥10 occurrences (MIMIC-IV-full, Mullenbach 2018); "
             "'none' = no filter.",
    )
    parser.add_argument(
        "--split-output",
        default=None,
        help="Path to save train/val/test temporal split metadata as JSON. "
             "Example: etl_worker/data/temporal_split.json",
    )
    parser.add_argument(
        "--label-index-output",
        default=None,
        dest="label_index_output",
        help="Path to save the global label index (ICD-10 code → index) as JSON. "
             "Used by the BERT backend to ensure cross-silo consistency. "
             "Example: etl_worker/data/label_index.json",
    )
    args = parser.parse_args()

    mimic_root  = Path(args.mimic_dir)
    hosp_dir    = mimic_root / "hosp"
    icu_dir     = mimic_root / "icu"
    note_dir    = Path(args.note_dir)
    bundles_dir = Path(args.bundles_dir)

    if not hosp_dir.exists():
        log.error(
            "'hosp/' directory not found in %s.\n"
            "Check --mimic-dir and that the MIMIC-IV download includes the hosp module.",
            mimic_root,
        )
        sys.exit(1)

    using_notes = (note_dir / "discharge.csv.gz").exists()
    if using_notes:
        log.info("Mode: real notes (MIMIC-IV-Note) — synthetic narratives disabled.")
    else:
        log.info("Mode: synthetic narratives (MIMIC-IV-Note not available at %s).", note_dir)

    # ── 1. Base data — oversample when using notes to guarantee max_admissions
    #       after filtering for available notes ──
    oversample = args.max_admissions * 6 if using_notes else args.max_admissions
    base = load_base_data(
        hosp_dir, oversample, args.seed,
        n_silos=args.n_silos,
        dirichlet_alpha=args.dirichlet_alpha,
        icd_version=args.icd_version,
        benchmark=args.benchmark,
        split_output=args.split_output,
        label_index_output=args.label_index_output,
    )

    # ── 2. Real notes (primary mode) ──
    notes: dict[int, str] = {}
    full_notes: dict[int, str] = {}
    labs: dict = {}
    microbiology: dict = {}
    procedures: dict = {}
    vitals: dict = {}
    medications: dict = {}

    if using_notes:
        hadm_candidates: set[int] = set(base["hadm_id"].astype(int))
        notes = load_notes(note_dir, hadm_candidates, args.note_chars)
        # Full notes (no truncation, no leakage filter) — ROUGE reference for Exp. B
        full_notes = load_notes(note_dir, hadm_candidates, max_chars=999_999, skip_leakage_filter=True)

        # Filter base to admissions with available notes and rebalance
        base = base[base["hadm_id"].astype(int).isin(notes.keys())].copy()
        rng = random.Random(args.seed)
        n_silos = args.n_silos
        per_silo = max(1, args.max_admissions // n_silos)
        groups = []
        for pid in range(n_silos):
            subset = base[base["partition_id"] == pid]
            n = min(per_silo, len(subset))
            if n < per_silo:
                log.warning(
                    "Silo %d has only %d admissions with notes (requested %d).",
                    pid, n, per_silo,
                )
            if n > 0:
                sampled_idx = rng.sample(list(subset.index), n)
                groups.append(subset.loc[sampled_idx])
        base = pd.concat(groups).reset_index(drop=True)
        dist_str = " | ".join(
            f"silo {pid}: {(base['partition_id'] == pid).sum()}" for pid in range(n_silos)
        )
        log.info("Admissions with notes selected: %d (%s)", len(base), dist_str)

        # Load medications to enrich Experiment B prompt
        final_hadm_ids: set[int] = set(base["hadm_id"].astype(int))
        medications = load_medications(hosp_dir, final_hadm_ids)
    else:
        # ── Synthetic mode: load structured data ──
        hadm_ids: set[int] = set(base["hadm_id"].astype(int))

        labs = load_labs(hosp_dir, hadm_ids)
        microbiology = load_microbiology(hosp_dir, hadm_ids)
        procedures = load_procedures(hosp_dir, hadm_ids)

        if not args.skip_vitals and icu_dir.exists():
            vitals = load_vitals(icu_dir, hadm_ids)
        elif args.skip_vitals:
            log.info("ICU vital signs skipped (--skip-vitals).")
        else:
            log.warning("'icu/' directory not found — vital signs skipped.")

    # ── 3. Build and serialize FHIR bundles ──
    log.info("Building narratives and serializing FHIR bundles to %s...", bundles_dir)
    write_fhir_bundles(
        base, labs, microbiology, procedures, vitals, bundles_dir,
        notes=notes, medications=medications, full_notes=full_notes,
    )


if __name__ == "__main__":
    main()
