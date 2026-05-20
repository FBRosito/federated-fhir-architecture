"""
etl_pipeline.py
---------------
Reads Non-IID partitioned clinical evolutions from CSV and converts them to
FHIR R5 resources (Patient, Condition, Composition, DocumentReference), sending
a Transaction Bundle via POST to the HAPI FHIR server for validation.

Non-IID partitioning:
  - Partition 0 (cardiology)      → predominance of cardiovascular conditions
  - Partition 1 (pneumology)      → predominance of respiratory conditions
  - Partition 2 (endocrinology)   → predominance of metabolic/endocrine conditions
  - Partition 3 (general)         → mixed distribution (emergency/general)

Usage:
    uv run python etl_worker/etl_pipeline.py [--partition <id>] [--data <path>]

Environment variables:
    FHIR_SERVER_URL   HAPI FHIR base URL (default: http://localhost:8080/fhir)
    ETL_PARTITION_ID  Partition to process; -1 processes all (default: -1)
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import httpx
from fhir.resources.bundle import Bundle, BundleEntry, BundleEntryRequest
from fhir.resources.composition import Composition
from fhir.resources.condition import Condition
from fhir.resources.documentreference import DocumentReference
from fhir.resources.patient import Patient

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("etl_pipeline")

# ── FHIR constants ─────────────────────────────────────────────────────────────

LOINC_PROGRESS_NOTE = {"system": "http://loinc.org", "code": "11506-3", "display": "Progress note"}
LOINC_DISCHARGE_SUMMARY = {"system": "http://loinc.org", "code": "18842-5", "display": "Discharge summary"}
SNOMED_ENCOUNTER = {"system": "http://snomed.info/sct", "code": "371531000", "display": "Report of clinical encounter"}

COND_CLINICAL_ACTIVE = {
    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active", "display": "Active"}]
}
COND_VER_CONFIRMED = {
    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed", "display": "Confirmed"}]
}

# ── Diagnosis → ICD-10 mapping ─────────────────────────────────────────────────
# Keys are Portuguese diagnosis names from the CSV (Brazilian clinical records).
# Do NOT translate — they must match the raw_diagnosis column values exactly.

ICD10_MAP: dict[str, tuple[str, str]] = {
    # Cardiovascular
    "Hipertensão arterial sistêmica":      ("I10",   "Hipertensão essencial (primária)"),
    "Insuficiência cardíaca congestiva":   ("I50.0",  "Insuficiência cardíaca congestiva"),
    "Angina instável":                     ("I20.0",  "Angina instável"),
    "Doença valvar aórtica":               ("I35.9",  "Transtorno da valva aórtica, não especificado"),
    "AVC isquêmico":                       ("I63.9",  "Infarto cerebral, não especificado"),
    "Infarto agudo do miocárdio":          ("I21.9",  "Infarto agudo do miocárdio, não especificado"),
    # Respiratory
    "DPOC exacerbado":                     ("J44.1",  "DPOC com exacerbação aguda"),
    "Asma brônquica":                      ("J45.9",  "Asma, não especificada"),
    "Pneumonia":                           ("J18.9",  "Pneumonia não especificada"),
    "Derrame pleural":                     ("J90",    "Derrame pleural não classificado em outra parte"),
    "Tromboembolismo pulmonar":            ("I26.9",  "Embolia pulmonar sem cor pulmonale agudo"),
    # Metabolic / Endocrine
    "Diabetes mellitus tipo 2":            ("E11.9",  "Diabetes mellitus tipo 2 sem complicações"),
    "Hipotireoidismo":                     ("E03.9",  "Hipotireoidismo, não especificado"),
    "Síndrome metabólica":                 ("E88.81", "Síndrome metabólica"),
    "Hiperparatireoidismo primário":       ("E21.0",  "Hiperparatireoidismo primário"),
    "Síndrome dos ovários policísticos":   ("E28.2",  "Síndrome dos ovários policísticos"),
    # General / Other
    "Lombalgia aguda":                     ("M54.5",  "Dor lombar baixa"),
    "Infecção do trato urinário":          ("N39.0",  "Infecção do trato urinário, local não especificado"),
    "Artrite reumatoide":                  ("M05.9",  "Artrite reumatoide soropositiva, não especificada"),
    "Depressão":                           ("F32.9",  "Episódio depressivo, não especificado"),
    "Doença renal crônica":                ("N18.3",  "Doença renal crônica, estágio 3"),
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_path: Path, partition_id: int = -1) -> pd.DataFrame:
    """
    Loads the clinical evolutions CSV.

    Args:
        data_path:    Path to the CSV file.
        partition_id: Filter by specific partition (-1 = all).

    Returns:
        DataFrame with evolutions for the requested partition(s).
    """
    df = pd.read_csv(data_path, dtype={"partition_id": int})
    log.info("CSV loaded: %d records across %d partitions", len(df), df["partition_id"].nunique())

    if partition_id >= 0:
        df = df[df["partition_id"] == partition_id].copy()
        if df.empty:
            log.warning("No records found for partition %d.", partition_id)
        else:
            log.info("Partition %d (%s): %d records", partition_id, df["partition_label"].iloc[0], len(df))

    return df

# ── FHIR resource builders ─────────────────────────────────────────────────────

def _build_condition_note(row: pd.Series) -> str:
    """
    Builds the Condition.note text with partition metadata and multi-label info.
    Format: "partition_id=X label=Y all_codes=CODE1,CODE2,CODE3"
    """
    note = f"partition_id={row['partition_id']} label={row['partition_label']}"
    all_codes_val = str(row.get("all_icd_codes", "")).strip()
    if all_codes_val and all_codes_val != "nan":
        note += f" all_codes={all_codes_val}"
    return note

def build_patient(row: pd.Series) -> tuple[str, Patient]:
    """
    Creates a Patient resource from one CSV row.

    Returns:
        (urn_ref, Patient) where urn_ref is the urn:uuid used for internal references.
    """
    patient_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{patient_uid}"

    name_parts = row["patient_name"].split()
    family = name_parts[-1] if len(name_parts) > 1 else name_parts[0]
    given = name_parts[:-1] if len(name_parts) > 1 else []

    patient = Patient.model_validate({
        "resourceType": "Patient",
        "id": patient_uid,
        "identifier": [{
            "system": "http://hospital.example.org/patients",
            "value": row["patient_id"],
        }],
        "name": [{"family": family, "given": given, "text": row["patient_name"]}],
        "gender": row["gender"],
        "birthDate": row["birth_date"],
    })
    return urn, patient


def build_condition(row: pd.Series, patient_urn: str) -> tuple[str, Condition]:
    """
    Maps a textual diagnosis to ICD-10 and creates a Condition resource.

    Unmapped diagnoses receive code Z03.89 ("No significant diagnosis") and
    are flagged with WARNING for manual review.
    """
    cond_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{cond_uid}"
    raw = str(row["raw_diagnosis"]).strip()

    # If CSV provides 'icd_code' directly (e.g. MIMIC-IV data), use it without lookup.
    direct = str(row.get("icd_code", "")).strip()
    if direct:
        icd_code, icd_display = direct, raw
    elif raw in ICD10_MAP:
        icd_code, icd_display = ICD10_MAP[raw]
    else:
        log.warning("Unmapped diagnosis: '%s'. Using Z03.89.", raw)
        icd_code, icd_display = "Z03.89", "Sem diagnóstico relevante relevado"

    condition = Condition.model_validate({
        "resourceType": "Condition",
        "id": cond_uid,
        "clinicalStatus": COND_CLINICAL_ACTIVE,
        "verificationStatus": COND_VER_CONFIRMED,
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                "code": "encounter-diagnosis",
                "display": "Encounter Diagnosis",
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://hl7.org/fhir/sid/icd-10",
                "code": icd_code,
                "display": icd_display,
            }],
            "text": raw,
        },
        "subject": {"reference": patient_urn},
        "recordedDate": row["record_date"],
        "note": [{"text": _build_condition_note(row)}],
    })
    return urn, condition


def build_composition(
    row: pd.Series,
    patient_urn: str,
    condition_urn: str,
) -> tuple[str, Composition]:
    """
    Creates a Composition resource (clinical progress note) referencing the
    Patient and Condition already built.
    """
    comp_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{comp_uid}"
    practitioner_display = str(row.get("practitioner", "Profissional não identificado"))

    # Section text formatted as minimal XHTML (required by FHIR Narrative).
    # html.escape() is mandatory: clinical notes may contain < and > (e.g. "trop < 0.01")
    # which break HAPI FHIR's XML parser without escaping.
    xhtml_text = (
        f'<div xmlns="http://www.w3.org/1999/xhtml">'
        f"<p><b>Evolução:</b> {html.escape(str(row['clinical_text']))}</p>"
        f"<p><b>Diagnóstico:</b> {html.escape(str(row['raw_diagnosis']))}</p>"
        f"</div>"
    )

    composition = Composition.model_validate({
        "resourceType": "Composition",
        "id": comp_uid,
        "status": "final",
        "type": {"coding": [LOINC_PROGRESS_NOTE]},
        "subject": [{"reference": patient_urn}],
        "date": row["record_date"],
        "author": [{"display": practitioner_display}],
        "title": f"Evolução Clínica — {row['partition_label'].capitalize()}",
        "section": [{
            "title": "Evolução e Conduta",
            "code": {"coding": [LOINC_PROGRESS_NOTE]},
            "text": {"status": "generated", "div": xhtml_text},
            "entry": [{"reference": condition_urn}],
        }],
    })
    return urn, composition


def build_document_reference(
    row: pd.Series,
    patient_urn: str,
    composition_urn: str,
) -> tuple[str, DocumentReference]:
    """
    Creates a DocumentReference pointing to the corresponding Composition.
    The clinical text is embedded as a base64 attachment (text/plain).
    """
    doc_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{doc_uid}"

    encoded_text = base64.b64encode(row["clinical_text"].encode("utf-8")).decode("ascii")

    doc_ref = DocumentReference.model_validate({
        "resourceType": "DocumentReference",
        "id": doc_uid,
        "status": "current",
        "docStatus": "final",
        "type": {"coding": [LOINC_PROGRESS_NOTE]},
        "category": [{"coding": [SNOMED_ENCOUNTER]}],
        "subject": {"reference": patient_urn},
        "date": row["record_date"],
        "author": [{"display": str(row.get("practitioner", ""))}],
        "description": f"Evolução clínica — {row['patient_name']} — {row['record_date'][:10]}",
        "content": [{
            "attachment": {
                "contentType": "text/plain;charset=UTF-8",
                "data": encoded_text,
                "title": f"Evolução {row['record_date'][:10]}",
                "creation": row["record_date"],
            }
        }],
        # relatesTo.target requires a reference to another DocumentReference (FHIR R4 §10.3.2).
        # The link with the Composition is maintained via Composition.section.entry (build_composition).

    })
    return urn, doc_ref

def build_discharge_summary_doc_ref(
    row: pd.Series,
    patient_urn: str,
) -> tuple[str, DocumentReference]:
    """
    Creates a DocumentReference for the complete discharge note (LOINC 18842-5).

    Used as the evaluation reference (ROUGE/BERTScore) in Experiment B.
    Distinguished from the progress note DocumentReference by its type.coding LOINC.
    """
    doc_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{doc_uid}"

    summary_text = str(row.get("discharge_summary", ""))
    if not summary_text:
        return urn, None  # type: ignore[return-value]

    encoded = base64.b64encode(summary_text.encode("utf-8")).decode("ascii")

    doc_ref = DocumentReference.model_validate({
        "resourceType": "DocumentReference",
        "id": doc_uid,
        "status": "current",
        "docStatus": "final",
        "type": {"coding": [LOINC_DISCHARGE_SUMMARY]},
        "category": [{"coding": [SNOMED_ENCOUNTER]}],
        "subject": {"reference": patient_urn},
        "date": row["record_date"],
        "author": [{"display": str(row.get("practitioner", ""))}],
        "description": f"Discharge summary — {row['patient_name']} — {row['record_date'][:10]}",
        "content": [{
            "attachment": {
                "contentType": "text/plain;charset=UTF-8",
                "data": encoded,
                "title": f"Discharge summary {row['record_date'][:10]}",
                "creation": row["record_date"],
            }
        }],
    })
    return urn, doc_ref


# ── Transaction Bundle assembly ────────────────────────────────────────────────

def _entry(urn: str, resource: Any) -> dict[str, Any]:
    """Assembles one Transaction Bundle entry from a FHIR resource."""
    return {
        "fullUrl": urn,
        "resource": json.loads(resource.model_dump_json(exclude_none=True)),
        "request": {
            "method": "POST",
            "url": resource.__resource_type__,
        },
    }


def build_transaction_bundle(
    patient: Patient, patient_urn: str,
    condition: Condition, condition_urn: str,
    composition: Composition, composition_urn: str,
    doc_ref: DocumentReference, doc_ref_urn: str,
    discharge_doc: DocumentReference | None = None,
    discharge_doc_urn: str | None = None,
) -> Bundle:
    """
    Groups resources into a single Transaction Bundle.

    Base resources (Experiment A + B): Patient, Condition, Composition, DocumentReference.
    Optional resource (Experiment B): second DocumentReference with the complete discharge note
    (LOINC 18842-5), used as the ROUGE/BERTScore reference.
    """
    entries = [
        _entry(patient_urn, patient),
        _entry(condition_urn, condition),
        _entry(composition_urn, composition),
        _entry(doc_ref_urn, doc_ref),
    ]
    if discharge_doc is not None and discharge_doc_urn is not None:
        entries.append(_entry(discharge_doc_urn, discharge_doc))

    return Bundle.model_validate({
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": entries,
    })

# ── HAPI FHIR submission ───────────────────────────────────────────────────────

def _post_payload(
    payload: bytes,
    fhir_url: str,
    *,
    label: str = "bundle",
    retry_interval: float = 5.0,
    timeout_total: float = 120.0,
) -> dict[str, Any]:
    """
    Sends Transaction Bundle bytes via POST with automatic retries.

    Transient errors (server still initialising, 5xx) are retried every
    `retry_interval` seconds for up to `timeout_total` seconds before giving up.
    Permanent client errors (4xx with FHIR/JSON Content-Type) are not retried.
    """
    headers = {"Content-Type": "application/fhir+json; charset=UTF-8"}
    deadline = time.monotonic() + timeout_total
    attempt = 0

    while True:
        attempt += 1
        try:
            response = httpx.post(fhir_url, content=payload, headers=headers, timeout=30)

            content_type = response.headers.get("content-type", "")
            is_fhir_json = "json" in content_type or "fhir" in content_type

            if 200 <= response.status_code < 300:
                return response.json()

            # 4xx with JSON/FHIR → real client error (invalid bundle); do not retry
            if 400 <= response.status_code < 500 and is_fhir_json:
                response.raise_for_status()

            # 4xx with HTML = Tomcat still initialising (JPA boot incomplete);
            # 5xx = server overloaded/restarting — both are transient
            raise httpx.HTTPStatusError(
                f"Server returned {response.status_code} (Content-Type: {content_type!r})",
                request=response.request,
                response=response,
            )

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            remaining = deadline - time.monotonic()
            is_permanent_client_error = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code < 500
                and ("json" in exc.response.headers.get("content-type", "")
                     or "fhir" in exc.response.headers.get("content-type", ""))
            )
            if is_permanent_client_error or remaining <= 0:
                raise

            log.warning(
                "Attempt %d [%s] — HAPI FHIR unavailable (%s). "
                "Retrying in %.0f s (%.0f s remaining).",
                attempt, label, exc, retry_interval, remaining,
            )
            time.sleep(min(retry_interval, remaining))


def post_bundle(
    bundle: Bundle,
    fhir_url: str,
    *,
    retry_interval: float = 5.0,
    timeout_total: float = 120.0,
) -> dict[str, Any]:
    """Serialises and sends a Transaction Bundle via POST with retries."""
    payload = bundle.model_dump_json(exclude_none=True).encode("utf-8")
    return _post_payload(payload, fhir_url, retry_interval=retry_interval, timeout_total=timeout_total)


def _summarise_response(resp: dict[str, Any], patient_id: str) -> None:
    """Logs the result for each entry in the Bundle response."""
    for entry in resp.get("entry", []):
        req = entry.get("response", {})
        status = req.get("status", "?")
        location = req.get("location", "—")
        outcome = req.get("outcome", {})
        issues = outcome.get("issue", []) if outcome else []
        severity = issues[0].get("severity", "") if issues else ""
        if severity in ("error", "fatal"):
            log.error("[%s] %s → %s  %s", patient_id, status, location, issues)
        else:
            log.info("[%s] %s → %s", patient_id, status, location)

# ── Pre-built bundles mode ─────────────────────────────────────────────────────

def run_from_bundles(bundles_dir: Path, fhir_url: str) -> None:
    """
    Loads pre-built FHIR bundles (bundle_NNNNNN.json) and sends them to HAPI FHIR.

    Does not rebuild any FHIR resource — only POSTs the existing bytes.
    Allows reloading HAPI FHIR after a `make clean` without re-reading MIMIC.

    Args:
        bundles_dir: Directory containing bundle_*.json files.
        fhir_url:    HAPI FHIR server base URL.
    """
    bundle_files = sorted(bundles_dir.glob("bundle_*.json"))
    if not bundle_files:
        log.warning("No bundle_*.json files found in %s.", bundles_dir)
        return

    log.info("Loading %d pre-built bundles from %s...", len(bundle_files), bundles_dir)
    success = errors = 0

    for bf in bundle_files:
        try:
            payload = bf.read_bytes()
            resp = _post_payload(payload, fhir_url, label=bf.stem)
            _summarise_response(resp, bf.stem)
            success += 1
        except Exception as exc:
            log.error("Error sending %s: %s", bf.name, exc)
            errors += 1

    log.info(
        "Pipeline completed: %d/%d bundles sent successfully.",
        success, success + errors,
    )
    if errors:
        log.warning("%d bundle(s) failed — check the logs above.", errors)
        sys.exit(1)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_row(row: pd.Series, fhir_url: str, dry_run: bool = False) -> bool:
    """
    Processes one CSV row: builds resources, assembles the Bundle, and sends it.

    Args:
        row:      pandas DataFrame row.
        fhir_url: HAPI FHIR base URL (e.g. http://localhost:8080/fhir).
        dry_run:  If True, serialises the Bundle but does not send it.

    Returns:
        True on success, False on error.
    """
    try:
        patient_urn,   patient    = build_patient(row)
        condition_urn, condition  = build_condition(row, patient_urn)
        comp_urn,      composition = build_composition(row, patient_urn, condition_urn)
        doc_urn,       doc_ref    = build_document_reference(row, patient_urn, comp_urn)

        bundle = build_transaction_bundle(
            patient, patient_urn,
            condition, condition_urn,
            composition, comp_urn,
            doc_ref, doc_urn,
        )

        if dry_run:
            log.info("[DRY-RUN] Bundle for %s — %d entries, %d bytes",
                     row["patient_id"], len(bundle.entry),
                     len(bundle.model_dump_json(exclude_none=True)))
            return True

        resp = post_bundle(bundle, fhir_url)
        _summarise_response(resp, row["patient_id"])
        return True

    except httpx.HTTPStatusError as exc:
        log.error("HTTP %s processing %s: %s",
                  exc.response.status_code, row["patient_id"], exc.response.text[:300])
    except httpx.ConnectError:
        log.error("No connection to FHIR at %s. Check that the hapi_fhir container is running.", fhir_url)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error processing %s: %s", row["patient_id"], exc)

    return False


def run(
    data_path: Path,
    fhir_url: str,
    partition_id: int = -1,
    dry_run: bool = False,
) -> None:
    """
    Executes the full ETL pipeline for the requested partition(s).

    Args:
        data_path:    Path to the clinical evolutions CSV.
        fhir_url:     HAPI FHIR server base URL.
        partition_id: Partition to process (-1 = all).
        dry_run:      Validate without sending to the server.
    """
    df = load_data(data_path, partition_id)
    if df.empty:
        log.warning("No data to process. Exiting.")
        return

    success = errors = 0
    for _, row in df.iterrows():
        if process_row(row, fhir_url, dry_run=dry_run):
            success += 1
        else:
            errors += 1

    total = success + errors
    log.info(
        "Pipeline completed: %d/%d records sent successfully%s.",
        success, total,
        " [DRY-RUN]" if dry_run else "",
    )
    if errors:
        log.warning("%d record(s) failed — check the logs above.", errors)
        sys.exit(1)

# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETL: loads pre-built FHIR bundles or builds from CSV → HAPI FHIR",
    )
    parser.add_argument(
        "--bundles-dir",
        type=Path,
        default=Path(os.getenv("ETL_BUNDLES_PATH", "")),
        help="Directory with pre-built FHIR bundles (bundle_*.json). "
             "When set, ignores --data and --partition.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.getenv(
            "ETL_DATA_PATH",
            str(Path(__file__).parent.parent.parent / "data" / "clinical_evolutions.csv"),
        )),
        help="Path to the clinical evolutions CSV (only used without --bundles-dir).",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=int(os.getenv("ETL_PARTITION_ID", "-1")),
        help="Partition ID to process (-1 = all). Ignored with --bundles-dir.",
    )
    parser.add_argument(
        "--fhir-url",
        default=os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir"),
        help="HAPI FHIR server base URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and serialise Bundles without sending to the server (CSV mode only).",
    )
    args = parser.parse_args()

    log.info("=== ETL Pipeline started ===")

    if args.bundles_dir and args.bundles_dir.is_dir():
        log.info("Mode: pre-built bundles | Dir: %s | FHIR: %s", args.bundles_dir, args.fhir_url)
        run_from_bundles(bundles_dir=args.bundles_dir, fhir_url=args.fhir_url)
    else:
        log.info("Mode: CSV | Data: %s | FHIR: %s | Partition: %s | Dry-run: %s",
                 args.data, args.fhir_url,
                 args.partition if args.partition >= 0 else "all",
                 args.dry_run)
        run(
            data_path=args.data,
            fhir_url=args.fhir_url,
            partition_id=args.partition,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
