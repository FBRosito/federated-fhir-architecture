"""
fhir_consumer.py
----------------
Queries the HAPI FHIR server to retrieve Condition resources (ICD-10 labels)
and DocumentReference resources (clinical texts), joining them into training
examples ready for the federated fine-tuning pipeline.

Environment variables:
    FHIR_SERVER_URL   Base URL of the HAPI FHIR server (default: http://localhost:8080/fhir)

Direct usage:
    uv run python ai_client/fhir_consumer.py [--fhir-url URL] [--patient-id ID]
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ICD10_SYSTEM      = "http://hl7.org/fhir/sid/icd-10"
LOINC_SYSTEM      = "http://loinc.org"
_DEFAULT_PAGE_SIZE = 50          # _count per page in FHIR searches
_REQUEST_TIMEOUT   = 30          # seconds

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """
    (clinical_text, ICD-10_label) pair extracted from the FHIR server.

    Attributes:
        patient_ref:      Canonical Patient reference (e.g. "Patient/uuid").
        clinical_text:    Clinical note text decoded from base64.
        icd10_code:       Primary ICD-10 code (main diagnosis, for training).
        icd10_display:    Textual description of the ICD-10 code.
        condition_id:     Condition resource ID on the server.
        doc_ref_id:       DocumentReference resource ID on the server.
        partition_note:   Non-IID partition tag extracted from the note field, if present.
        all_icd10_codes:  All ICD-10 codes for the admission (multi-label evaluation).
                          Empty = use only icd10_code as ground truth in evaluation.
    """
    patient_ref:      str
    clinical_text:    str
    icd10_code:       str
    icd10_display:    str
    condition_id:     str = ""
    doc_ref_id:       str = ""
    partition_note:   str = ""
    all_icd10_codes:  list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """
        Formats the example using the instruction template used during fine-tuning
        (Alpaca-style, compatible with Llama-3).

        WARNING: The Portuguese strings below are intentional — they match the
        prompt format the model was trained on. Do NOT translate them.
        """
        return (
            "### Instrução:\n"
            "Analise a evolução clínica abaixo e identifique o código CID-10 correspondente.\n\n"
            "### Evolução Clínica:\n"
            f"{self.clinical_text.strip()}\n\n"
            "### Resposta:\n"
            f"{self.icd10_code} — {self.icd10_display}"
        )


@dataclass
class FHIRConsumerStats:
    """Statistics from the last `fetch_training_examples` run."""
    conditions_fetched:      int = 0
    doc_refs_fetched:        int = 0
    examples_paired:         int = 0
    conditions_no_icd10:     int = 0
    doc_refs_no_text:        int = 0
    patients_with_both:      int = 0
    non_clinical_filtered:   int = 0
    warnings:                list[str] = field(default_factory=list)


# ── FHIR pagination ───────────────────────────────────────────────────────────

def _next_link(bundle: dict[str, Any]) -> str | None:
    """Extracts the next-page URL from a searchset Bundle, or None."""
    for link in bundle.get("link", []):
        if link.get("relation") == "next":
            return link.get("url")
    return None


def _iter_bundle_entries(
    session: httpx.Client,
    url: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Traverses all pages of a FHIR searchset and returns all entries.

    Args:
        session: Reusable HTTP client.
        url:     Search endpoint (e.g. "{base}/Condition").
        params:  Search query-string (_count, _fields, subject, etc.).

    Returns:
        List of all `entry[].resource` dicts found.
    """
    resources: list[dict[str, Any]] = []
    next_url: str | None = url
    first_call = True

    while next_url:
        resp = session.get(
            next_url,
            params=params if first_call else None,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        bundle = resp.json()

        for entry in bundle.get("entry", []):
            resource = entry.get("resource")
            if resource:
                resources.append(resource)

        next_url = _next_link(bundle)
        first_call = False

    return resources


# ── Condition extraction ──────────────────────────────────────────────────────

def _extract_icd10(condition: dict[str, Any]) -> tuple[str, str] | None:
    """
    Extracts (code, display) ICD-10 from the coding array of a Condition.

    Returns None if no coding with the ICD-10 system is found.
    """
    code_cc = condition.get("code", {})
    for coding in code_cc.get("coding", []):
        if coding.get("system") == ICD10_SYSTEM:
            return coding.get("code", ""), coding.get("display", "")
    return None


def _parse_note_field(note_text: str) -> dict[str, str]:
    """Parses the Condition note field in 'key=value key2=value2' format."""
    result: dict[str, str] = {}
    for token in note_text.split():
        if "=" in token:
            k, _, v = token.partition("=")
            result[k.strip()] = v.strip()
    return result


def get_conditions(
    fhir_url: str,
    patient_id: str | None = None,
    session: httpx.Client | None = None,
) -> dict[str, tuple[str, str, str, str, list[str]]]:
    """
    Retrieves Condition resources from the HAPI FHIR server.

    Args:
        fhir_url:   Server base URL (e.g. "http://localhost:8080/fhir").
        patient_id: If provided, filters to only that patient's Conditions.
        session:    HTTP client to reuse (created internally if None).

    Returns:
        Dict `{patient_ref: (condition_id, icd10_code, icd10_display, note_text, all_codes)}`.
        Only conditions with a valid ICD-10 code are included.
        When multiple Conditions exist for the same patient, the first
        (in arrival order) is used as the primary diagnosis.
        all_codes is extracted from the note field (format: "all_codes=CODE1,CODE2,...").
    """
    sess = session or httpx.Client()
    params: dict[str, Any] = {"_count": _DEFAULT_PAGE_SIZE}
    if patient_id:
        params["subject"] = f"Patient/{patient_id}"

    endpoint = f"{fhir_url.rstrip('/')}/Condition"
    log.info("Fetching Conditions from %s (patient_id=%s)...", endpoint, patient_id or "all")

    raw = _iter_bundle_entries(sess, endpoint, params)
    log.info("  → %d Condition(s) returned", len(raw))

    result: dict[str, tuple[str, str, str, str, list[str]]] = {}
    for cond in raw:
        subj = cond.get("subject", {}).get("reference", "")
        if not subj:
            continue
        if subj in result:
            continue  # already have the primary diagnosis for this patient

        icd = _extract_icd10(cond)
        if icd is None:
            log.debug("Condition %s has no ICD-10 — skipped.", cond.get("id", "?"))
            continue

        note_text = ""
        for note_item in cond.get("note", []):
            note_text = note_item.get("text", "")
            if note_text:
                break

        # Extract all_codes from note: "partition_id=X label=Y all_codes=I10,I11.9,E11.9"
        note_fields = _parse_note_field(note_text)
        all_codes_str = note_fields.get("all_codes", "")
        all_codes = [c.strip() for c in all_codes_str.split(",") if c.strip()] if all_codes_str else []
        if not all_codes:
            all_codes = [icd[0]]  # fallback: primary code only

        result[subj] = (cond.get("id", ""), icd[0], icd[1], note_text, all_codes)

    return result


# ── DocumentReference extraction ─────────────────────────────────────────────

def _decode_attachment(content_list: list[dict]) -> str | None:
    """
    Decodes the first base64 text/* attachment from a DocumentReference.

    Returns:
        Decoded text or None if no attachment with data is found.
    """
    for content_item in content_list:
        attachment = content_item.get("attachment", {})
        data_b64 = attachment.get("data")
        if data_b64:
            try:
                return base64.b64decode(data_b64).decode("utf-8", errors="replace")
            except Exception as exc:
                log.warning("Failed to decode attachment: %s", exc)
    return None


def get_document_references(
    fhir_url: str,
    patient_id: str | None = None,
    session: httpx.Client | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Retrieves DocumentReference resources from the HAPI FHIR server.

    Args:
        fhir_url:   Server base URL.
        patient_id: Filter by specific patient (optional).
        session:    Reusable HTTP client.

    Returns:
        Dict `{patient_ref: (doc_ref_id, clinical_text)}`.
        Only DocumentReferences with a decodeable text/* attachment are included.
        When multiple documents exist for the same patient, the first takes precedence.
    """
    sess = session or httpx.Client()
    params: dict[str, Any] = {"_count": _DEFAULT_PAGE_SIZE}
    if patient_id:
        params["subject"] = f"Patient/{patient_id}"

    endpoint = f"{fhir_url.rstrip('/')}/DocumentReference"
    log.info("Fetching DocumentReferences from %s (patient_id=%s)...", endpoint, patient_id or "all")

    raw = _iter_bundle_entries(sess, endpoint, params)
    log.info("  → %d DocumentReference(s) returned", len(raw))

    result: dict[str, tuple[str, str]] = {}
    for doc in raw:
        subj = doc.get("subject", {}).get("reference", "")
        if not subj:
            continue
        if subj in result:
            continue

        text = _decode_attachment(doc.get("content", []))
        if text is None:
            log.debug("DocumentReference %s has no decodeable attachment — skipped.", doc.get("id", "?"))
            continue

        result[subj] = (doc.get("id", ""), text)

    return result


# ── Non-clinical text filter ──────────────────────────────────────────────────

_LAB_PATTERNS = re.compile(
    r"\b(CULTURE|ORGANISM|SENSITIVITY|MIC|RESULT[S]?:|SUSCEPTIB|RESISTANT|INTERMEDIATE)\b",
    re.IGNORECASE,
)
_CLINICAL_KEYWORDS = re.compile(
    r"\b(paciente|apresenta|refere|nega|queixa|evolu[cç][aã]o|diagn[oó]stico|tratamento"
    r"|patient|presents|complains|denies|diagnosis|treatment|history)\b",
    re.IGNORECASE,
)


def _is_clinical_text(text: str) -> bool:
    """Returns True if text looks like a clinical note; False for lab/exam results."""
    stripped = text.strip()
    if len(stripped) < 50:
        return False
    if _LAB_PATTERNS.search(stripped):
        return False
    # Text that is mostly uppercase without clinical keywords is suspicious
    upper_ratio = sum(1 for c in stripped if c.isupper()) / max(len(stripped), 1)
    if upper_ratio > 0.7 and not _CLINICAL_KEYWORDS.search(stripped):
        return False
    return True


# ── Join into training examples ───────────────────────────────────────────────

def fetch_training_examples(
    fhir_url: str,
    patient_id: str | None = None,
    min_text_length: int = 20,
) -> tuple[list[TrainingExample], FHIRConsumerStats]:
    """
    Main entry point: joins Conditions and DocumentReferences by the `subject`
    (patient reference) field and returns training examples.

    An example is only generated when **both** resources are available for the
    same patient and the clinical text meets the minimum length requirement.

    Args:
        fhir_url:        HAPI FHIR server base URL.
        patient_id:      Filter by specific patient (optional).
        min_text_length: Minimum length (chars) of clinical text for inclusion.

    Returns:
        (examples, stats) where `stats` reports counts and warnings from the run.
    """
    stats = FHIRConsumerStats()
    examples: list[TrainingExample] = []

    with httpx.Client(headers={
        "Accept": "application/fhir+json",
        "Content-Type": "application/fhir+json",
    }) as session:
        try:
            conditions  = get_conditions(fhir_url, patient_id, session)
            doc_refs    = get_document_references(fhir_url, patient_id, session)
        except httpx.ConnectError:
            msg = f"No connection to FHIR at {fhir_url}. Check that the hapi_fhir container is running."
            log.error(msg)
            stats.warnings.append(msg)
            return [], stats
        except httpx.HTTPStatusError as exc:
            msg = f"HTTP {exc.response.status_code} querying FHIR: {exc.response.text[:200]}"
            log.error(msg)
            stats.warnings.append(msg)
            return [], stats

    stats.conditions_fetched = len(conditions)
    stats.doc_refs_fetched   = len(doc_refs)

    # Coverage diagnostics
    only_conditions  = set(conditions) - set(doc_refs)
    only_doc_refs    = set(doc_refs)   - set(conditions)
    both             = set(conditions) & set(doc_refs)

    stats.patients_with_both    = len(both)
    stats.conditions_no_icd10   = 0   # already filtered in get_conditions
    stats.doc_refs_no_text      = 0   # already filtered in get_document_references

    if only_conditions:
        w = f"{len(only_conditions)} patient(s) with Condition but no DocumentReference: {sorted(only_conditions)[:5]}..."
        log.warning(w)
        stats.warnings.append(w)
    if only_doc_refs:
        w = f"{len(only_doc_refs)} patient(s) with DocumentReference but no Condition: {sorted(only_doc_refs)[:5]}..."
        log.warning(w)
        stats.warnings.append(w)

    # Build examples
    for patient_ref in sorted(both):
        cond_id, icd10_code, icd10_display, note_text, all_codes = conditions[patient_ref]
        doc_id,  clinical_text                                    = doc_refs[patient_ref]

        if len(clinical_text.strip()) < min_text_length:
            log.debug("Text too short for %s (%d chars) — skipped.", patient_ref, len(clinical_text))
            continue

        if not _is_clinical_text(clinical_text):
            log.debug("Non-clinical text discarded for %s — possible lab result.", patient_ref)
            stats.non_clinical_filtered += 1
            continue

        examples.append(TrainingExample(
            patient_ref     = patient_ref,
            clinical_text   = clinical_text,
            icd10_code      = icd10_code,
            icd10_display   = icd10_display,
            condition_id    = cond_id,
            doc_ref_id      = doc_id,
            partition_note  = note_text,
            all_icd10_codes = all_codes,
        ))

    stats.examples_paired = len(examples)
    log.info(
        "Pairing complete: %d/%d patients with both resources → %d training examples.",
        len(both), max(len(conditions), len(doc_refs), 1), len(examples),
    )
    return examples, stats


# ── Diagnostic CLI ────────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="FHIR consumer diagnostic — lists available training examples.",
    )
    parser.add_argument(
        "--fhir-url",
        default=os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir"),
        help="HAPI FHIR server base URL.",
    )
    parser.add_argument("--patient-id", default=None, help="Filter by Patient.id.")
    parser.add_argument("--show-prompts", action="store_true", help="Display formatted prompts.")
    args = parser.parse_args()

    examples, stats = fetch_training_examples(args.fhir_url, args.patient_id)

    print(f"\n{'─'*60}")
    print(f"  Conditions fetched    : {stats.conditions_fetched}")
    print(f"  DocumentRefs fetched  : {stats.doc_refs_fetched}")
    print(f"  Patients with both    : {stats.patients_with_both}")
    print(f"  Non-clinical filtered : {stats.non_clinical_filtered}")
    print(f"  Training examples     : {stats.examples_paired}")
    if stats.warnings:
        print(f"  Warnings              : {len(stats.warnings)}")
        for w in stats.warnings:
            print(f"    ⚠ {w}")
    print(f"{'─'*60}\n")

    if args.show_prompts:
        for i, ex in enumerate(examples, 1):
            print(f"[{i}] patient={ex.patient_ref}")
            print(ex.to_prompt())
            print()


if __name__ == "__main__":
    _cli()
