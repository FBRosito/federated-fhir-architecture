"""
fhir_consumer_summarization.py
-------------------------------
FHIR consumer for Experiment B: discharge summary (Llama-3.2-3B).

Fetches from FHIR R4:
  - Patient: demographics (gender, birthDate)
  - Condition: admission ICD-10 codes
  - DocumentReference (LOINC 11506-3): structured clinical text (model input)
  - DocumentReference (LOINC 18842-5): complete discharge note (ROUGE/BERTScore reference)

Builds a SummarizationExample with:
  - prompt: structured text for the model to generate the summary
  - reference_summary: real discharge note (post-training evaluation)

The prompt follows instruction→response (Alpaca) format compatible with
train_one_round() in model_setup.py via the to_prompt() method.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

_DEFAULT_PAGE_SIZE = 50
_REQUEST_TIMEOUT = 30

# LOINC codes used to distinguish the two DocumentReferences in bundles
_LOINC_PROGRESS_NOTE = "11506-3"
_LOINC_DISCHARGE_SUMMARY = "18842-5"


@dataclass
class SummarizationExample:
    """
    (structured_prompt, reference_summary) pair for clinical summary fine-tuning.

    Attributes:
        patient_ref:        Reference to the Patient resource in FHIR R4.
        structured_context: Structured clinical text (demographics + ICD + labs + meds).
        reference_summary:  Real discharge note — reference for ROUGE/BERTScore.
        icd10_codes:        List of all admission ICD-10 codes.
        partition_note:     Non-IID partition tag (for silo filtering).
    """

    patient_ref: str
    structured_context: str
    reference_summary: str
    icd10_codes: list[str] = field(default_factory=list)
    partition_note: str = ""

    # Alpaca instruction→response format — compatible with ClinicalICD10Dataset
    _INSTRUCTION = (
        "You are a clinical physician assistant. "
        "Based on the structured clinical information below, "
        "write a concise and accurate discharge summary for this patient."
    )
    # WARNING: The Portuguese strings below are intentional — they match the
    # prompt format the model was trained on. Do NOT translate them.
    _RESPONSE_SEP = "### Resposta:\n"

    def to_prompt(self) -> str:
        """Assembles the instruction→response prompt for fine-tuning."""
        return (
            f"### Instrução:\n{self._INSTRUCTION}\n\n"
            f"### Contexto clínico:\n{self.structured_context}\n\n"
            f"{self._RESPONSE_SEP}{self.reference_summary}"
        )

    def to_inference_prompt(self) -> str:
        """Inference prompt (without the response) for summary generation."""
        return (
            f"### Instrução:\n{self._INSTRUCTION}\n\n"
            f"### Contexto clínico:\n{self.structured_context}\n\n"
            f"{self._RESPONSE_SEP}"
        )


def _decode_attachment(content_list: list[dict]) -> str:
    """Decodes the first base64 attachment from a DocumentReference.content."""
    for content in content_list:
        attachment = content.get("attachment", {})
        data = attachment.get("data", "")
        if data:
            try:
                return base64.b64decode(data).decode("utf-8", errors="replace")
            except Exception:
                pass
        url = attachment.get("url", "")
        if url:
            return f"[external: {url}]"
    return ""


def _get_loinc_code(doc_ref: dict) -> str:
    """Extracts the first LOINC code from the type.coding of a DocumentReference."""
    for coding in doc_ref.get("type", {}).get("coding", []):
        if "loinc" in coding.get("system", "").lower():
            return coding.get("code", "")
    return ""


def _paginate(client: httpx.Client, url: str) -> list[dict]:
    """Iterates through FHIR result pages and returns all resources."""
    resources: list[dict] = []
    next_url: str | None = url

    while next_url:
        resp = client.get(next_url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        bundle = resp.json()
        for entry in bundle.get("entry", []):
            r = entry.get("resource", {})
            if r:
                resources.append(r)
        next_url = None
        for link in bundle.get("link", []):
            if link.get("relation") == "next":
                next_url = link.get("url")
                break

    return resources


def fetch_summarization_examples(
    fhir_url: str,
    patient_id: str | None = None,
    max_examples: int = 0,
) -> tuple[list[SummarizationExample], Any]:
    """
    Fetches discharge summary examples from FHIR R4.

    For each patient with a discharge note DocumentReference (LOINC 18842-5),
    builds a SummarizationExample with:
      - structured_context: text from the progress note (LOINC 11506-3)
      - reference_summary:  text from the discharge note (LOINC 18842-5)

    Args:
        fhir_url:      Base URL of the FHIR R4 server.
        patient_id:    Filter by specific patient (None = all).
        max_examples:  Cap on returned examples (0 = no limit). Useful for
                       smoke/dev runs; ROUGE/BERTScore should use 0 (full test set).

    Returns:
        (examples, stats)
    """
    from dataclasses import dataclass as _dc

    @_dc
    class Stats:
        total_patients: int = 0
        total_examples: int = 0
        missing_summary: int = 0
        warnings: list[str] = field(default_factory=list)

    stats = Stats()
    examples: list[SummarizationExample] = []

    base = fhir_url.rstrip("/")
    with httpx.Client(base_url=base) as client:

        # 1. Fetch discharge summary DocumentReferences (LOINC 18842-5)
        summary_url = (
            f"{base}/DocumentReference"
            f"?type=http://loinc.org|{_LOINC_DISCHARGE_SUMMARY}"
            f"&_count={_DEFAULT_PAGE_SIZE}"
        )
        if patient_id:
            summary_url += f"&subject={patient_id}"

        summaries = _paginate(client, summary_url)
        log.info("Discharge summary DocumentReferences found: %d", len(summaries))

        if not summaries:
            stats.warnings.append(
                "No DocumentReference with LOINC 18842-5 found. "
                "Check that bundles were generated with --note-dir."
            )
            return [], stats

        # Index summaries by patient reference
        summary_by_patient: dict[str, tuple[str, str]] = (
            {}
        )  # patient_ref → (text, condition_note)
        for doc in summaries:
            patient_ref = doc.get("subject", {}).get("reference", "")
            text = _decode_attachment(doc.get("content", []))
            if patient_ref and text:
                summary_by_patient[patient_ref] = (text, "")

        # 2. Fetch clinical progress note DocumentReferences (LOINC 11506-3)
        progress_url = (
            f"{base}/DocumentReference"
            f"?type=http://loinc.org|{_LOINC_PROGRESS_NOTE}"
            f"&_count={_DEFAULT_PAGE_SIZE}"
        )
        if patient_id:
            progress_url += f"&subject={patient_id}"

        progress_docs = _paginate(client, progress_url)
        log.info(
            "Clinical progress note DocumentReferences found: %d", len(progress_docs)
        )

        progress_by_patient: dict[str, str] = {}
        for doc in progress_docs:
            patient_ref = doc.get("subject", {}).get("reference", "")
            text = _decode_attachment(doc.get("content", []))
            if patient_ref and text:
                progress_by_patient[patient_ref] = text

        # 3. Fetch Conditions with all_codes (partition note)
        cond_url = f"{base}/Condition?_count={_DEFAULT_PAGE_SIZE}"
        if patient_id:
            cond_url += f"&subject={patient_id}"

        conditions = _paginate(client, cond_url)
        conditions_by_patient: dict[str, tuple[list[str], str]] = {}
        for cond in conditions:
            patient_ref = cond.get("subject", {}).get("reference", "")
            icd_codes: list[str] = []
            for coding in cond.get("code", {}).get("coding", []):
                code = coding.get("code", "")
                if code:
                    icd_codes.append(code)
            note_text = ""
            for note in cond.get("note", []):
                note_text += note.get("text", "")
            # Parse all_codes from note
            all_codes: list[str] = []
            if "all_codes=" in note_text:
                part = note_text.split("all_codes=")[-1].split()[0]
                all_codes = [c for c in part.split(",") if c]
            else:
                all_codes = icd_codes
            if patient_ref:
                conditions_by_patient[patient_ref] = (all_codes, note_text)

        # 4. Assemble SummarizationExamples
        for patient_ref, (reference_summary, _) in summary_by_patient.items():
            structured_context = progress_by_patient.get(patient_ref, "")
            if not structured_context:
                stats.missing_summary += 1
                continue

            icd_codes, partition_note = conditions_by_patient.get(patient_ref, ([], ""))

            examples.append(
                SummarizationExample(
                    patient_ref=patient_ref,
                    structured_context=structured_context,
                    reference_summary=reference_summary,
                    icd10_codes=icd_codes,
                    partition_note=partition_note,
                )
            )

    if max_examples > 0 and len(examples) > max_examples:
        examples = examples[:max_examples]

    stats.total_examples = len(examples)
    log.info(
        "SummarizationExamples assembled: %d | missing_summary=%d",
        stats.total_examples,
        stats.missing_summary,
    )
    return examples, stats
