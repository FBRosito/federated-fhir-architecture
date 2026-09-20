"""
Tests for etl_worker.etl_pipeline FHIR resource builders.

Covers bug-hunting checklist item: "FHIR Bundle parsing: missing resources,
duplicates, empty text, malformed codes, silent except: pass."
"""

import pandas as pd

from etl_worker.etl_pipeline import (
    ICD10_MAP,
    _build_condition_note,
    _entry,
    build_condition,
    build_discharge_summary_doc_ref,
    build_document_reference,
    build_patient,
)


def _base_row(**overrides) -> pd.Series:
    data = {
        "patient_id": "P001",
        "patient_name": "Maria Silva",
        "gender": "female",
        "birth_date": "1980-01-01",
        "raw_diagnosis": "Hipertensão arterial sistêmica",
        "record_date": "2024-03-01T10:00:00-03:00",
        "partition_id": 0,
        "partition_label": "cardiorespiratory",
        "clinical_text": "Paciente estável, sem queixas.",
        "practitioner": "Dr. Souza",
    }
    data.update(overrides)
    return pd.Series(data)


class TestBuildPatient:
    def test_should_split_family_and_given_names(self):
        urn, patient = build_patient(_base_row(patient_name="Maria Silva"))
        assert urn.startswith("urn:uuid:")
        assert patient.name[0].family == "Silva"
        assert patient.name[0].given == ["Maria"]

    def test_should_use_single_word_as_family_when_no_given_name(self):
        _, patient = build_patient(_base_row(patient_name="Madonna"))
        assert patient.name[0].family == "Madonna"
        assert patient.name[0].given == []


class TestBuildCondition:
    def test_should_map_known_diagnosis_to_icd10(self):
        _, cond = build_condition(_base_row(), "urn:uuid:patient")
        expected_code, expected_display = ICD10_MAP["Hipertensão arterial sistêmica"]
        coding = cond.code.coding[0]
        assert coding.code == expected_code
        assert coding.display == expected_display

    def test_should_use_direct_icd_code_when_provided(self):
        row = _base_row(raw_diagnosis="anything", icd_code="E11.9")
        _, cond = build_condition(row, "urn:uuid:patient")
        assert cond.code.coding[0].code == "E11.9"

    def test_should_fallback_to_z0389_for_unmapped_diagnosis(self):
        row = _base_row(raw_diagnosis="Some Unmapped Diagnosis Text")
        _, cond = build_condition(row, "urn:uuid:patient")
        assert cond.code.coding[0].code == "Z03.89"


class TestBuildConditionNote:
    def test_should_include_partition_metadata(self):
        note = _build_condition_note(_base_row())
        assert "partition_id=0" in note
        assert "label=cardiorespiratory" in note

    def test_should_omit_all_codes_when_absent(self):
        note = _build_condition_note(_base_row())
        assert "all_codes=" not in note

    def test_should_include_all_codes_when_present(self):
        note = _build_condition_note(_base_row(all_icd_codes="I10,E11.9"))
        assert "all_codes=I10,E11.9" in note

    def test_should_treat_nan_all_codes_as_absent(self):
        # pandas represents a missing CSV column value as float('nan').
        note = _build_condition_note(_base_row(all_icd_codes=float("nan")))
        assert "all_codes=" not in note


class TestBuildDocumentReferenceMissingText:
    """BUG: build_document_reference (etl_worker/src/etl_worker/etl_pipeline.py:328)
    does `row["clinical_text"].encode("utf-8")` with no null/empty guard, unlike
    build_discharge_summary_doc_ref (line 374), which casts with `str(row.get(...))`
    and explicitly returns None for empty text. A row with a missing/NaN
    clinical_text value (a real possibility for CSV-sourced ETL input, and
    exactly the "empty text" case called out in the bug-hunting checklist)
    crashes the whole ETL row with an unhandled AttributeError instead of
    being skipped or logged like the discharge-summary path.
    """

    def test_should_not_crash_on_missing_clinical_text(self):
        # A CSV row with a missing/empty clinical_text column arrives here as
        # float('nan') (pandas' representation of a missing value). This must
        # not blow up the whole ETL row with an unhandled AttributeError —
        # currently it does, which is the bug this test exposes.
        row = _base_row(clinical_text=float("nan"))
        build_document_reference(row, "urn:uuid:patient", "urn:uuid:composition")

    def test_should_encode_normal_clinical_text_successfully(self):
        row = _base_row(clinical_text="Texto clínico normal.")
        urn, doc_ref = build_document_reference(
            row, "urn:uuid:patient", "urn:uuid:composition"
        )
        assert urn.startswith("urn:uuid:")
        assert doc_ref.content[0].attachment.data


class TestBuildDischargeSummaryDocRefEmptyTextHandled:
    """Contrast case: this sibling function DOES guard against empty text
    (returns None gracefully) — used here as a regression pin for the
    correct pattern build_document_reference should also follow."""

    def test_should_return_none_for_empty_discharge_summary(self):
        row = _base_row(discharge_summary="")
        _, doc_ref = build_discharge_summary_doc_ref(row, "urn:uuid:patient")
        assert doc_ref is None

    def test_should_build_doc_ref_for_nonempty_discharge_summary(self):
        row = _base_row(discharge_summary="Resumo de alta completo.")
        urn, doc_ref = build_discharge_summary_doc_ref(row, "urn:uuid:patient")
        assert urn.startswith("urn:uuid:")
        assert doc_ref is not None


class TestEntryAssembly:
    def test_should_assemble_transaction_entry_with_post_method(self):
        _, patient = build_patient(_base_row())
        entry = _entry("urn:uuid:x", patient)
        assert entry["fullUrl"] == "urn:uuid:x"
        assert entry["request"]["method"] == "POST"
        assert entry["request"]["url"] == "Patient"
        assert entry["resource"]["resourceType"] == "Patient"
