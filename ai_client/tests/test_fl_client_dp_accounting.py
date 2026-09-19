"""
Tests for the pure/DP-accounting helper functions in ai_client.fl_client:
  - _stratified_split      (train/eval split — no data leakage across sets)
  - _compute_cumulative_epsilon (RDP composition across FL rounds)
  - verify_effective_learning_rate (Fase 0 LR-mismatch guard)
  - _extract_icd_codes     (ICD-10 regex extraction used for eval metrics)

Covers critical invariants:
  1. RDP accountant: accumulated epsilon must never decrease with more steps;
     delta is respected.
  5. Fixed seed -> reproducible train/eval split.
"""

import math
import os

import pytest

from ai_client.fhir_consumer import TrainingExample
from ai_client.fl_client import (
    ICD10_PATTERN,
    _compute_cumulative_epsilon,
    _extract_icd_codes,
    _stratified_split,
    verify_effective_learning_rate,
)


def _example(code: str, text: str = "note") -> TrainingExample:
    return TrainingExample(
        patient_ref=f"Patient/{code}-{text}",
        clinical_text=text,
        icd10_code=code,
        icd10_display=f"{code} display",
    )


class TestStratifiedSplitNoLeakage:
    def test_should_not_duplicate_examples_across_train_and_eval(self):
        examples = [_example("A10", f"note-{i}") for i in range(20)] + [
            _example("B20", f"note-{i}") for i in range(20)
        ]
        train, eval_ = _stratified_split(examples, train_ratio=0.8, seed=42)

        train_refs = {e.patient_ref for e in train}
        eval_refs = {e.patient_ref for e in eval_}
        assert train_refs.isdisjoint(eval_refs), (
            "train and eval sets must be disjoint — any overlap is a data "
            "leakage bug for FL evaluation."
        )
        assert len(train) + len(eval_) == len(examples)

    def test_should_preserve_per_code_proportion_roughly(self):
        examples = [_example("A10", f"n{i}") for i in range(100)]
        train, eval_ = _stratified_split(examples, train_ratio=0.8, seed=42)
        assert len(train) == 80
        assert len(eval_) == 20

    def test_should_be_deterministic_given_fixed_seed(self):
        examples = [_example(f"C{i % 5}", f"n{i}") for i in range(50)]
        train1, eval1 = _stratified_split(examples, seed=42)
        train2, eval2 = _stratified_split(examples, seed=42)
        assert [e.patient_ref for e in train1] == [e.patient_ref for e in train2]
        assert [e.patient_ref for e in eval1] == [e.patient_ref for e in eval2]

    def test_should_fallback_to_global_split_when_all_codes_singleton(self):
        # 20 unique ICD-10 codes with 1 example each — pure per-group
        # stratification would put everything in train and leave eval empty,
        # which crashes aggregate_evaluate on the server (divide-by-zero).
        examples = [_example(f"UNIQUE{i}") for i in range(20)]
        train, eval_ = _stratified_split(examples, train_ratio=0.8, seed=42)
        assert len(eval_) > 0, (
            "fallback split must guarantee at least one eval example when "
            "every ICD-10 code group has exactly one example."
        )

    def test_should_handle_empty_input(self):
        train, eval_ = _stratified_split([], seed=42)
        assert train == []
        assert eval_ == []


class TestCumulativeEpsilon:
    def test_should_return_inf_when_noise_multiplier_zero(self):
        assert _compute_cumulative_epsilon(0.0, 0.1, 100, 1e-5) == float("inf")

    def test_should_return_inf_when_no_steps_taken(self):
        assert _compute_cumulative_epsilon(1.0, 0.1, 0, 1e-5) == float("inf")

    def test_should_never_decrease_as_steps_accumulate(self):
        eps_10 = _compute_cumulative_epsilon(1.0, 0.05, 10, 1e-5)
        eps_50 = _compute_cumulative_epsilon(1.0, 0.05, 50, 1e-5)
        eps_100 = _compute_cumulative_epsilon(1.0, 0.05, 100, 1e-5)
        assert eps_10 <= eps_50 <= eps_100, (
            "critical invariant #1: accumulated epsilon must never decrease "
            "as more DP-SGD steps are composed"
        )

    def test_should_respect_delta_parameter(self):
        # A looser (larger) delta must yield epsilon <= that of a stricter
        # (smaller) delta for the same (sigma, q, steps).
        eps_loose = _compute_cumulative_epsilon(1.0, 0.05, 50, 1e-3)
        eps_strict = _compute_cumulative_epsilon(1.0, 0.05, 50, 1e-5)
        assert eps_loose <= eps_strict

    def test_should_decrease_epsilon_with_higher_noise_multiplier(self):
        eps_low_sigma = _compute_cumulative_epsilon(0.5, 0.05, 50, 1e-5)
        eps_high_sigma = _compute_cumulative_epsilon(2.0, 0.05, 50, 1e-5)
        assert eps_high_sigma < eps_low_sigma


class TestVerifyEffectiveLearningRate:
    def test_should_noop_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("FL_LEARNING_RATE", raising=False)
        # Must not raise even with a wildly different learning_rate.
        verify_effective_learning_rate(1, 9.99)

    def test_should_raise_on_mismatch_within_first_two_rounds(self, monkeypatch):
        monkeypatch.setenv("FL_LEARNING_RATE", "2e-4")
        with pytest.raises(RuntimeError):
            verify_effective_learning_rate(1, 5e-5)

    def test_should_pass_when_learning_rate_matches(self, monkeypatch):
        monkeypatch.setenv("FL_LEARNING_RATE", "2e-4")
        verify_effective_learning_rate(1, 2e-4)
        verify_effective_learning_rate(2, 2e-4)

    def test_should_not_check_rounds_after_two(self, monkeypatch):
        # Server LR decays to 40% starting round 3 — the check must not fire
        # a false positive there.
        monkeypatch.setenv("FL_LEARNING_RATE", "2e-4")
        verify_effective_learning_rate(3, 2e-4 * 0.4)


class TestExtractIcdCodes:
    def test_should_extract_simple_code(self):
        assert _extract_icd_codes("Diagnosis: E11.9 type 2 diabetes") == {"E11.9"}

    def test_should_extract_multiple_codes_with_two_digit_subcodes(self):
        # Only 1-2 digit sub-codes survive intact with the current pattern.
        codes = _extract_icd_codes("Codes: I10, E78.00 and E11.9")
        assert codes == {"I10", "E78.00", "E11.9"}

    def test_should_return_empty_set_when_no_code_present(self):
        assert _extract_icd_codes("no codes here at all") == set()

    def test_should_uppercase_lowercase_codes(self):
        assert _extract_icd_codes("dx: e11.9") == {"E11.9"}


class TestExtractIcdCodesTruncatesLongerSubcodes:
    """BUG: ICD10_PATTERN (ai_client/src/ai_client/fl_client.py:206) is
    ``\\b([A-Z]\\d{2}[A-Z0-9]{0,4}(?:\\.\\d{1,2})?)\\b`` — the decimal group
    only accepts 1-2 digits (``\\d{1,2}``) with no trailing letter. Real
    ICD-10-CM codes routinely have 3-digit or alphanumeric (7th-character
    extension) sub-codes, e.g.:
      - J45.909  (Unspecified asthma, uncomplicated) — 3 digits
      - S06.0X0A (Concussion w/o loss of consciousness, initial encounter)

    Because the regex's trailing ``\\b`` matches right after 2 digits, these
    codes are silently truncated to "J45" / "S06" instead of being extracted
    (or rejected) whole. This directly corrupts the ICD-10 extraction
    accuracy metrics computed in ai_client.fl_client._evaluate_local
    (hits_at_1, recall@k, precision@k, F1@k): a model that correctly
    generates "J45.909" is scored as a MISMATCH against ground truth
    "J45.909" because the extractor reduces both sides inconsistently
    depending on where in the text the match starts, and reduces the
    predicted code's specificity, systematically distorting reported
    accuracy for common S/T-chapter (injury) and respiratory codes.
    """

    def test_should_extract_full_three_digit_subcode_not_truncate_it(self):
        codes = _extract_icd_codes("Diagnosis: J45.909 unspecified asthma")
        assert "J45.909" in codes, (
            f"expected the full code 'J45.909' in the extracted set, got "
            f"{codes!r} — ICD10_PATTERN's `\\.\\d{{1,2}}` truncates any "
            "3-digit ICD-10-CM sub-code to its 2-digit prefix."
        )

    def test_should_not_split_one_code_into_a_truncated_duplicate(self):
        # "J45.909" must not be reported as bare "J45" once its 3-digit
        # sub-code is dropped — that silently invents a different, less
        # specific code that was never in the source text as such.
        codes = _extract_icd_codes("J45.909")
        assert "J45" not in codes, (
            f"got {codes!r} — the regex fabricated a truncated 'J45' code "
            "that collides with genuine 2-digit ICD-10 codes like J45 "
            "(unspecified asthma with no sub-code), corrupting exact-match "
            "accuracy metrics."
        )

    def test_should_extract_seventh_character_extension_codes(self):
        # Injury / trauma codes (S/T chapters) commonly carry an alphanumeric
        # 7th-character extension (initial/subsequent encounter, sequela).
        codes = _extract_icd_codes("S06.0X0A closed head injury, initial encounter")
        assert "S06.0X0A" in codes, (
            f"got {codes!r} — 7th-character extension codes are dropped "
            "entirely (truncated to the 3-char category 'S06'), which will "
            "never match any real ground-truth label in that chapter."
        )
