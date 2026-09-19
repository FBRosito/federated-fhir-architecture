"""
Tests for evaluation.icd_metrics (ICD-10 multi-label coding metrics used to
report FL round quality — Mullenbach et al. 2018 methodology).
"""

import numpy as np
import pytest

from evaluation.icd_metrics import (
    _precision_at_k,
    _recall_at_k,
    compute_icd_metrics,
    find_optimal_threshold,
)


class TestPrecisionRecallAtKNormalCase:
    def test_precision_at_k_perfect_match(self):
        y_true = np.array([[1, 1, 0, 0]])
        y_score = np.array([[0.9, 0.8, 0.1, 0.05]])
        assert _precision_at_k(y_true, y_score, k=2) == 1.0

    def test_recall_at_k_perfect_match(self):
        y_true = np.array([[1, 1, 0, 0]])
        y_score = np.array([[0.9, 0.8, 0.1, 0.05]])
        assert _recall_at_k(y_true, y_score, k=2) == 1.0

    def test_precision_at_k_partial_match(self):
        y_true = np.array([[1, 0, 0, 0]])
        y_score = np.array([[0.9, 0.8, 0.1, 0.05]])
        # top-2 predicted = {0, 1}; relevant = {0} -> 1/2
        assert _precision_at_k(y_true, y_score, k=2) == 0.5


class TestPrecisionAtKUndercountsWhenKExceedsLabelSpace:
    """BUG: _precision_at_k / _recall_at_k (evaluation/src/evaluation/icd_metrics.py:92-114)
    divide by the *requested* k, not by min(k, n_labels). When a benchmark has
    fewer labels than the requested k (e.g. a small partition, or any config
    where k_list=[8, 15] is used against a label space smaller than 15 —
    plausible whenever BERT_LABEL_INDEX_PATH / BERT_BENCHMARK yields a small
    local label set), `top_k = np.argsort(y_score, axis=1)[:, -k:]` silently
    returns only n_labels columns (numpy slicing does not raise), but the
    denominator is still the full `k`. This deflates precision@k below what
    it should be (dividing by a k the model could never have satisfied) even
    when every true label was correctly ranked at the top.
    """

    def test_should_not_deflate_precision_when_k_exceeds_label_count(self):
        # n_labels=5, k=8: only 5 possible predictions exist, and all 5 are
        # truly relevant AND correctly top-ranked -> precision should be 1.0.
        y_true = np.array([[1, 1, 1, 1, 1]])
        y_score = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])
        p = _precision_at_k(y_true, y_score, k=8)
        assert p == pytest.approx(1.0), (
            f"expected precision@8 == 1.0 (all 5 available labels are "
            f"relevant and correctly ranked), got {p} — the function divides "
            "by the requested k=8 instead of min(k, n_labels)=5."
        )


class TestFindOptimalThreshold:
    def test_should_return_a_threshold_from_the_grid(self):
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        t = find_optimal_threshold(y_true, y_score)
        assert t in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    def test_should_accept_custom_threshold_grid(self):
        y_true = np.array([[1, 0], [0, 1]])
        y_score = np.array([[0.6, 0.4], [0.4, 0.6]])
        t = find_optimal_threshold(y_true, y_score, thresholds=[0.5])
        assert t == 0.5


class TestComputeIcdMetrics:
    def test_should_return_nan_auc_when_insufficient_label_variance(self):
        # Every label is present in every sample -> zero variance -> AUC
        # cannot be computed, must be nan rather than raising.
        y_true = np.ones((3, 2))
        y_score = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
        metrics = compute_icd_metrics(y_true, y_score, k_list=[1])
        assert np.isnan(metrics.auc_roc_micro)
        assert np.isnan(metrics.auc_roc_macro)

    def test_should_compute_finite_auc_with_sufficient_variance(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=(50, 4))
        y_score = rng.random((50, 4))
        metrics = compute_icd_metrics(y_true, y_score, k_list=[2])
        assert np.isfinite(metrics.auc_roc_micro)

    def test_to_flat_dict_includes_at_k_metrics(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_score = np.array([[0.8, 0.1, 0.7], [0.2, 0.9, 0.3]])
        metrics = compute_icd_metrics(y_true, y_score, k_list=[1])
        flat = metrics.to_flat_dict()
        assert "P@1" in flat and "R@1" in flat and "F1@1" in flat
