"""
icd_metrics.py
--------------
ICD-10 multi-label coding metrics following Mullenbach et al. (2018) methodology.

Metrics implemented:
  - Micro-F1, Macro-F1 (with 0.5 threshold)
  - AUC-ROC micro and macro (scikit-learn)
  - Precision@k: fraction of top-k predictions that are in the ground truth
  - Recall@k: fraction of the ground truth covered by the top-k predictions
  - F1@k: harmonic mean of P@k and R@k

Usage:
    from evaluation.icd_metrics import ICD10Metrics, compute_icd_metrics

    metrics = compute_icd_metrics(y_true, y_score, k_list=[8, 15])
    print(metrics)

y_true:  np.ndarray [n_samples, n_labels], values {0, 1}
y_score: np.ndarray [n_samples, n_labels], probabilities (sigmoid outputs)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


@dataclass
class ICD10Metrics:
    """Container for ICD-10 coding metrics."""
    micro_f1:       float
    macro_f1:       float
    auc_roc_micro:  float
    auc_roc_macro:  float
    # @k metrics: dictionaries {k: value}
    precision_at_k: dict[int, float]
    recall_at_k:    dict[int, float]
    f1_at_k:        dict[int, float]
    n_samples:      int
    n_labels:       int
    threshold:      float = 0.5
    avg_loss:       float = 0.0  # mean BCE loss from the evaluation round

    def to_flat_dict(self) -> dict[str, float]:
        """Converts to a flat dictionary for logging (e.g. MLflow, CSV)."""
        d: dict[str, float] = {
            "eval_loss":     self.avg_loss,
            "micro_f1":      self.micro_f1,
            "macro_f1":      self.macro_f1,
            "auc_roc_micro": self.auc_roc_micro,
            "auc_roc_macro": self.auc_roc_macro,
            "n_samples":     float(self.n_samples),
            "n_labels":      float(self.n_labels),
        }
        for k, v in self.precision_at_k.items():
            d[f"P@{k}"] = v
        for k, v in self.recall_at_k.items():
            d[f"R@{k}"] = v
        for k, v in self.f1_at_k.items():
            d[f"F1@{k}"] = v
        return d

    def __str__(self) -> str:
        lines = [
            f"ICD-10 Metrics (n={self.n_samples}, labels={self.n_labels}, threshold={self.threshold:.2f})",
            f"  Micro-F1:      {self.micro_f1:.4f}",
            f"  Macro-F1:      {self.macro_f1:.4f}",
            f"  AUC-ROC micro: {self.auc_roc_micro:.4f}",
            f"  AUC-ROC macro: {self.auc_roc_macro:.4f}",
        ]
        for k in sorted(self.precision_at_k):
            lines.append(
                f"  P@{k}={self.precision_at_k[k]:.4f} | "
                f"R@{k}={self.recall_at_k[k]:.4f} | "
                f"F1@{k}={self.f1_at_k[k]:.4f}"
            )
        return "\n".join(lines)


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Mean P@k across all samples."""
    top_k = np.argsort(y_score, axis=1)[:, -k:]  # indices of the k highest scores
    n = y_true.shape[0]
    total = 0.0
    for i in range(n):
        predicted = set(top_k[i])
        relevant  = set(np.where(y_true[i] == 1)[0])
        total += len(predicted & relevant) / k
    return total / max(n, 1)


def _recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Mean R@k across all samples."""
    top_k = np.argsort(y_score, axis=1)[:, -k:]
    n = y_true.shape[0]
    total = 0.0
    for i in range(n):
        predicted = set(top_k[i])
        relevant  = set(np.where(y_true[i] == 1)[0])
        denom = max(len(relevant), 1)
        total += len(predicted & relevant) / denom
    return total / max(n, 1)


_THRESHOLD_GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float] | None = None,
) -> float:
    """Grid-search the threshold that maximises micro-F1 on (y_true, y_score)."""
    candidates = thresholds if thresholds is not None else _THRESHOLD_GRID
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = float(f1_score(y_true, (y_score >= t).astype(int), average="micro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    log.debug("Optimal threshold: %.2f  (micro-F1=%.4f)", best_t, best_f1)
    return best_t


def compute_icd_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_list: list[int] | None = None,
    threshold: float | None = None,
) -> ICD10Metrics:
    """
    Computes the full Mullenbach 2018 metric set for ICD-10 multi-label coding.

    Args:
        y_true:    [n_samples, n_labels] — binary ground truth {0, 1}.
        y_score:   [n_samples, n_labels] — sigmoid probabilities from the model.
        k_list:    List of k values for @k metrics (default: [8, 15]).
        threshold: Threshold for binary F1. If None (default), the threshold is
                   selected via grid search to maximise micro-F1 on this set.

    Returns:
        ICD10Metrics with all metrics computed.
    """
    if k_list is None:
        k_list = [8, 15]

    n_samples, n_labels = y_true.shape

    # Adaptive threshold: find the value that maximises micro-F1 on this set.
    # With many labels (e.g. 7756), sigmoid outputs are typically well below 0.5
    # even for correct predictions, so a fixed 0.5 threshold produces F1=0.
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    # Binary predictions via threshold
    y_pred = (y_score >= threshold).astype(int)

    # Micro and macro F1
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC-ROC: requires at least one column with both classes
    try:
        # Filter out labels with zero variance (all 0 or all 1) to avoid sklearn errors
        valid_cols = np.where(y_true.sum(axis=0) > 0)[0]
        if len(valid_cols) < 2:
            auc_roc_micro = auc_roc_macro = float("nan")
        else:
            auc_roc_micro = float(roc_auc_score(
                y_true[:, valid_cols], y_score[:, valid_cols], average="micro"
            ))
            auc_roc_macro = float(roc_auc_score(
                y_true[:, valid_cols], y_score[:, valid_cols], average="macro"
            ))
    except Exception as exc:
        log.warning("AUC-ROC not computable: %s", exc)
        auc_roc_micro = auc_roc_macro = float("nan")

    # @k metrics
    precision_at_k: dict[int, float] = {}
    recall_at_k:    dict[int, float] = {}
    f1_at_k:        dict[int, float] = {}

    for k in k_list:
        p = _precision_at_k(y_true, y_score, k)
        r = _recall_at_k(y_true, y_score, k)
        f = 2 * p * r / max(p + r, 1e-9)
        precision_at_k[k] = round(p, 6)
        recall_at_k[k]    = round(r, 6)
        f1_at_k[k]        = round(f, 6)

    return ICD10Metrics(
        micro_f1       = round(micro_f1, 6),
        macro_f1       = round(macro_f1, 6),
        auc_roc_micro  = round(auc_roc_micro, 6) if not np.isnan(auc_roc_micro) else float("nan"),
        auc_roc_macro  = round(auc_roc_macro, 6) if not np.isnan(auc_roc_macro) else float("nan"),
        precision_at_k = precision_at_k,
        recall_at_k    = recall_at_k,
        f1_at_k        = f1_at_k,
        n_samples      = n_samples,
        n_labels       = n_labels,
        threshold      = threshold,
    )


def evaluate_bert_model(
    model,           # PLMICDModel
    dataloader,      # DataLoader with ICD10MultiLabelDataset
    device,          # torch.device
    k_list: list[int] | None = None,
) -> ICD10Metrics:
    """
    Evaluates PLMICDModel on a DataLoader and returns ICD-10 metrics.

    Args:
        model:      PLMICDModel with LoRA adapters.
        dataloader: DataLoader of ICD10MultiLabelDataset.
        device:     Inference device.
        k_list:     List of k values for @k metrics.

    Returns:
        Complete ICD10Metrics.
    """
    import torch

    model.eval()
    all_logits: list = []
    all_labels: list = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            all_logits.append(out["logits"].float().cpu())
            all_labels.append(labels.float().cpu())
            if "loss" in out and out["loss"] is not None:
                total_loss += out["loss"].item()

    avg_loss = total_loss / max(len(dataloader), 1)
    logits_np = torch.cat(all_logits, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()

    # Sigmoid to convert logits to probabilities
    scores_np = 1.0 / (1.0 + np.exp(-logits_np))

    metrics = compute_icd_metrics(labels_np, scores_np, k_list=k_list)
    metrics.avg_loss = round(avg_loss, 6)
    log.info("BERT evaluation: loss=%.4f\n%s", avg_loss, metrics)
    return metrics
