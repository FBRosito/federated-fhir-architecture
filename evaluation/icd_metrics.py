"""
icd_metrics.py
--------------
Métricas de codificação ICD-10 multi-label seguindo metodologia Mullenbach et al. (2018).

Métricas implementadas:
  - Micro-F1, Macro-F1 (com threshold 0.5)
  - AUC-ROC micro e macro (scikit-learn)
  - Precision@k: fração de top-k predições que estão no ground truth
  - Recall@k: fração do ground truth coberta pelos top-k predições
  - F1@k: harmônica de P@k e R@k

Uso:
    from evaluation.icd_metrics import ICD10Metrics, compute_icd_metrics

    metrics = compute_icd_metrics(y_true, y_score, k_list=[8, 15])
    print(metrics)

y_true:  np.ndarray [n_samples, n_labels], valores {0, 1}
y_score: np.ndarray [n_samples, n_labels], probabilidades (sigmoid outputs)
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
    """Contêiner de métricas de codificação ICD-10."""
    micro_f1:       float
    macro_f1:       float
    auc_roc_micro:  float
    auc_roc_macro:  float
    # Métricas @k: dicionários {k: valor}
    precision_at_k: dict[int, float]
    recall_at_k:    dict[int, float]
    f1_at_k:        dict[int, float]
    n_samples:      int
    n_labels:       int
    threshold:      float = 0.5
    avg_loss:       float = 0.0  # BCE loss médio do round de avaliação

    def to_flat_dict(self) -> dict[str, float]:
        """Converte para dicionário plano para logging (ex.: MLflow, CSV)."""
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
            f"ICD-10 Metrics (n={self.n_samples}, labels={self.n_labels})",
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
    """P@k médio sobre todas as amostras."""
    top_k = np.argsort(y_score, axis=1)[:, -k:]  # índices dos k maiores scores
    n = y_true.shape[0]
    total = 0.0
    for i in range(n):
        predicted = set(top_k[i])
        relevant  = set(np.where(y_true[i] == 1)[0])
        total += len(predicted & relevant) / k
    return total / max(n, 1)


def _recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """R@k médio sobre todas as amostras."""
    top_k = np.argsort(y_score, axis=1)[:, -k:]
    n = y_true.shape[0]
    total = 0.0
    for i in range(n):
        predicted = set(top_k[i])
        relevant  = set(np.where(y_true[i] == 1)[0])
        denom = max(len(relevant), 1)
        total += len(predicted & relevant) / denom
    return total / max(n, 1)


def compute_icd_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_list: list[int] | None = None,
    threshold: float = 0.5,
) -> ICD10Metrics:
    """
    Calcula o conjunto completo de métricas Mullenbach 2018 para ICD-10 multi-label.

    Args:
        y_true:    [n_samples, n_labels] — ground truth binário {0, 1}.
        y_score:   [n_samples, n_labels] — probabilidades sigmoid do modelo.
        k_list:    Lista de k para métricas @k (padrão: [8, 15]).
        threshold: Threshold para F1 binário (padrão: 0.5).

    Returns:
        ICD10Metrics com todas as métricas calculadas.
    """
    if k_list is None:
        k_list = [8, 15]

    n_samples, n_labels = y_true.shape

    # Predições binárias via threshold
    y_pred = (y_score >= threshold).astype(int)

    # Micro e macro F1
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC-ROC: requer pelo menos uma coluna com ambas as classes
    try:
        # Filtra rótulos com variância nula (todos 0 ou todos 1) para não quebrar o sklearn
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
        log.warning("AUC-ROC não calculável: %s", exc)
        auc_roc_micro = auc_roc_macro = float("nan")

    # Métricas @k
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
    dataloader,      # DataLoader com ICD10MultiLabelDataset
    device,          # torch.device
    k_list: list[int] | None = None,
) -> ICD10Metrics:
    """
    Avalia o PLMICDModel sobre um DataLoader e retorna as métricas ICD-10.

    Args:
        model:      PLMICDModel com adaptadores LoRA.
        dataloader: DataLoader de ICD10MultiLabelDataset.
        device:     Dispositivo de inferência.
        k_list:     Lista de k para métricas @k.

    Returns:
        ICD10Metrics completo.
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

    # Sigmoid para converter logits em probabilidades
    scores_np = 1.0 / (1.0 + np.exp(-logits_np))

    metrics = compute_icd_metrics(labels_np, scores_np, k_list=k_list)
    metrics.avg_loss = round(avg_loss, 6)
    log.info("Avaliação BERT: loss=%.4f\n%s", avg_loss, metrics)
    return metrics
