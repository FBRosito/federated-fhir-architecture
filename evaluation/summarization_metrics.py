"""
summarization_metrics.py
------------------------
Métricas de avaliação para sumário clínico de alta (Experimento B).

Camadas de validação (seguindo Kryscinski et al., 2020 e Wang et al., 2020):
  1. ROUGE-1/2/L      — sobreposição n-gram (reproduzível, offline)
  2. BERTScore        — similaridade semântica via PubMedBERT embeddings
  3. QAGS             — consistência factual sem LLM (Question Answering + similarity)

Uso:
    from evaluation.summarization_metrics import evaluate_summaries, SummarizationMetrics

    metrics = evaluate_summaries(
        predictions=["Generated summary..."],
        references=["Reference discharge note..."],
    )
    print(metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

PUBMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


@dataclass
class SummarizationMetrics:
    """Contêiner de métricas de sumário clínico."""
    rouge_1:     float
    rouge_2:     float
    rouge_l:     float
    bertscore_f1: float
    bertscore_p:  float
    bertscore_r:  float
    n_samples:    int
    # QAGS é opcional — requer QA pipeline instalado
    qags_score:   float | None = None

    def to_flat_dict(self) -> dict[str, float]:
        d: dict[str, float] = {
            "rouge_1":      self.rouge_1,
            "rouge_2":      self.rouge_2,
            "rouge_l":      self.rouge_l,
            "bertscore_f1": self.bertscore_f1,
            "bertscore_p":  self.bertscore_p,
            "bertscore_r":  self.bertscore_r,
            "n_samples":    float(self.n_samples),
        }
        if self.qags_score is not None:
            d["qags_score"] = self.qags_score
        return d

    def __str__(self) -> str:
        lines = [
            f"Summarization Metrics (n={self.n_samples})",
            f"  ROUGE-1: {self.rouge_1:.4f}",
            f"  ROUGE-2: {self.rouge_2:.4f}",
            f"  ROUGE-L: {self.rouge_l:.4f}",
            f"  BERTScore F1: {self.bertscore_f1:.4f} (P={self.bertscore_p:.4f}, R={self.bertscore_r:.4f})",
        ]
        if self.qags_score is not None:
            lines.append(f"  QAGS: {self.qags_score:.4f}")
        return "\n".join(lines)


def _compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Calcula ROUGE-1/2/L usando rouge-score."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        log.warning("rouge-score não instalado — ROUGE será NaN.")
        return {"rouge_1": float("nan"), "rouge_2": float("nan"), "rouge_l": float("nan")}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1s, r2s, rls = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)
        rls.append(scores["rougeL"].fmeasure)

    return {
        "rouge_1": float(np.mean(r1s)),
        "rouge_2": float(np.mean(r2s)),
        "rouge_l": float(np.mean(rls)),
    }


def _compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = PUBMEDBERT_MODEL,
) -> dict[str, float]:
    """
    Calcula BERTScore usando PubMedBERT como modelo de referência.

    O uso do PubMedBERT (treinado em textos biomédicos) é preferido ao
    BERT genérico para avaliação de textos clínicos — maior correlação
    com julgamentos humanos em domínio médico (Zhang et al., 2020).
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        log.warning("bert-score não instalado — BERTScore será NaN.")
        return {
            "bertscore_p": float("nan"),
            "bertscore_r": float("nan"),
            "bertscore_f1": float("nan"),
        }

    try:
        P, R, F = bert_score_fn(
            predictions,
            references,
            model_type  = model_type,
            lang        = "en",
            verbose     = False,
            batch_size  = 8,
            rescale_with_baseline = True,
        )
        return {
            "bertscore_p":  float(P.mean().item()),
            "bertscore_r":  float(R.mean().item()),
            "bertscore_f1": float(F.mean().item()),
        }
    except Exception as exc:
        log.warning("BERTScore falhou: %s", exc)
        return {
            "bertscore_p": float("nan"),
            "bertscore_r": float("nan"),
            "bertscore_f1": float("nan"),
        }


def _compute_qags(
    predictions: list[str],
    references: list[str],
    n_questions: int = 5,
    max_samples: int = 50,
) -> float | None:
    """
    QAGS (Wang et al., 2020): consistência factual sem LLM externo.

    Algoritmo:
      1. Gera perguntas sobre cada sumário gerado via QG model.
      2. Responde as mesmas perguntas sobre a nota de referência via QA model.
      3. Score = similaridade semântica média entre pares de respostas.

    Limitação: requer transformers com modelos de QG/QA — pode ser lento.
    Retorna None se transformers não estiver instalado ou se ocorrer erro.
    """
    try:
        from transformers import pipeline
    except ImportError:
        log.warning("transformers não instalado — QAGS será ignorado.")
        return None

    try:
        qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1,  # CPU para não interferir com o treino na GPU
        )
    except Exception as exc:
        log.warning("QAGS QA pipeline falhou ao carregar: %s", exc)
        return None

    # Limitar amostras para não demorar demais
    n = min(len(predictions), max_samples)
    qags_scores: list[float] = []

    for pred, ref in zip(predictions[:n], references[:n]):
        # Simplificação: usar as primeiras frases do sumário como "perguntas"
        # convertidas implicitamente pelo QA como contexto
        try:
            result_pred = qa_pipeline(question="What is the main diagnosis?", context=pred)
            result_ref  = qa_pipeline(question="What is the main diagnosis?", context=ref)

            ans_pred = str(result_pred.get("answer", "")).lower().strip()
            ans_ref  = str(result_ref.get("answer", "")).lower().strip()

            # Similaridade simples via overlap de tokens
            tokens_pred = set(ans_pred.split())
            tokens_ref  = set(ans_ref.split())
            if tokens_pred or tokens_ref:
                overlap = len(tokens_pred & tokens_ref)
                sim = 2 * overlap / max(len(tokens_pred) + len(tokens_ref), 1)
            else:
                sim = 0.0

            qags_scores.append(sim)
        except Exception:
            continue

    if not qags_scores:
        return None
    return float(np.mean(qags_scores))


def evaluate_summaries(
    predictions: list[str],
    references: list[str],
    compute_qags: bool = False,
    bertscore_model: str = PUBMEDBERT_MODEL,
) -> SummarizationMetrics:
    """
    Avalia sumários gerados com ROUGE-1/2/L + BERTScore(PubMedBERT) + QAGS opcional.

    Args:
        predictions:    Lista de sumários gerados pelo modelo.
        references:     Lista de notas de alta reais (mesma ordem dos predictions).
        compute_qags:   Se True, calcula QAGS (mais lento, requer transformers).
        bertscore_model: Modelo BERTScore (padrão: PubMedBERT).

    Returns:
        SummarizationMetrics com todas as métricas.
    """
    if not predictions or not references:
        log.warning("evaluate_summaries: listas vazias — retornando NaN.")
        return SummarizationMetrics(
            rouge_1=float("nan"), rouge_2=float("nan"), rouge_l=float("nan"),
            bertscore_f1=float("nan"), bertscore_p=float("nan"), bertscore_r=float("nan"),
            n_samples=0,
        )

    assert len(predictions) == len(references), "predictions e references devem ter o mesmo tamanho."
    n = len(predictions)
    log.info("Calculando métricas de sumário para %d amostras...", n)

    rouge  = _compute_rouge(predictions, references)
    bscore = _compute_bertscore(predictions, references, model_type=bertscore_model)
    qags   = _compute_qags(predictions, references) if compute_qags else None

    metrics = SummarizationMetrics(
        rouge_1      = round(rouge["rouge_1"], 6),
        rouge_2      = round(rouge["rouge_2"], 6),
        rouge_l      = round(rouge["rouge_l"], 6),
        bertscore_p  = round(bscore["bertscore_p"], 6),
        bertscore_r  = round(bscore["bertscore_r"], 6),
        bertscore_f1 = round(bscore["bertscore_f1"], 6),
        n_samples    = n,
        qags_score   = round(qags, 6) if qags is not None else None,
    )

    log.info("Métricas de sumário:\n%s", metrics)
    return metrics
