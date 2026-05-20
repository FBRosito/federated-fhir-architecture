"""
llm_judge.py
------------
Avaliação LLM-as-Judge para sumários clínicos (Experimento B).

Ensemble de 3 modelos de famílias distintas — sem Llama para evitar
self-preference bias em relação ao modelo de treino (Llama-3.2-3B):

  1. Qwen/Qwen2.5-72B-Instruct   (Alibaba / Qwen)
  2. google/gemma-3-27b-it        (Google / Gemma)
  3. deepseek/deepseek-r1         (DeepSeek AI — NÃO usar variantes distiladas Llama)

Dimensões avaliadas (escala 1-5 para cada):
  1. Clinical accuracy    — diagnósticos e procedimentos corretos
  2. Completeness         — cobre os pontos principais da internação
  3. Coherence            — texto fluente e bem organizado
  4. Hallucination-free   — ausência de informações inventadas
  5. Clinical utility     — útil para o próximo médico que atender o paciente

Score final = média das 3 avaliações sobre as 5 dimensões.
Reportar: Spearman ρ entre pares de juízes (robustez do ensemble).

Variáveis de ambiente:
    OPENROUTER_API_KEY  Chave da API OpenRouter (obrigatória para avaliação LLM)
    JUDGE_MAX_WORKERS   Threads paralelas (default: 3 — um por juiz)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import httpx

log = logging.getLogger(__name__)

# ── Configuração dos juízes ───────────────────────────────────────────────────

JUDGES: list[dict[str, str]] = [
    {
        "name":     "qwen2.5-72b",
        "model_id": "qwen/qwen-2.5-72b-instruct",
        "family":   "Qwen (Alibaba)",
    },
    {
        "name":     "gemma-3-27b",
        "model_id": "google/gemma-3-27b-it",
        "family":   "Gemma (Google)",
    },
    {
        "name":     "deepseek-r1",
        "model_id": "deepseek/deepseek-r1",
        "family":   "DeepSeek AI",
    },
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_PROMPT_TEMPLATE = """\
You are an expert clinical physician evaluating an AI-generated hospital discharge summary.

## Reference note (ground truth)
{reference}

## Generated summary
{prediction}

## Evaluation instructions
Score the generated summary on **5 dimensions** using a scale of 1-5:
1 = very poor, 3 = acceptable, 5 = excellent.

Respond ONLY with a valid JSON object, no explanations outside the JSON:
{{
  "clinical_accuracy": <1-5>,
  "completeness": <1-5>,
  "coherence": <1-5>,
  "hallucination_free": <1-5>,
  "clinical_utility": <1-5>,
  "reasoning": "<one sentence explaining your scores>"
}}
"""

# ── Tipos de dados ────────────────────────────────────────────────────────────

@dataclass
class JudgeScore:
    """Pontuações de um único juiz para um exemplo."""
    judge_name:         str
    clinical_accuracy:  float
    completeness:       float
    coherence:          float
    hallucination_free: float
    clinical_utility:   float
    reasoning:          str = ""
    error:              str = ""

    @property
    def mean_score(self) -> float:
        return np.mean([
            self.clinical_accuracy, self.completeness, self.coherence,
            self.hallucination_free, self.clinical_utility,
        ])

    def to_dict(self) -> dict:
        return {
            "judge":              self.judge_name,
            "clinical_accuracy":  self.clinical_accuracy,
            "completeness":       self.completeness,
            "coherence":          self.coherence,
            "hallucination_free": self.hallucination_free,
            "clinical_utility":   self.clinical_utility,
            "mean":               round(self.mean_score, 4),
            "reasoning":          self.reasoning,
            "error":              self.error,
        }


@dataclass
class EnsembleScore:
    """Pontuações do ensemble (média dos 3 juízes) para um exemplo."""
    scores_per_judge:   list[JudgeScore] = field(default_factory=list)
    spearman_rho:       dict[str, float] = field(default_factory=dict)

    @property
    def ensemble_mean(self) -> float:
        valid = [s.mean_score for s in self.scores_per_judge if not s.error]
        return float(np.mean(valid)) if valid else float("nan")

    @property
    def per_dimension_mean(self) -> dict[str, float]:
        dims = ["clinical_accuracy", "completeness", "coherence", "hallucination_free", "clinical_utility"]
        result = {}
        for dim in dims:
            vals = [getattr(s, dim) for s in self.scores_per_judge if not s.error]
            result[dim] = float(np.mean(vals)) if vals else float("nan")
        return result

    def to_dict(self) -> dict:
        d = {
            "ensemble_mean": round(self.ensemble_mean, 4),
            "per_judge":     [s.to_dict() for s in self.scores_per_judge],
            "spearman_rho":  {k: round(v, 4) for k, v in self.spearman_rho.items()},
        }
        d.update({f"dim_{k}": round(v, 4) for k, v in self.per_dimension_mean.items()})
        return d


# ── Chamada à API ─────────────────────────────────────────────────────────────

def _call_judge(
    judge: dict[str, str],
    prediction: str,
    reference: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> JudgeScore:
    """Chama um único juiz via OpenRouter API e parseia o JSON de resposta."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        reference  = reference[:2000],   # truncar para não ultrapassar context
        prediction = prediction[:2000],
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer":  "https://github.com/federated-fhir-architecture",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": judge["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Parsear JSON da resposta
            scores_raw = json.loads(content)
            return JudgeScore(
                judge_name         = judge["name"],
                clinical_accuracy  = float(scores_raw.get("clinical_accuracy", 3)),
                completeness       = float(scores_raw.get("completeness", 3)),
                coherence          = float(scores_raw.get("coherence", 3)),
                hallucination_free = float(scores_raw.get("hallucination_free", 3)),
                clinical_utility   = float(scores_raw.get("clinical_utility", 3)),
                reasoning          = str(scores_raw.get("reasoning", "")),
            )
        except Exception as exc:
            last_error = str(exc)
            log.warning("Juiz %s tentativa %d/%d falhou: %s", judge["name"], attempt + 1, max_retries, exc)
            time.sleep(retry_delay * (attempt + 1))

    return JudgeScore(
        judge_name         = judge["name"],
        clinical_accuracy  = float("nan"),
        completeness       = float("nan"),
        coherence          = float("nan"),
        hallucination_free = float("nan"),
        clinical_utility   = float("nan"),
        error              = last_error,
    )


# ── Spearman ρ ────────────────────────────────────────────────────────────────

def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Calcula o coeficiente de correlação de Spearman entre dois vetores."""
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except ImportError:
        # Implementação manual se scipy não estiver disponível
        n = len(x)
        if n < 2:
            return float("nan")
        rank_x = np.argsort(np.argsort(x)).astype(float)
        rank_y = np.argsort(np.argsort(y)).astype(float)
        d2 = ((rank_x - rank_y) ** 2).sum()
        return float(1 - 6 * d2 / (n * (n**2 - 1)))


# ── Interface principal ───────────────────────────────────────────────────────

def evaluate_with_llm_judges(
    predictions: list[str],
    references: list[str],
    api_key: str | None = None,
    max_samples: int = 50,
    max_workers: int | None = None,
) -> list[EnsembleScore]:
    """
    Avalia sumários com o ensemble de 3 juízes LLM via OpenRouter.

    Args:
        predictions:  Sumários gerados pelo modelo.
        references:   Notas de alta reais (mesma ordem).
        api_key:      Chave OpenRouter (fallback: OPENROUTER_API_KEY env).
        max_samples:  Limita para max_samples amostras (custo API).
        max_workers:  Threads paralelas por exemplo (default: 3 — um por juiz).

    Returns:
        Lista de EnsembleScore, um por amostra avaliada.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error(
            "OPENROUTER_API_KEY não definida — avaliação LLM-as-judge ignorada. "
            "Defina a variável de ambiente para habilitar."
        )
        return []

    n = min(len(predictions), len(references), max_samples)
    if n < len(predictions):
        log.info("LLM-as-judge: limitado a %d/%d amostras.", n, len(predictions))

    workers = max_workers or int(os.getenv("JUDGE_MAX_WORKERS", "3"))
    ensemble_scores: list[EnsembleScore] = []

    log.info("Iniciando avaliação LLM-as-judge: %d amostras × %d juízes...", n, len(JUDGES))

    for i in range(n):
        pred = predictions[i]
        ref  = references[i]
        judge_scores: list[JudgeScore] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_call_judge, judge, pred, ref, api_key): judge["name"]
                for judge in JUDGES
            }
            for future in as_completed(futures):
                score = future.result()
                judge_scores.append(score)

        # Calcular Spearman ρ entre pares de juízes (sobre as 5 dimensões)
        spearman: dict[str, float] = {}
        valid_scores = [s for s in judge_scores if not s.error]
        if len(valid_scores) >= 2:
            dims = ["clinical_accuracy", "completeness", "coherence", "hallucination_free", "clinical_utility"]
            for j1 in range(len(valid_scores)):
                for j2 in range(j1 + 1, len(valid_scores)):
                    s1 = valid_scores[j1]
                    s2 = valid_scores[j2]
                    x = [getattr(s1, d) for d in dims]
                    y = [getattr(s2, d) for d in dims]
                    pair = f"{s1.judge_name}_vs_{s2.judge_name}"
                    spearman[pair] = _spearman_rho(x, y)

        ensemble = EnsembleScore(
            scores_per_judge = judge_scores,
            spearman_rho     = spearman,
        )
        ensemble_scores.append(ensemble)

        if (i + 1) % 10 == 0:
            log.info("LLM-as-judge: %d/%d amostras avaliadas.", i + 1, n)

    # Resumo geral
    all_means = [s.ensemble_mean for s in ensemble_scores if not np.isnan(s.ensemble_mean)]
    if all_means:
        log.info(
            "LLM-as-judge concluído: n=%d | ensemble_mean=%.3f ± %.3f",
            len(all_means), float(np.mean(all_means)), float(np.std(all_means)),
        )

    # Spearman ρ médio entre pares
    all_pairs: dict[str, list[float]] = {}
    for s in ensemble_scores:
        for pair, rho in s.spearman_rho.items():
            all_pairs.setdefault(pair, []).append(rho)
    for pair, rhos in all_pairs.items():
        log.info("Spearman ρ (%s): %.3f (n=%d)", pair, float(np.mean(rhos)), len(rhos))

    return ensemble_scores


def aggregate_judge_scores(scores: list[EnsembleScore]) -> dict[str, float]:
    """
    Agrega os EnsembleScores em métricas resumidas para o paper.

    Returns:
        Dicionário com médias e desvios padrão de todas as dimensões.
    """
    if not scores:
        return {}

    dims = ["clinical_accuracy", "completeness", "coherence", "hallucination_free", "clinical_utility"]
    result: dict[str, float] = {}

    ensemble_means = [s.ensemble_mean for s in scores if not np.isnan(s.ensemble_mean)]
    if ensemble_means:
        result["ensemble_mean"]  = round(float(np.mean(ensemble_means)), 4)
        result["ensemble_std"]   = round(float(np.std(ensemble_means)), 4)

    for dim in dims:
        vals = [s.per_dimension_mean.get(dim, float("nan")) for s in scores]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            result[f"{dim}_mean"] = round(float(np.mean(vals)), 4)
            result[f"{dim}_std"]  = round(float(np.std(vals)), 4)

    # Spearman ρ médio global
    all_rhos: dict[str, list[float]] = {}
    for s in scores:
        for pair, rho in s.spearman_rho.items():
            all_rhos.setdefault(pair, []).append(rho)
    for pair, rhos in all_rhos.items():
        result[f"spearman_rho_{pair}"] = round(float(np.mean(rhos)), 4)

    return result
