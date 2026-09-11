"""
llm_judge.py
------------
LLM-as-Judge evaluation for clinical summaries (Experiment B).

Ensemble of 3 models from distinct families — no Llama to avoid
self-preference bias relative to the training model (Llama-3.2-3B):

  1. Qwen/Qwen2.5-72B-Instruct   (Alibaba / Qwen)
  2. google/gemma-3-27b-it        (Google / Gemma)
  3. deepseek/deepseek-chat-v3.1  (DeepSeek AI — V3.1 chosen over R1: reliable
                                   JSON output and ~3× lower cost; R1's chain-of-
                                   thought reasoning tokens inflate cost without
                                   improving structured evaluation quality)

Evaluation dimensions (scale 1-5 Likert for each):
  1. Clinical accuracy    — correct diagnoses and procedures
  2. Completeness         — covers the main points of the admission
  3. Coherence            — fluent and well-organised text
  4. Hallucination-free   — absence of fabricated information
  5. Clinical utility     — useful for the next clinician seeing the patient

Final score = average of the 3 evaluations across 5 dimensions.
Inter-judge agreement: Krippendorff's α (ordinal, for 1-5 Likert scale) and
Spearman ρ between judge pairs (rank correlation).

Environment variables:
    OPENROUTER_API_KEY  OpenRouter API key (required; get one at openrouter.ai/keys)
    JUDGE_MAX_WORKERS   Parallel threads (default: 3 — one per judge)
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx
import numpy as np

log = logging.getLogger(__name__)

# ── Judge configuration ────────────────────────────────────────────────────────

JUDGES: list[dict[str, str]] = [
    {
        "name": "qwen2.5-72b",
        "model_id": "qwen/qwen-2.5-72b-instruct",
        "family": "Qwen (Alibaba)",
    },
    {
        "name": "gemma-3-27b",
        "model_id": "google/gemma-3-27b-it",
        "family": "Gemma (Google)",
    },
    {
        "name": "deepseek-v3.1",
        "model_id": "deepseek/deepseek-chat-v3.1",
        "family": "DeepSeek AI",
    },
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_PROMPT_TEMPLATE = """\
You are an expert clinical physician evaluating a generated hospital discharge summary.

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

# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class JudgeScore:
    """Scores from a single judge for one example."""

    judge_name: str
    clinical_accuracy: float
    completeness: float
    coherence: float
    hallucination_free: float
    clinical_utility: float
    reasoning: str = ""
    error: str = ""

    @property
    def mean_score(self) -> float:
        """Mean of the five evaluation dimensions."""
        return np.mean(
            [
                self.clinical_accuracy,
                self.completeness,
                self.coherence,
                self.hallucination_free,
                self.clinical_utility,
            ]
        )

    def to_dict(self) -> dict:
        """Serialize the judge's scores to a plain dict (JSON-ready)."""
        return {
            "judge": self.judge_name,
            "clinical_accuracy": self.clinical_accuracy,
            "completeness": self.completeness,
            "coherence": self.coherence,
            "hallucination_free": self.hallucination_free,
            "clinical_utility": self.clinical_utility,
            "mean": round(self.mean_score, 4),
            "reasoning": self.reasoning,
            "error": self.error,
        }


@dataclass
class EnsembleScore:
    """Ensemble scores (average of 3 judges) for one example."""

    scores_per_judge: list[JudgeScore] = field(default_factory=list)
    spearman_rho: dict[str, float] = field(default_factory=dict)
    krippendorff_alpha: dict[str, float] = field(default_factory=dict)

    @property
    def ensemble_mean(self) -> float:
        """Mean of per-judge mean scores, excluding failed judges (NaN if none)."""
        valid = [s.mean_score for s in self.scores_per_judge if not s.error]
        return float(np.mean(valid)) if valid else float("nan")

    @property
    def per_dimension_mean(self) -> dict[str, float]:
        """Per-dimension mean across judges, excluding failed judges."""
        dims = [
            "clinical_accuracy",
            "completeness",
            "coherence",
            "hallucination_free",
            "clinical_utility",
        ]
        result = {}
        for dim in dims:
            vals = [getattr(s, dim) for s in self.scores_per_judge if not s.error]
            result[dim] = float(np.mean(vals)) if vals else float("nan")
        return result

    def to_dict(self) -> dict:
        """Serialize the ensemble scores to a plain dict (JSON-ready)."""
        d = {
            "ensemble_mean": round(self.ensemble_mean, 4),
            "per_judge": [s.to_dict() for s in self.scores_per_judge],
            "spearman_rho": {k: round(v, 4) for k, v in self.spearman_rho.items()},
            "krippendorff_alpha": {
                k: round(v, 4) for k, v in self.krippendorff_alpha.items()
            },
        }
        d.update({f"dim_{k}": round(v, 4) for k, v in self.per_dimension_mean.items()})
        return d


# ── API call ───────────────────────────────────────────────────────────────────


def _call_judge(
    judge: dict[str, str],
    prediction: str,
    reference: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> JudgeScore:
    """Calls a single judge via the OpenRouter API and parses the JSON response."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        reference=reference[:2000],  # truncate to stay within context
        prediction=prediction[:2000],
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/federated-fhir-architecture",
        "Content-Type": "application/json",
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

            scores_raw = json.loads(content)
            return JudgeScore(
                judge_name=judge["name"],
                clinical_accuracy=float(scores_raw.get("clinical_accuracy", 3)),
                completeness=float(scores_raw.get("completeness", 3)),
                coherence=float(scores_raw.get("coherence", 3)),
                hallucination_free=float(scores_raw.get("hallucination_free", 3)),
                clinical_utility=float(scores_raw.get("clinical_utility", 3)),
                reasoning=str(scores_raw.get("reasoning", "")),
            )
        except Exception as exc:
            last_error = str(exc)
            log.warning(
                "Judge %s attempt %d/%d failed: %s",
                judge["name"],
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(retry_delay * (attempt + 1))

    return JudgeScore(
        judge_name=judge["name"],
        clinical_accuracy=float("nan"),
        completeness=float("nan"),
        coherence=float("nan"),
        hallucination_free=float("nan"),
        clinical_utility=float("nan"),
        error=last_error,
    )


# ── Spearman ρ ─────────────────────────────────────────────────────────────────


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Computes the Spearman rank correlation coefficient between two vectors."""
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(x, y)
        return float(rho)
    except ImportError:
        # Manual implementation if scipy is not available
        n = len(x)
        if n < 2:
            return float("nan")
        rank_x = np.argsort(np.argsort(x)).astype(float)
        rank_y = np.argsort(np.argsort(y)).astype(float)
        d2 = ((rank_x - rank_y) ** 2).sum()
        return float(1 - 6 * d2 / (n * (n**2 - 1)))


# ── Krippendorff's α (ordinal) ─────────────────────────────────────────────────


def _krippendorff_alpha_ordinal(ratings: list[list[float]]) -> float:
    """Computes Krippendorff's α with ordinal metric for 1-5 Likert ratings.

    Args:
        ratings: ratings[judge][sample] — shape (n_judges, n_samples).
                 NaN entries are treated as missing and excluded.

    Returns:
        α ∈ [-1, 1]; 1 = perfect agreement, 0 = chance, <0 = systematic disagreement.
        Returns NaN if fewer than 2 judges or fewer than 2 samples have valid data.

    Formula: α = 1 - D_o / D_e
      D_o = observed disagreement: mean pairwise d²(c,k) per item, averaged over items.
      D_e = expected disagreement: d²(c,k) over all rating pairs regardless of item.
      Ordinal metric: d²(c,k) = (c - k)²
    """
    arr = np.array(ratings, dtype=float)  # (n_judges, n_samples)
    n_judges, n_items = arr.shape

    if n_judges < 2:
        return float("nan")

    # Build pairing matrix — only count pairs where both raters gave valid ratings
    d_o_sum = 0.0
    pair_count = 0

    for item in range(n_items):
        col = arr[:, item]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        # All ordered pairs (c, k) within the item
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                d_o_sum += (valid[i] - valid[j]) ** 2
                pair_count += 1

    if pair_count == 0:
        return float("nan")

    D_o = d_o_sum / pair_count

    # Expected disagreement: all pairs from the flattened pool of ratings
    all_ratings = arr[~np.isnan(arr)]
    n_all = len(all_ratings)
    if n_all < 2:
        return float("nan")

    # Vectorised: d_e = mean of (c - k)^2 over all ordered pairs (c != k position-wise)
    d_e_sum = 0.0
    de_count = 0
    for i in range(n_all):
        for j in range(i + 1, n_all):
            d_e_sum += (all_ratings[i] - all_ratings[j]) ** 2
            de_count += 1

    if de_count == 0 or d_e_sum == 0.0:
        return 1.0  # all ratings identical → perfect agreement

    D_e = d_e_sum / de_count
    return float(1.0 - D_o / D_e)


# ── Main interface ─────────────────────────────────────────────────────────────


def evaluate_with_llm_judges(
    predictions: list[str],
    references: list[str],
    api_key: str | None = None,
    max_samples: int = 50,
    max_workers: int | None = None,
    judge_models: list[dict] | None = None,
) -> list[EnsembleScore]:
    """Evaluates summaries with the LLM judge ensemble via OpenRouter.

    Args:
        predictions:  Model-generated summaries.
        references:   Real discharge notes (same order).
        api_key:      OpenRouter key (fallback: OPENROUTER_API_KEY env).
        max_samples:  Limit to max_samples examples (API cost control).
        max_workers:  Parallel threads per example (default: 3 — one per judge).
        judge_models: Custom judge list (default: global JUDGES — 3-model ensemble).

    Returns:
        List of EnsembleScore, one per evaluated sample.
        Each EnsembleScore includes Spearman ρ and Krippendorff's α (ordinal).
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error(
            "OPENROUTER_API_KEY not set — LLM-as-judge evaluation skipped. "
            "Get a key at https://openrouter.ai/keys and set OPENROUTER_API_KEY."
        )
        return []

    active_judges = judge_models or JUDGES
    n = min(len(predictions), len(references), max_samples)
    if n < len(predictions):
        log.info(
            "LLM-as-judge: limited to %d/%d samples (max_samples=%d).",
            n,
            len(predictions),
            max_samples,
        )

    workers = max_workers or int(os.getenv("JUDGE_MAX_WORKERS", "3"))
    ensemble_scores: list[EnsembleScore] = []
    dims = [
        "clinical_accuracy",
        "completeness",
        "coherence",
        "hallucination_free",
        "clinical_utility",
    ]

    log.info(
        "Starting LLM-as-judge evaluation: %d samples × %d judges...",
        n,
        len(active_judges),
    )

    for i in range(n):
        pred = predictions[i]
        ref = references[i]
        judge_scores: list[JudgeScore] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_call_judge, judge, pred, ref, api_key): judge["name"]
                for judge in active_judges
            }
            for future in as_completed(futures):
                judge_scores.append(future.result())

        valid_scores = [s for s in judge_scores if not s.error]

        # Spearman ρ: pairwise rank correlation over the 5 dimensions
        spearman: dict[str, float] = {}
        if len(valid_scores) >= 2:
            for j1 in range(len(valid_scores)):
                for j2 in range(j1 + 1, len(valid_scores)):
                    s1, s2 = valid_scores[j1], valid_scores[j2]
                    x = [getattr(s1, d) for d in dims]
                    y = [getattr(s2, d) for d in dims]
                    spearman[f"{s1.judge_name}_vs_{s2.judge_name}"] = _spearman_rho(
                        x, y
                    )

        # Krippendorff's α: ordinal reliability per dimension and overall
        krippendorff: dict[str, float] = {}
        if len(valid_scores) >= 2:
            for dim in dims:
                # Per-dimension α is computed globally in aggregate_judge_scores
                krippendorff[dim] = float("nan")
            # Overall α across all 5 dimensions (each treated as a separate item)
            ratings_all = [[getattr(s, d) for d in dims] for s in valid_scores]
            krippendorff["overall"] = _krippendorff_alpha_ordinal(ratings_all)

        ensemble_scores.append(
            EnsembleScore(
                scores_per_judge=judge_scores,
                spearman_rho=spearman,
                krippendorff_alpha=krippendorff,
            )
        )

        if (i + 1) % 10 == 0:
            log.info("LLM-as-judge: %d/%d samples evaluated.", i + 1, n)

    # Summary log
    all_means = [
        s.ensemble_mean for s in ensemble_scores if not np.isnan(s.ensemble_mean)
    ]
    if all_means:
        log.info(
            "LLM-as-judge completed: n=%d | ensemble_mean=%.3f ± %.3f",
            len(all_means),
            float(np.mean(all_means)),
            float(np.std(all_means)),
        )

    all_pairs: dict[str, list[float]] = {}
    for s in ensemble_scores:
        for pair, rho in s.spearman_rho.items():
            all_pairs.setdefault(pair, []).append(rho)
    for pair, rhos in all_pairs.items():
        log.info("Spearman ρ (%s): %.3f (n=%d)", pair, float(np.mean(rhos)), len(rhos))

    return ensemble_scores


def aggregate_judge_scores(scores: list[EnsembleScore]) -> dict[str, float]:
    """Aggregates EnsembleScores into summary metrics for the paper.

    Computes:
      - ensemble_mean / std across all samples
      - per-dimension mean / std
      - mean Spearman ρ per judge pair
      - Krippendorff's α (ordinal) per dimension and overall, computed across
        ALL samples (more reliable than per-sample α)

    Returns:
        Dict with all summary metrics (suitable for JSON export).
    """
    if not scores:
        return {}

    dims = [
        "clinical_accuracy",
        "completeness",
        "coherence",
        "hallucination_free",
        "clinical_utility",
    ]
    result: dict[str, float] = {}

    ensemble_means = [s.ensemble_mean for s in scores if not np.isnan(s.ensemble_mean)]
    if ensemble_means:
        result["ensemble_mean"] = round(float(np.mean(ensemble_means)), 4)
        result["ensemble_std"] = round(float(np.std(ensemble_means)), 4)

    for dim in dims:
        vals = [s.per_dimension_mean.get(dim, float("nan")) for s in scores]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            result[f"{dim}_mean"] = round(float(np.mean(vals)), 4)
            result[f"{dim}_std"] = round(float(np.std(vals)), 4)

    # Global Spearman ρ — mean across all samples per judge pair
    all_rhos: dict[str, list[float]] = {}
    for s in scores:
        for pair, rho in s.spearman_rho.items():
            if not np.isnan(rho):
                all_rhos.setdefault(pair, []).append(rho)
    for pair, rhos in all_rhos.items():
        result[f"spearman_rho_{pair}"] = round(float(np.mean(rhos)), 4)

    # Global Krippendorff's α — computed across ALL samples for each dimension.
    # ratings[judge_idx][sample_idx] is more reliable than per-sample α.
    judge_names = list(
        {s.judge_name for es in scores for s in es.scores_per_judge if not s.error}
    )
    if len(judge_names) >= 2:
        for dim in dims:
            ratings_dim: list[list[float]] = []
            for jname in judge_names:
                row = []
                for es in scores:
                    match = next(
                        (
                            s
                            for s in es.scores_per_judge
                            if s.judge_name == jname and not s.error
                        ),
                        None,
                    )
                    row.append(getattr(match, dim) if match else float("nan"))
                ratings_dim.append(row)
            alpha = _krippendorff_alpha_ordinal(ratings_dim)
            result[f"krippendorff_alpha_{dim}"] = round(alpha, 4)

        # Overall α: flatten all 5 dimensions as separate items
        ratings_all: list[list[float]] = []
        for jname in judge_names:
            row = []
            for es in scores:
                match = next(
                    (
                        s
                        for s in es.scores_per_judge
                        if s.judge_name == jname and not s.error
                    ),
                    None,
                )
                if match:
                    row.extend([getattr(match, d) for d in dims])
                else:
                    row.extend([float("nan")] * len(dims))
            ratings_all.append(row)
        result["krippendorff_alpha_overall"] = round(
            _krippendorff_alpha_ordinal(ratings_all), 4
        )
        log.info(
            "Krippendorff α overall=%.3f | %s",
            result["krippendorff_alpha_overall"],
            " | ".join(
                f"{d[:8]}={result[f'krippendorff_alpha_{d}']:.3f}" for d in dims
            ),
        )

    return result
