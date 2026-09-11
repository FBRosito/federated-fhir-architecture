"""
post_eval.py — Post-training LLM-as-judge evaluation for Experiment B.

Loads a trained LoRA checkpoint produced by FL_SAVE_CHECKPOINT, generates
discharge summaries for up to N FHIR examples, runs the 3-judge ensemble
(Qwen2.5-72B + Gemma-3-27B + DeepSeek-R1), and appends results to the run JSON.

Usage:
    python -m evaluation.post_eval \\
        --run-json experiment_logs/fl_fedprox_alpha0.5_nodp_llm_seed42.json \\
        --checkpoint /tmp/checkpoints/silo_0 \\
        --fhir-url http://localhost:8080/fhir \\
        --max-samples 50

Requires: OPENROUTER_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

_DEFAULT_FHIR_URL = os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir")
_DEFAULT_BASE_MODEL = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
_DEFAULT_MAX_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))


def load_model_from_checkpoint(
    checkpoint_dir: str,
    base_model_name: str,
    device: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads base model + LoRA adapter weights from checkpoint_dir.

    The base model is loaded at NF4 4-bit precision (BitsAndBytes) as during
    FL training. LoRA weights saved by save_pretrained() are merged on top.

    Args:
        checkpoint_dir: Directory written by model.save_pretrained() after FL.
        base_model_name: HuggingFace model ID (must match the base used in FL).
        device: Device map for model placement ("auto", "cuda", "cpu").

    Returns:
        (model, tokenizer) ready for inference.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install transformers, peft, and bitsandbytes."
        )

    log.info("Loading base model %s (NF4 4-bit)...", base_model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )
    log.info("Loading LoRA adapter from %s...", checkpoint_dir)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    log.info("Model loaded successfully.")
    return model, tokenizer


def generate_summaries(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list,
    max_input_length: int = 384,
    max_new_tokens: int = 256,
) -> list[str]:
    """
    Generates discharge summaries for a list of SummarizationExamples.

    Uses greedy decoding with beam search (num_beams=4) for deterministic output
    suitable for ROUGE/BERTScore comparison.

    Args:
        model:            Loaded model (base + LoRA adapter).
        tokenizer:        Corresponding tokenizer.
        examples:         List of SummarizationExample objects.
        max_input_length: Max tokens for the input prompt (leaves room for generation).
        max_new_tokens:   Max new tokens to generate.

    Returns:
        List of generated summary strings, aligned with examples.
    """
    import torch

    device = next(model.parameters()).device
    predictions: list[str] = []

    for i, ex in enumerate(examples):
        prompt = ex.to_inference_prompt()
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the input prefix)
        new_tokens = output_ids[0][input_ids.shape[1] :]
        summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predictions.append(summary)

        if (i + 1) % 10 == 0:
            log.info("  Generated %d/%d summaries...", i + 1, len(examples))

    return predictions


def _stratified_judge_sample(
    examples: list,
    predictions: list[str],
    references: list[str],
    n_per_length_bin: int = 50,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int], dict]:
    """Stratified random sample for LLM-as-judge evaluation.

    Stratification grid (primary × secondary):
      Primary axis  — Document length: short / medium / long
                      Defined by tertile cutoffs on reference summary word count.
                      Target: n_per_length_bin examples per bin (default 50 → 150 total).
      Secondary axis — Complexity: few-codes vs many-codes.
                      Threshold is the MEDIAN ICD-10 count of the dataset, so both
                      groups are always ~50/50 regardless of the dataset's code
                      distribution (MIMIC-IV median ~16 codes; hardcoded ≤1 would
                      leave the "few" bucket nearly empty on ICU data).
                      Within each length bin, samples are drawn proportional to the
                      natural few/many ratio so both are always represented.

    Graceful degradation:
      - Bins with fewer examples than n_per_length_bin contribute all they have.
      - If one complexity group is exhausted, the remainder is drawn from the other.

    Args:
        examples:          List of SummarizationExample objects (full test set).
        predictions:       Generated summaries aligned with examples.
        references:        Reference summaries aligned with examples.
        n_per_length_bin:  Target samples per length bin (default 50).
        seed:              RNG seed for reproducibility.

    Returns:
        (judge_preds, judge_refs, judge_indices, strata_report)
        strata_report — dict with per-bin counts for logging/JSON output.
    """
    import random as _rnd
    import statistics as _stats

    _rnd.seed(seed)
    n = len(examples)

    # ── Length proxy: word count of reference summary ──
    lengths = [len(ex.reference_summary.split()) for ex in examples]
    sorted_lengths = sorted(lengths)
    # Tertile boundaries (values, not indices — avoids off-by-one on uneven n)
    t1 = sorted_lengths[n // 3]
    t2 = sorted_lengths[(2 * n) // 3]

    def _length_bin(wc: int) -> str:
        if wc <= t1:
            return "short"
        elif wc <= t2:
            return "medium"
        return "long"

    # ── Complexity proxy: ICD-10 code count, split at dataset median ──
    # Using the median as threshold rather than a hardcoded value ensures a
    # balanced split on any clinical dataset (MIMIC-IV median ≈ 16 codes;
    # ≤1 would leave the "few" group nearly empty on ICU data).
    icd_counts = [len(getattr(ex, "icd10_codes", [])) for ex in examples]
    icd_median = _stats.median(icd_counts)

    def _complexity(ex) -> str:
        return "few" if len(getattr(ex, "icd10_codes", [])) <= icd_median else "many"

    # ── Build 3×2 strata pools ──
    strata: dict[str, dict[str, list[int]]] = {
        "short": {"few": [], "many": []},
        "medium": {"few": [], "many": []},
        "long": {"few": [], "many": []},
    }
    for i, ex in enumerate(examples):
        strata[_length_bin(lengths[i])][_complexity(ex)].append(i)

    selected: list[int] = []
    report: dict = {"icd_median_threshold": icd_median}

    for bin_name in ("short", "medium", "long"):
        f_pool = strata[bin_name]["few"]
        m_pool = strata[bin_name]["many"]
        n_f, n_m = len(f_pool), len(m_pool)
        n_bin_total = n_f + n_m

        if n_bin_total == 0:
            report[bin_name] = {
                "available_few": 0,
                "available_many": 0,
                "sampled_few": 0,
                "sampled_many": 0,
                "sampled_total": 0,
            }
            continue

        if n_bin_total <= n_per_length_bin:
            draw_f, draw_m = n_f, n_m
        else:
            # Proportional allocation preserving natural complexity ratio
            draw_f = round(n_per_length_bin * n_f / n_bin_total)
            draw_m = n_per_length_bin - draw_f
            # Clamp to available, then fill any shortfall from the other group
            draw_f = min(draw_f, n_f)
            draw_m = min(draw_m, n_m)
            shortfall = n_per_length_bin - draw_f - draw_m
            if shortfall > 0:
                if draw_f < n_f:
                    draw_f = min(draw_f + shortfall, n_f)
                else:
                    draw_m = min(draw_m + shortfall, n_m)

        _rnd.shuffle(f_pool)
        _rnd.shuffle(m_pool)
        selected.extend(f_pool[:draw_f])
        selected.extend(m_pool[:draw_m])
        report[bin_name] = {
            "available_few": n_f,
            "available_many": n_m,
            "sampled_few": draw_f,
            "sampled_many": draw_m,
            "sampled_total": draw_f + draw_m,
        }

    selected = sorted(selected)
    return (
        [predictions[i] for i in selected],
        [references[i] for i in selected],
        selected,
        report,
    )


def run_post_eval(
    run_json_path: str,
    checkpoint_dir: str,
    fhir_url: str = _DEFAULT_FHIR_URL,
    base_model_name: str = _DEFAULT_BASE_MODEL,
    max_samples: int = 0,
    judge_samples: int = 150,
    max_input_length: int = 384,
    max_new_tokens: int = 256,
    judges: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Full post-evaluation pipeline for Experiment B.

    Two-tier evaluation strategy:
      1. ROUGE + BERTScore on ALL generated summaries (free/local, proves
         architectural stability across the full test set).
      2. LLM-as-judge on a STRATIFIED sample of judge_samples (paid API, proves
         semantic quality and clinical safety on representative cases).
         Stratification: 3 length bins (short/medium/long) × 2 complexity groups
         (simple ≤1 ICD vs complex ≥2 ICD) → 50+50+50 = 150 examples.
         150 samples with Wilcoxon signed-rank provides p<0.05 power for publication.

    Args:
        run_json_path:    Path to the run JSON from run_experiments.sh.
        checkpoint_dir:   Path to LoRA checkpoint directory.
        fhir_url:         FHIR R4 server URL.
        base_model_name:  HuggingFace model ID for the base model.
        max_samples:      Total examples to generate for (0 = no limit).
        judge_samples:    Examples passed to LLM judges (random sample, default 150).
        max_input_length: Max input tokens for generation.
        max_new_tokens:   Max tokens to generate per example.
        judges:           LLM judge model IDs (default: global JUDGES ensemble).
        seed:             RNG seed for reproducible judge sampling.

    Returns:
        Updated run JSON dict with "post_eval" key added.
    """
    from ai_client.fhir_consumer_summarization import fetch_summarization_examples
    from evaluation.llm_judge import evaluate_with_llm_judges
    from evaluation.summarization_metrics import evaluate_summaries

    # ── Load run JSON ──
    run_path = Path(run_json_path)
    with run_path.open(encoding="utf-8") as f:
        run_data = json.load(f)

    log.info("Post-eval for run: %s", run_data.get("tag", run_json_path))
    log.info(
        "Checkpoint: %s | FHIR: %s | max_samples=%s | judge_samples=%d",
        checkpoint_dir,
        fhir_url,
        max_samples or "all",
        judge_samples,
    )

    # ── Load model ──
    model, tokenizer = load_model_from_checkpoint(checkpoint_dir, base_model_name)

    # ── Fetch FHIR examples ──
    log.info("Fetching summarization examples from %s...", fhir_url)
    examples, fetch_stats = fetch_summarization_examples(
        fhir_url, max_examples=max_samples if max_samples > 0 else 0
    )

    if not examples:
        log.warning("No summarization examples fetched — post-eval aborted.")
        return run_data

    log.info(
        "Fetched %d examples (missing_summary=%d).",
        len(examples),
        fetch_stats.get("missing_summary", 0),
    )

    references = [ex.reference_summary for ex in examples]
    examples_with_ref = [ex for ex in examples if ex.reference_summary]
    if len(examples_with_ref) < len(examples):
        log.warning(
            "%d examples have no reference summary — using examples with references only.",
            len(examples) - len(examples_with_ref),
        )
        examples = examples_with_ref
        references = [ex.reference_summary for ex in examples]

    # ── Generate summaries ──
    log.info("Generating summaries for %d examples...", len(examples))
    t0 = time.perf_counter()
    predictions = generate_summaries(
        model,
        tokenizer,
        examples,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )
    gen_time = time.perf_counter() - t0
    log.info(
        "Generation completed in %.1fs (%.2fs/example).",
        gen_time,
        gen_time / max(len(examples), 1),
    )

    # ── ROUGE + BERTScore ──
    log.info("Computing ROUGE and BERTScore...")
    summ_metrics = evaluate_summaries(predictions, references)
    auto_metrics = summ_metrics.to_flat_dict()
    log.info(
        "ROUGE-1=%.4f | ROUGE-2=%.4f | ROUGE-L=%.4f | BERTScore-F1=%.4f",
        auto_metrics.get("rouge_1", float("nan")),
        auto_metrics.get("rouge_2", float("nan")),
        auto_metrics.get("rouge_l", float("nan")),
        auto_metrics.get("bertscore_f1", float("nan")),
    )

    # ── LLM-as-judge ensemble — stratified subsample ──
    # ROUGE/BERTScore ran on 100% of the test set (free/local — proves stability).
    # LLM judges run on a stratified sample: length (short/medium/long) ×
    # complexity (simple ≤1 ICD / complex ≥2 ICD) → 50+50+50 = 150 examples.
    # This guarantees both document-length and clinical-complexity coverage,
    # which pure random sampling can miss on skewed clinical datasets.
    n_per_bin = judge_samples // 3
    judge_preds, judge_refs, judge_indices, strata_report = _stratified_judge_sample(
        examples,
        predictions,
        references,
        n_per_length_bin=n_per_bin,
        seed=seed,
    )
    n_judge = len(judge_indices)
    log.info(
        "LLM-as-judge: stratified sample %d/%d examples (seed=%d). "
        "ROUGE/BERTScore computed on all %d.",
        n_judge,
        len(predictions),
        seed,
        len(predictions),
    )
    log.info(
        "  ICD median threshold: %.1f codes",
        strata_report.get("icd_median_threshold", 0),
    )
    for bin_name, bin_stats in strata_report.items():
        if not isinstance(bin_stats, dict):
            continue
        log.info(
            "  length=%-6s: %2d sampled (%d few-codes + %d many-codes) "
            "from %d available",
            bin_name,
            bin_stats["sampled_total"],
            bin_stats["sampled_few"],
            bin_stats["sampled_many"],
            bin_stats["available_few"] + bin_stats["available_many"],
        )
    judge_result = evaluate_with_llm_judges(
        predictions=judge_preds,
        references=judge_refs,
        judge_models=judges,
    )

    # ── Update run JSON ──
    run_data["post_eval"] = {
        "n_samples": len(examples),
        "n_judge_samples": n_judge,
        "generation_time_s": round(gen_time, 2),
        "auto_metrics": auto_metrics,
        "judge_strata": strata_report,
        "llm_judge": judge_result,
        "sample_outputs": [
            {
                "prediction": predictions[i][:500],
                "reference": references[i][:500],
            }
            for i in range(min(5, len(predictions)))
        ],
    }

    run_path.write_text(
        json.dumps(run_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Run JSON updated: %s", run_path)
    return run_data


def main() -> None:
    """CLI entry point: generate summaries and run the LLM-as-judge ensemble."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Post-training LLM-as-judge evaluation for Experiment B (discharge summary)."
    )
    parser.add_argument(
        "--run-json",
        required=True,
        help="Path to the run JSON produced by run_experiments.sh.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the LoRA checkpoint directory (FL_SAVE_CHECKPOINT output).",
    )
    parser.add_argument(
        "--fhir-url",
        default=_DEFAULT_FHIR_URL,
        help="FHIR R4 server base URL.",
    )
    parser.add_argument(
        "--base-model",
        default=_DEFAULT_BASE_MODEL,
        help="HuggingFace model ID for the base model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max FHIR examples for ROUGE/BERTScore (0 = no limit, fetch all test set).",
    )
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=150,
        help="Random subsample size for LLM-as-judge (default 150; sufficient for Wilcoxon p<0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible judge sampling.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=384,
        help="Max input tokens for the generation prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per example.",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="LLM judge model IDs (default: Qwen2.5-72B, Gemma-3-27B, DeepSeek-V3.1).",
    )
    args = parser.parse_args()

    result = run_post_eval(
        run_json_path=args.run_json,
        checkpoint_dir=args.checkpoint,
        fhir_url=args.fhir_url,
        base_model_name=args.base_model,
        max_samples=args.max_samples,
        judge_samples=args.judge_samples,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        judges=args.judges,
        seed=args.seed,
    )

    summary = result.get("post_eval", {})
    print(
        json.dumps(
            {
                "n_samples": summary.get("n_samples"),
                "auto_metrics": summary.get("auto_metrics"),
                "llm_judge": {
                    k: v
                    for k, v in summary.get("llm_judge", {}).items()
                    if k != "per_sample"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
