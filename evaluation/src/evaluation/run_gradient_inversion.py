"""
run_gradient_inversion.py — DLG gradient inversion security evaluation.

Quantifies how much patient data can be reconstructed from gradients intercepted
by an adversarial FL server, before and after DP noise is applied.

Threat model:
  An adversarial FL server intercepts the gradient update from a client during
  round 1 (model at initialization — worst case, no DP yet applied). The DLG
  attack (Zhu et al., 2019) attempts to reconstruct the original clinical text
  from the gradient alone.

Design:
  - Uses a FRESH model (no training needed): DLG reconstructs from ANY gradient.
  - Compares σ=0.0 (no DP) vs σ>0 (DP-SGD with noise) to show raw DP protection.
  - Reports ROUGE-1 and BERTScore(F1) between original text and DLG reconstruction.

Usage:
    python -m evaluation.run_gradient_inversion \\
        --fhir-url http://localhost:8080/fhir \\
        --sigma-values 0.0 0.5 1.0 2.0 \\
        --n-samples 10 \\
        --n-iterations 300 \\
        --output experiment_logs/gradient_inversion.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_FHIR_URL   = os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir")
_DEFAULT_BASE_MODEL = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
_DEFAULT_MAX_LEN    = int(os.getenv("MAX_SEQ_LEN", "512"))


def _load_fresh_model(base_model_name: str, max_seq_len: int):
    """
    Loads a fresh (untrained) base model at NF4 4-bit precision.

    Returns (model, tokenizer). Model parameters are at random initialization —
    this represents the worst-case attack scenario (round 1, no prior training).
    """
    try:
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Install transformers, peft, bitsandbytes.")

    log.info("Loading fresh model %s (NF4 4-bit) for DLG attack...", base_model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_len

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA to match the FL training setup (only LoRA gradients are shared)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    log.info("Fresh model ready.")
    return model, tokenizer


def run_attack_suite(
    fhir_url: str = _DEFAULT_FHIR_URL,
    base_model_name: str = _DEFAULT_BASE_MODEL,
    sigma_values: list[float] | None = None,
    n_samples: int = 10,
    n_iterations: int = 300,
    max_seq_len: int = _DEFAULT_MAX_LEN,
    output_path: str = "experiment_logs/gradient_inversion.json",
) -> dict:
    """
    Runs the DLG attack for each σ value and reports ROUGE-1 and BERTScore.

    Args:
        fhir_url:        HAPI FHIR server URL to fetch clinical text samples.
        base_model_name: HuggingFace model ID (must match FL training config).
        sigma_values:    List of DP noise multipliers to evaluate.
                         σ=0.0 → no DP (maximum leakage).
                         σ=2.0 → strong DP (minimum leakage).
        n_samples:       Number of FHIR examples per σ value.
        n_iterations:    DLG optimization iterations per sample.
        max_seq_len:     Max token sequence length.
        output_path:     Path to write gradient_inversion.json.

    Returns:
        Results dict (also written to output_path).
    """
    from evaluation.gradient_inversion import run_dlg_attack, GradientInversionResult

    if sigma_values is None:
        sigma_values = [0.0, 0.5, 1.0, 2.0]

    # ── Fetch FHIR examples ──
    log.info("Fetching %d clinical text samples from %s...", n_samples, fhir_url)
    try:
        from ai_client.fhir_consumer import fetch_training_examples
        examples, stats = fetch_training_examples(fhir_url, max_examples=n_samples)
    except Exception as exc:
        log.error("Failed to fetch FHIR examples: %s", exc)
        raise

    if not examples:
        raise RuntimeError(
            f"No examples fetched from {fhir_url}. "
            "Ensure HAPI FHIR is running and contains data."
        )

    original_texts = [ex.clinical_text for ex in examples[:n_samples]]
    log.info("Using %d examples for gradient inversion.", len(original_texts))

    # ── Load fresh model ──
    model, tokenizer = _load_fresh_model(base_model_name, max_seq_len)

    # ── Run attack for each σ ──
    results_by_sigma: dict[str, dict] = {}
    overall_start = time.perf_counter()

    for sigma in sigma_values:
        log.info("=" * 60)
        log.info("Running DLG attack with σ=%.2f (%d samples, %d iterations)...",
                 sigma, len(original_texts), n_iterations)

        sigma_start = time.perf_counter()
        attack_result: GradientInversionResult = run_dlg_attack(
            model=model,
            tokenizer=tokenizer,
            original_texts=original_texts,
            noise_multiplier=sigma,
            n_iterations=n_iterations,
            max_seq_len=max_seq_len,
        )
        sigma_elapsed = time.perf_counter() - sigma_start

        results_by_sigma[f"sigma_{sigma:.2f}"] = {
            "sigma":              sigma,
            "n_samples":          len(original_texts),
            "n_iterations":       n_iterations,
            "elapsed_s":          round(sigma_elapsed, 2),
            "mean_rouge1":        round(attack_result.mean_rouge1, 4),
            "mean_bertscore_f1":  round(attack_result.mean_bertscore_f1, 4),
            "mean_dlg_loss":      round(attack_result.mean_dlg_loss, 6),
            "per_sample":         [
                {
                    "original_snippet":     s.original_text[:200],
                    "reconstructed_snippet": s.reconstructed_text[:200],
                    "rouge1":               round(s.rouge1, 4),
                    "bertscore_f1":         round(s.bertscore_f1, 4),
                }
                for s in attack_result.per_sample
            ],
        }

        log.info(
            "σ=%.2f | mean ROUGE-1=%.4f | mean BERTScore-F1=%.4f | elapsed=%.1fs",
            sigma,
            attack_result.mean_rouge1,
            attack_result.mean_bertscore_f1,
            sigma_elapsed,
        )

    total_elapsed = time.perf_counter() - overall_start

    # ── Compute DP protection gain ──
    no_dp   = results_by_sigma.get("sigma_0.00", {})
    best_dp = min(
        (v for k, v in results_by_sigma.items() if k != "sigma_0.00"),
        key=lambda v: v.get("mean_rouge1", 1.0),
        default=None,
    )
    protection_summary = {}
    if no_dp and best_dp:
        r1_reduction = no_dp["mean_rouge1"] - best_dp["mean_rouge1"]
        bs_reduction = no_dp["mean_bertscore_f1"] - best_dp["mean_bertscore_f1"]
        protection_summary = {
            "rouge1_reduction_vs_no_dp":     round(r1_reduction, 4),
            "bertscore_reduction_vs_no_dp":  round(bs_reduction, 4),
            "best_sigma":                    best_dp["sigma"],
        }
        log.info(
            "DP protection: ROUGE-1 reduced by %.4f | BERTScore-F1 reduced by %.4f "
            "(σ=0 vs σ=%.2f)",
            r1_reduction, bs_reduction, best_dp["sigma"],
        )

    output = {
        "experiment":        "gradient_inversion",
        "fhir_url":          fhir_url,
        "base_model":        base_model_name,
        "sigma_values":      sigma_values,
        "n_samples":         len(original_texts),
        "n_iterations":      n_iterations,
        "total_elapsed_s":   round(total_elapsed, 2),
        "protection_summary": protection_summary,
        "results_by_sigma":  results_by_sigma,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Gradient inversion results saved to: %s", out_path)

    return output


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="DLG gradient inversion security evaluation for FL clinical NLP."
    )
    parser.add_argument(
        "--fhir-url",
        default=_DEFAULT_FHIR_URL,
        help="HAPI FHIR server base URL.",
    )
    parser.add_argument(
        "--base-model",
        default=_DEFAULT_BASE_MODEL,
        help="HuggingFace model ID (must match the FL training configuration).",
    )
    parser.add_argument(
        "--sigma-values",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0],
        help="DP noise multiplier values to evaluate. σ=0.0 = no DP.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of FHIR examples to use per σ value.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=300,
        help="DLG optimization iterations per sample.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=_DEFAULT_MAX_LEN,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--output",
        default="experiment_logs/gradient_inversion.json",
        help="Output path for gradient_inversion.json.",
    )
    args = parser.parse_args()

    result = run_attack_suite(
        fhir_url=args.fhir_url,
        base_model_name=args.base_model,
        sigma_values=args.sigma_values,
        n_samples=args.n_samples,
        n_iterations=args.n_iterations,
        max_seq_len=args.max_seq_len,
        output_path=args.output,
    )

    # Print compact summary
    print("\n=== Gradient Inversion Summary ===")
    for key, res in result.get("results_by_sigma", {}).items():
        print(
            f"  σ={res['sigma']:.2f} | ROUGE-1={res['mean_rouge1']:.4f} "
            f"| BERTScore-F1={res['mean_bertscore_f1']:.4f}"
        )

    ps = result.get("protection_summary", {})
    if ps:
        print(
            f"\nDP protection (σ=0 vs σ={ps.get('best_sigma', '?'):.2f}):\n"
            f"  ROUGE-1 reduction:    {ps.get('rouge1_reduction_vs_no_dp', 0):.4f}\n"
            f"  BERTScore reduction:  {ps.get('bertscore_reduction_vs_no_dp', 0):.4f}"
        )

    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
