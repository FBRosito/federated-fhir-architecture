"""
gradient_inversion.py
---------------------
Security evaluation: gradient inversion attack on LoRA gradients.

Implements an adapted version of the DLG attack (Deep Leakage from Gradients,
Zhu et al. 2019) for LoRA adapter gradients in quantized models.

Objective: quantify the real protection of DP-SGD against reconstruction
of patient data from gradients exchanged in the federation.

Methodology:
  1. Without DP: extract LoRA gradients from a sample and optimize a dummy
     input so that its gradients approximate the real ones (DLG).
  2. With DP: repeat with noisy gradients (σ > 0) and measure the degradation
     in reconstruction quality.

Reconstruction metrics:
  - Gradient MSE: ||∇dummy - ∇real||² / ||∇real||²
  - ROUGE-1 between reconstructed text and original text
  - BERTScore between reconstructed and original

References:
  - Zhu et al. (2019) "Deep Leakage from Gradients" (NeurIPS 2019)
  - Zhao et al. (2020) "iDLG: Improved Deep Leakage from Gradients"
  - Deng et al. (2021) "TAG: Gradient Attack on Transformer-based Language Models"

WARNING: This implementation is for defensive research purposes only.
         The attack is computationally intensive for large transformers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)


@dataclass
class GradientInversionResult:
    """Result of a DLG attack on one example."""

    original_text: str
    reconstructed_text: str
    grad_mse_ratio: float  # ||∇dummy - ∇real||² / ||∇real||²
    rouge_1: float  # ROUGE-1 between original and reconstructed
    bertscore_f1: float  # BERTScore F1
    n_iterations: int
    dp_noise_sigma: float  # DP noise σ applied (0.0 = no DP)
    converged: bool

    def __str__(self) -> str:
        return (
            f"GradientInversion(σ={self.dp_noise_sigma:.2f} | "
            f"grad_mse={self.grad_mse_ratio:.4f} | "
            f"rouge1={self.rouge_1:.4f} | bertscore={self.bertscore_f1:.4f} | "
            f"converged={self.converged} | iters={self.n_iterations})"
        )


def _compute_grad_mse_ratio(
    grad_real: list,
    grad_dummy: list,
) -> float:
    """Computes ||∇dummy - ∇real||² / ||∇real||² across all LoRA tensors."""
    numerator = sum(
        (g_d - g_r).pow(2).sum().item() for g_r, g_d in zip(grad_real, grad_dummy)
    )
    denominator = sum(g_r.pow(2).sum().item() for g_r in grad_real)
    return float(numerator / max(denominator, 1e-12))


def run_dlg_attack(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    example_text: str,
    n_iterations: int = 300,
    lr: float = 0.1,
    dp_noise_sigma: float = 0.0,
    max_grad_norm: float = 1.0,
    max_length: int = 128,
    device: str = "cuda",
) -> GradientInversionResult:
    """
    Runs the DLG attack adapted for LoRA gradients of LLMs.

    Args:
        model:           PeftModel with LoRA adapters (eval mode for gradient extraction).
        tokenizer:       Corresponding tokenizer.
        example_text:    Original clinical text (attack label).
        n_iterations:    Number of dummy optimization iterations.
        lr:              Attacker optimizer learning rate.
        dp_noise_sigma:  σ of DP noise applied to real gradients (0.0 = no DP).
        max_grad_norm:   DP clipping norm (only relevant if sigma > 0).
        max_length:      Maximum tokenization length.
        device:          Inference device.

    Returns:
        GradientInversionResult with reconstructed text and metrics.
    """
    import torch

    model = model.to(device)
    model.eval()

    # ── 1. Extract real gradients ───────────────────────────────────────────
    enc = tokenizer(
        example_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()

    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    grad_real = [
        p.grad.detach().clone().float()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Apply DP to the real gradient (simulates what the server receives)
    if dp_noise_sigma > 0.0:
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            for i, g in enumerate(grad_real):
                noise = torch.normal(
                    0.0, max_grad_norm * dp_noise_sigma, size=g.shape, device=g.device
                )
                grad_real[i] = g + noise

    model.zero_grad()

    # ── 2. Optimize dummy input ──────────────────────────────────────────────
    # For transformers: we optimize the initial (continuous) embeddings as a proxy.
    # This avoids the non-differentiability of argmax in token space.
    # Reference: TAG attack (Deng et al., 2021).

    embedding_layer = model.get_input_embeddings()
    dummy_embeds = embedding_layer(input_ids).detach().clone().requires_grad_(True)
    optimizer_atk = torch.optim.Adam([dummy_embeds], lr=lr)

    converged = False
    for iteration in range(n_iterations):
        optimizer_atk.zero_grad()
        model.zero_grad()

        # Forward pass with dummy embeddings
        outputs_dummy = model(
            inputs_embeds=dummy_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        outputs_dummy.loss.backward(retain_graph=True)

        grad_dummy = [
            p.grad.detach().clone().float()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        # Minimize distance between dummy and real gradients
        grad_loss = sum(
            (g_d - g_r).pow(2).mean() for g_r, g_d in zip(grad_real, grad_dummy)
        )
        model.zero_grad()
        grad_loss.backward()
        optimizer_atk.step()

        if iteration % 50 == 0:
            mse = _compute_grad_mse_ratio(grad_real, grad_dummy)
            log.debug(
                "DLG iter %d/%d | grad_mse=%.4f | atk_loss=%.4f",
                iteration,
                n_iterations,
                mse,
                grad_loss.item(),
            )
            if mse < 1e-4:
                converged = True
                log.info("DLG converged at iteration %d.", iteration)
                break

    # ── 3. Decode optimized embeddings → text ──────────────────────────────
    with torch.no_grad():
        # Find the nearest token for each optimized embedding
        all_embeds = embedding_layer.weight.detach()  # [vocab_size, d_model]
        cosine_sim = (
            torch.nn.functional.normalize(dummy_embeds[0], dim=-1)
            @ torch.nn.functional.normalize(all_embeds, dim=-1).T
        )
        best_tokens = cosine_sim.argmax(dim=-1)
        # Mask padding
        best_tokens = best_tokens * attention_mask[0]
        reconstructed = tokenizer.decode(
            best_tokens[attention_mask[0].bool()], skip_special_tokens=True
        )

    # ── 4. Reconstruction metrics ───────────────────────────────────────────
    grad_mse = _compute_grad_mse_ratio(grad_real, grad_dummy)

    # Simple ROUGE-1
    orig_tokens = set(example_text.lower().split())
    rec_tokens = set(reconstructed.lower().split())
    if orig_tokens or rec_tokens:
        overlap = len(orig_tokens & rec_tokens)
        rouge_1 = 2 * overlap / max(len(orig_tokens) + len(rec_tokens), 1)
    else:
        rouge_1 = 0.0

    # BERTScore (optional — expensive on CPU)
    bertscore_f1 = float("nan")
    try:
        from bert_score import score as bert_score_fn

        _, _, F = bert_score_fn(
            [reconstructed],
            [example_text],
            model_type="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            lang="en",
            verbose=False,
            batch_size=1,
            rescale_with_baseline=True,
        )
        bertscore_f1 = float(F[0].item())
    except Exception:
        pass

    return GradientInversionResult(
        original_text=example_text,
        reconstructed_text=reconstructed,
        grad_mse_ratio=round(grad_mse, 6),
        rouge_1=round(rouge_1, 4),
        bertscore_f1=(
            round(bertscore_f1, 4) if not math.isnan(bertscore_f1) else float("nan")
        ),
        n_iterations=n_iterations,
        dp_noise_sigma=dp_noise_sigma,
        converged=converged,
    )


def evaluate_dp_protection(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[str],
    sigma_values: list[float],
    max_grad_norm: float = 0.01,
    n_iterations: int = 200,
    max_length: int = 128,
    max_samples: int = 10,
) -> dict[float, list[GradientInversionResult]]:
    """
    Evaluates DP-SGD protection against gradient inversion for multiple σ values.

    Args:
        model:          PeftModel with LoRA adapters.
        tokenizer:      Tokeniser.
        examples:       Real clinical texts to attack.
        sigma_values:   List of σ values to test (e.g. [0.0, 0.5, 1.0, 2.0]).
        max_grad_norm:  DP clipping norm (calibrated for LoRA).
        n_iterations:   DLG iterations per sample.
        max_length:     Maximum tokenisation length.
        max_samples:    Limits the number of attacked samples.

    Returns:
        {sigma: [GradientInversionResult, ...]}
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = min(len(examples), max_samples)
    results: dict[float, list[GradientInversionResult]] = {}

    for sigma in sigma_values:
        log.info("Gradient inversion attack: σ=%.2f | n_samples=%d", sigma, n)
        sigma_results = []
        for i, text in enumerate(examples[:n]):
            log.info("  Sample %d/%d...", i + 1, n)
            result = run_dlg_attack(
                model=model,
                tokenizer=tokenizer,
                example_text=text,
                n_iterations=n_iterations,
                dp_noise_sigma=sigma,
                max_grad_norm=max_grad_norm,
                max_length=max_length,
                device=device,
            )
            sigma_results.append(result)
            log.info("    %s", result)
        results[sigma] = sigma_results

        # Summary per sigma
        rouge_scores = [r.rouge_1 for r in sigma_results if not math.isnan(r.rouge_1)]
        log.info(
            "  σ=%.2f | mean ROUGE-1: %.4f (n=%d)",
            sigma,
            float(np.mean(rouge_scores)) if rouge_scores else float("nan"),
            len(rouge_scores),
        )

    return results
