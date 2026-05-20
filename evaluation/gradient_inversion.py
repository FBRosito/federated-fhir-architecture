"""
gradient_inversion.py
---------------------
Avaliação de segurança: gradient inversion attack sobre gradientes LoRA.

Implementa uma versão adaptada do ataque DLG (Deep Leakage from Gradients,
Zhu et al. 2019) para gradientes de adaptadores LoRA em modelos quantizados.

Objetivo: quantificar a proteção real do DP-SGD contra reconstrução de
dados de pacientes a partir dos gradientes trocados na federação.

Metodologia:
  1. Sem DP: extrair gradientes LoRA de uma amostra e otimizar um input
     dummy para que seus gradientes se aproximem dos reais (DLG).
  2. Com DP: repetir com gradientes ruidosos (σ > 0) e medir degradação
     da qualidade de reconstrução.

Métricas de reconstrução:
  - MSE dos gradientes: ||∇dummy - ∇real||² / ||∇real||²
  - ROUGE-1 entre texto reconstruído e texto original
  - BERTScore entre texto reconstruído e original

Referências:
  - Zhu et al. (2019) "Deep Leakage from Gradients" (NeurIPS 2019)
  - Zhao et al. (2020) "iDLG: Improved Deep Leakage from Gradients"
  - Deng et al. (2021) "TAG: Gradient Attack on Transformer-based Language Models"

AVISO: Esta implementação é para fins de pesquisa defensiva apenas.
       O ataque é computacionalmente intensivo para transformers grandes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GradientInversionResult:
    """Resultado de um ataque DLG sobre um exemplo."""
    original_text:      str
    reconstructed_text: str
    grad_mse_ratio:     float   # ||∇dummy - ∇real||² / ||∇real||²
    rouge_1:            float   # ROUGE-1 entre original e reconstruído
    bertscore_f1:       float   # BERTScore F1
    n_iterations:       int
    dp_noise_sigma:     float   # σ do DP aplicado (0.0 = sem DP)
    converged:          bool

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
    """Calcula ||∇dummy - ∇real||² / ||∇real||² sobre todos os tensores LoRA."""
    import torch
    numerator   = sum((g_d - g_r).pow(2).sum().item() for g_r, g_d in zip(grad_real, grad_dummy))
    denominator = sum(g_r.pow(2).sum().item() for g_r in grad_real)
    return float(numerator / max(denominator, 1e-12))


def run_dlg_attack(
    model,
    tokenizer,
    example_text: str,
    n_iterations: int = 300,
    lr: float = 0.1,
    dp_noise_sigma: float = 0.0,
    max_grad_norm: float = 1.0,
    max_length: int = 128,
    device: str = "cuda",
) -> GradientInversionResult:
    """
    Executa o ataque DLG adaptado para gradientes LoRA de LLMs.

    Args:
        model:           PeftModel com adaptadores LoRA (modo eval para extração de grads).
        tokenizer:       Tokenizador correspondente.
        example_text:    Texto clínico original (label do ataque).
        n_iterations:    Número de iterações de otimização do dummy.
        lr:              Learning rate do otimizador do ataque.
        dp_noise_sigma:  σ do ruído DP aplicado aos gradientes reais (0.0 = sem DP).
        max_grad_norm:   Norma de clipping DP (apenas relevante se sigma > 0).
        max_length:      Comprimento máximo de tokenização.
        device:          Dispositivo de inferência.

    Returns:
        GradientInversionResult com texto reconstruído e métricas.
    """
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    model.eval()

    # ── 1. Extrair gradientes reais ─────────────────────────────────────────
    enc = tokenizer(
        example_text,
        max_length     = max_length,
        truncation     = True,
        padding        = "max_length",
        return_tensors = "pt",
    ).to(device)

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels         = input_ids.clone()

    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    grad_real = [
        p.grad.detach().clone().float()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Aplicar DP ao gradiente real (simula o que o servidor recebe)
    if dp_noise_sigma > 0.0:
        with torch.no_grad():
            total_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_grad_norm
            )
            for i, g in enumerate(grad_real):
                noise = torch.normal(0.0, max_grad_norm * dp_noise_sigma, size=g.shape, device=g.device)
                grad_real[i] = g + noise

    model.zero_grad()

    # ── 2. Otimizar input dummy ──────────────────────────────────────────────
    # Para transformers: otimizamos os embeddings iniciais (contínuos) como proxy.
    # Isso evita a não-diferenciabilidade do argmax no espaço de tokens.
    # Referência: TAG attack (Deng et al., 2021).

    embedding_layer = model.get_input_embeddings()
    dummy_embeds = embedding_layer(input_ids).detach().clone().requires_grad_(True)
    optimizer_atk = torch.optim.Adam([dummy_embeds], lr=lr)

    converged = False
    for iteration in range(n_iterations):
        optimizer_atk.zero_grad()
        model.zero_grad()

        # Forward pass com embeddings dummy
        outputs_dummy = model(
            inputs_embeds  = dummy_embeds,
            attention_mask = attention_mask,
            labels         = labels,
        )
        outputs_dummy.loss.backward(retain_graph=True)

        grad_dummy = [
            p.grad.detach().clone().float()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        # Minimizar distância entre gradientes dummy e reais
        grad_loss = sum(
            (g_d - g_r).pow(2).mean()
            for g_r, g_d in zip(grad_real, grad_dummy)
        )
        model.zero_grad()
        grad_loss.backward()
        optimizer_atk.step()

        if iteration % 50 == 0:
            mse = _compute_grad_mse_ratio(grad_real, grad_dummy)
            log.debug("DLG iter %d/%d | grad_mse=%.4f | atk_loss=%.4f",
                      iteration, n_iterations, mse, grad_loss.item())
            if mse < 1e-4:
                converged = True
                log.info("DLG convergiu na iteração %d.", iteration)
                break

    # ── 3. Decodificar embeddings otimizados → texto ─────────────────────────
    with torch.no_grad():
        # Encontrar o token mais próximo de cada embedding otimizado
        all_embeds = embedding_layer.weight.detach()  # [vocab_size, d_model]
        cosine_sim = torch.nn.functional.normalize(dummy_embeds[0], dim=-1) @ \
                     torch.nn.functional.normalize(all_embeds, dim=-1).T
        best_tokens = cosine_sim.argmax(dim=-1)
        # Mascarar padding
        best_tokens = best_tokens * attention_mask[0]
        reconstructed = tokenizer.decode(best_tokens[attention_mask[0].bool()], skip_special_tokens=True)

    # ── 4. Métricas de reconstrução ──────────────────────────────────────────
    grad_mse = _compute_grad_mse_ratio(grad_real, grad_dummy)

    # ROUGE-1 simples
    orig_tokens = set(example_text.lower().split())
    rec_tokens  = set(reconstructed.lower().split())
    if orig_tokens or rec_tokens:
        overlap = len(orig_tokens & rec_tokens)
        rouge_1 = 2 * overlap / max(len(orig_tokens) + len(rec_tokens), 1)
    else:
        rouge_1 = 0.0

    # BERTScore (opcional — costoso em CPU)
    bertscore_f1 = float("nan")
    try:
        from bert_score import score as bert_score_fn
        _, _, F = bert_score_fn(
            [reconstructed], [example_text],
            model_type="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            lang="en", verbose=False, batch_size=1, rescale_with_baseline=True,
        )
        bertscore_f1 = float(F[0].item())
    except Exception:
        pass

    return GradientInversionResult(
        original_text      = example_text,
        reconstructed_text = reconstructed,
        grad_mse_ratio     = round(grad_mse, 6),
        rouge_1            = round(rouge_1, 4),
        bertscore_f1       = round(bertscore_f1, 4) if not math.isnan(bertscore_f1) else float("nan"),
        n_iterations       = n_iterations,
        dp_noise_sigma     = dp_noise_sigma,
        converged          = converged,
    )


def evaluate_dp_protection(
    model,
    tokenizer,
    examples: list[str],
    sigma_values: list[float],
    max_grad_norm: float = 0.01,
    n_iterations: int = 200,
    max_length: int = 128,
    max_samples: int = 10,
) -> dict[float, list[GradientInversionResult]]:
    """
    Avalia a proteção do DP-SGD contra gradient inversion para múltiplos σ.

    Args:
        model:          PeftModel com adaptadores LoRA.
        tokenizer:      Tokenizador.
        examples:       Textos clínicos reais para o ataque.
        sigma_values:   Lista de σ a testar (ex: [0.0, 0.5, 1.0, 2.0]).
        max_grad_norm:  Norma de clipping DP (calibrada para LoRA).
        n_iterations:   Iterações DLG por amostra.
        max_length:     Comprimento máximo de tokenização.
        max_samples:    Limita o número de amostras atacadas.

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
            log.info("  Amostra %d/%d...", i + 1, n)
            result = run_dlg_attack(
                model          = model,
                tokenizer      = tokenizer,
                example_text   = text,
                n_iterations   = n_iterations,
                dp_noise_sigma = sigma,
                max_grad_norm  = max_grad_norm,
                max_length     = max_length,
                device         = device,
            )
            sigma_results.append(result)
            log.info("    %s", result)
        results[sigma] = sigma_results

        # Resumo por sigma
        rouge_scores = [r.rouge_1 for r in sigma_results if not math.isnan(r.rouge_1)]
        log.info(
            "  σ=%.2f | ROUGE-1 médio: %.4f (n=%d)",
            sigma, float(np.mean(rouge_scores)) if rouge_scores else float("nan"), len(rouge_scores),
        )

    return results
