"""
model_setup.py
--------------
Configura uma LLM open-source (Llama-3-8B por padrão) com quantização
4-bits via BitsAndBytesConfig, aplica adaptadores LoRA (PEFT) sobre os pesos
base congelados para a tarefa de extração de códigos CID-10, e expõe uma
função de treinamento PyTorch padrão compatível com o loop federado do Flower.

Fluxo:
    1. load_quantized_model()   → modelo base NF4 em 4-bits
    2. apply_lora()             → adiciona adaptadores LoRA treináveis
    3. build_dataset()          → tokeniza os exemplos do fhir_consumer
    4. train_one_round()        → loop PyTorch + retorna pesos LoRA para o Flower

Requisitos de hardware:
    - GPU NVIDIA com CUDA ≥ 12.4 e ≥ 10 GB VRAM (recomendado 16 GB+).
    - bitsandbytes ≥ 0.43 instalado no ambiente CUDA.

Variáveis de ambiente:
    MODEL_NAME      ID HuggingFace do modelo base (default: meta-llama/Meta-Llama-3-8B-Instruct)
    HF_TOKEN        Token de acesso HuggingFace (necessário para modelos gated)
    MAX_SEQ_LEN     Comprimento máximo de sequência para tokenização (default: 512)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# fhir_consumer está no mesmo pacote — importação relativa ao workspace
from ai_client.fhir_consumer import TrainingExample

log = logging.getLogger(__name__)

# ── Constantes e defaults ─────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))

# Módulos de atenção e FFN do Llama-3 que receberão adaptadores LoRA.
# q/v são suficientes para tarefas de extração; inclua k/o/gate/up/down
# para maior capacidade expressiva (a custo de mais parâmetros treináveis).
LLAMA3_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ── Configurações ─────────────────────────────────────────────────────────────

@dataclass
class QuantizationConfig:
    """Parâmetros de quantização NF4 via bitsandbytes."""
    load_in_4bit:             bool  = True
    bnb_4bit_quant_type:      str   = "nf4"           # Normal Float 4
    bnb_4bit_use_double_quant: bool = True             # dupla quantização economiza ~0.4 bpp
    bnb_4bit_compute_dtype:   str   = "bfloat16"       # dtype da aritmética de forward/backward


@dataclass
class LoRAAdapterConfig:
    """Hiperparâmetros dos adaptadores LoRA."""
    r:               int        = 16          # rank da decomposição (capacidade ↑ com r ↑)
    lora_alpha:      int        = 32          # escala efetiva = lora_alpha / r = 2.0
    lora_dropout:    float      = 0.05
    bias:            str        = "none"      # "none" | "all" | "lora_only"
    use_rslora:      bool       = False       # RSLoRA: normaliza escala pelo sqrt(r)
    target_modules:  list[str]  = field(default_factory=lambda: LLAMA3_LORA_TARGET_MODULES)


@dataclass
class TrainingConfig:
    """Configuração do loop de treinamento federado."""
    learning_rate:        float = 2e-4
    weight_decay:         float = 0.01
    num_epochs:           int   = 1           # épocas por round federado
    batch_size:           int   = 2           # pequeno por limitação de VRAM
    gradient_accum_steps: int   = 8           # batch efetivo = batch_size × accum_steps = 16
    max_grad_norm:        float = 1.0
    warmup_ratio:         float = 0.1         # fração de steps usados para warmup
    use_amp:              bool  = True         # mixed-precision (bf16 se suportado)


# ── Carregamento do modelo quantizado ────────────────────────────────────────

def build_bnb_config(cfg: QuantizationConfig) -> BitsAndBytesConfig:
    """Constrói o BitsAndBytesConfig a partir dos parâmetros de quantização."""
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit               = cfg.load_in_4bit,
        bnb_4bit_quant_type        = cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant  = cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype     = compute_dtype,
    )


def load_quantized_model(
    model_name: str = DEFAULT_MODEL_NAME,
    quant_cfg: QuantizationConfig | None = None,
    device_map: str | dict = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Carrega o modelo e o tokenizador com quantização NF4 4-bits.

    Os pesos base são carregados em NF4 e **permanecem congelados**.
    Nenhum gradiente é calculado para eles durante o treinamento.

    Args:
        model_name: ID HuggingFace ou caminho local do modelo.
        quant_cfg:  Parâmetros de quantização (usa QuantizationConfig() se None).
        device_map: Mapeamento de dispositivos para sharding multi-GPU ("auto"
                    distribui automaticamente entre as GPUs disponíveis).

    Returns:
        (model, tokenizer) prontos para `apply_lora()`.
    """
    if quant_cfg is None:
        quant_cfg = QuantizationConfig()

    bnb_config = build_bnb_config(quant_cfg)
    hf_token   = os.getenv("HF_TOKEN")

    log.info("Carregando tokenizador: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        use_fast=True,
        padding_side="right",  # causal LM: padding à direita
    )
    # Llama-3 não define pad_token por padrão — usamos eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info("Carregando modelo com quantização NF4 4-bits: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        token=hf_token,
        torch_dtype=torch.bfloat16,  # dtype para as camadas não-quantizadas (norm, embed)
        attn_implementation="eager",  # "flash_attention_2" se flash-attn estiver instalado
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("Parâmetros base — total: %d M | treináveis antes do LoRA: %d", total // 1_000_000, trainable)

    return model, tokenizer


# ── Aplicação dos adaptadores LoRA ────────────────────────────────────────────

def apply_lora(
    model: PreTrainedModel,
    lora_cfg: LoRAAdapterConfig | None = None,
) -> PreTrainedModel:
    """
    Prepara o modelo para treinamento k-bit e injeta adaptadores LoRA.

    Etapas:
        1. `prepare_model_for_kbit_training` ativa gradient checkpointing e
           converte as LayerNorms para float32 (necessário para estabilidade
           dos gradientes com pesos quantizados).
        2. `LoraConfig` define os hiperparâmetros dos adaptadores.
        3. `get_peft_model` congela os pesos base e adiciona os módulos LoRA
           treináveis nas projeções de atenção e FFN especificadas.

    Args:
        model:    Modelo carregado via `load_quantized_model`.
        lora_cfg: Hiperparâmetros LoRA (usa LoRAAdapterConfig() se None).

    Returns:
        PeftModel com somente os adaptadores LoRA marcados como treináveis.
    """
    if lora_cfg is None:
        lora_cfg = LoRAAdapterConfig()

    # Habilita gradient checkpointing e ajusta os dtypes das LayerNorms
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = lora_cfg.r,
        lora_alpha      = lora_cfg.lora_alpha,
        lora_dropout    = lora_cfg.lora_dropout,
        bias            = lora_cfg.bias,
        use_rslora      = lora_cfg.use_rslora,
        target_modules  = lora_cfg.target_modules,
        # Salva os embeddings de token ao exportar — necessário para
        # modelos com vocabulário expandido (ex.: tokens especiais adicionados)
        modules_to_save = ["embed_tokens", "lm_head"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parâmetros LoRA treináveis: %d (%.4f%% do total)",
             trainable,
             100 * trainable / sum(p.numel() for p in model.parameters()))

    return model


# ── Dataset e tokenização ─────────────────────────────────────────────────────

class ClinicalICD10Dataset(Dataset):
    """
    Dataset PyTorch para fine-tuning de extração de CID-10.

    Cada exemplo é formatado como um prompt instrução→resposta no estilo
    Alpaca e tokenizado com truncagem/padding para `max_length` tokens.
    A loss é calculada **apenas sobre os tokens da resposta** (tokens de
    instrução recebem label = -100 para serem ignorados no cross-entropy).
    """

    # Delimitador que separa instrução de resposta no prompt
    RESPONSE_SEPARATOR = "### Resposta:\n"

    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.items      = [self._tokenize(ex) for ex in examples]
        log.info("Dataset criado: %d exemplos, max_length=%d", len(self.items), max_length)

    def _tokenize(self, example: TrainingExample) -> dict[str, torch.Tensor]:
        full_prompt = example.to_prompt()

        # Tokeniza o prompt completo
        full_enc = self.tokenizer(
            full_prompt,
            max_length     = self.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )

        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Descobre onde começa a resposta para mascarar o prefixo de instrução
        prefix = full_prompt.split(self.RESPONSE_SEPARATOR)[0] + self.RESPONSE_SEPARATOR
        prefix_ids = self.tokenizer(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        prefix_len = min(len(prefix_ids), self.max_length)

        # Labels: -100 na parte de instrução, ids reais na parte de resposta
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        # Também ignora o padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def build_dataset(
    examples: list[TrainingExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
) -> ClinicalICD10Dataset:
    """Constrói o ClinicalICD10Dataset a partir dos exemplos do fhir_consumer."""
    return ClinicalICD10Dataset(examples, tokenizer, max_length)


# ── Loop de treinamento PyTorch ───────────────────────────────────────────────

def train_one_round(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[TrainingExample],
    train_cfg: TrainingConfig | None = None,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """
    Executa um round de treinamento federado sobre os exemplos FHIR locais.

    Retorna apenas os **pesos dos adaptadores LoRA** (não os pesos base
    quantizados), que o cliente Flower enviará ao servidor para agregação.

    Args:
        model:      PeftModel retornado por `apply_lora`.
        tokenizer:  Tokenizador correspondente ao modelo.
        examples:   Exemplos de treinamento de `fetch_training_examples`.
        train_cfg:  Hiperparâmetros de treinamento (usa TrainingConfig() se None).
        max_length: Comprimento máximo de tokenização.

    Returns:
        Tupla `(lora_parameters, num_examples, metrics)` no formato esperado
        pelo `NumPyClient.fit()` do Flower:
          - lora_parameters: lista de arrays NumPy com os pesos LoRA atualizados.
          - num_examples:    quantidade de exemplos usados no treinamento.
          - metrics:         dicionário com "train_loss" e "train_perplexity".
    """
    if not examples:
        log.warning("Nenhum exemplo para treinamento — round pulado.")
        return get_lora_parameters(model), 0, {"train_loss": 0.0, "train_perplexity": 1.0}

    if train_cfg is None:
        train_cfg = TrainingConfig()

    dataset    = build_dataset(examples, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size  = train_cfg.batch_size,
        shuffle     = True,
        drop_last   = False,
        pin_memory  = torch.cuda.is_available(),
    )

    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = train_cfg.learning_rate,
        weight_decay = train_cfg.weight_decay,
        betas        = (0.9, 0.95),
        eps          = 1e-8,
    )

    total_steps = len(dataloader) * train_cfg.num_epochs // train_cfg.gradient_accum_steps
    warmup_steps = max(1, int(total_steps * train_cfg.warmup_ratio))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))

    # AMP: bfloat16 em Ampere+; float16 como fallback
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler    = torch.amp.GradScaler("cuda", enabled=train_cfg.use_amp and amp_dtype == torch.float16)

    model.train()
    cumulative_loss  = 0.0
    total_tokens     = 0
    global_step      = 0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            # Move batch para o device do modelo
            device = next(model.parameters()).device
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=train_cfg.use_amp):
                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels,
                )
                # Normaliza a loss pelo número de tokens reais (não -100) para
                # que o gradient accumulation produza o mesmo resultado independente
                # do tamanho do batch
                num_active = (labels != -100).sum().item()
                loss = outputs.loss * (labels.shape[-1] / max(num_active, 1))
                loss_scaled = loss / train_cfg.gradient_accum_steps

            scaler.scale(loss_scaled).backward()

            epoch_loss  += loss.item()
            total_tokens += max(num_active, 1)

            # Atualização dos pesos a cada `gradient_accum_steps` mini-batches
            if (step + 1) % train_cfg.gradient_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    train_cfg.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if global_step >= warmup_steps:
                    scheduler.step()
                global_step += 1

                if global_step % 10 == 0:
                    log.info(
                        "Epoch %d/%d | step %d | loss=%.4f",
                        epoch + 1, train_cfg.num_epochs, global_step, epoch_loss / (step + 1),
                    )

        avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
        log.info("Epoch %d/%d concluída — loss média: %.4f", epoch + 1, train_cfg.num_epochs, avg_epoch_loss)
        cumulative_loss += avg_epoch_loss

    avg_loss    = cumulative_loss / max(train_cfg.num_epochs, 1)
    perplexity  = float(torch.exp(torch.tensor(avg_loss)).item())

    metrics = {
        "train_loss":       round(avg_loss, 6),
        "train_perplexity": round(perplexity, 4),
    }
    log.info("Round concluído — loss=%.4f | perplexity=%.2f", avg_loss, perplexity)

    return get_lora_parameters(model), len(examples), metrics


# ── Utilitários de parâmetros para Flower ─────────────────────────────────────

def get_lora_parameters(model: PreTrainedModel) -> list[np.ndarray]:
    """
    Extrai apenas os pesos dos adaptadores LoRA como lista de arrays NumPy.

    Esta é a representação trocada entre cliente e servidor Flower durante
    a agregação federada (FedAvg ou similar). Os pesos base quantizados
    **não** são incluídos — apenas os deltas LoRA.

    Returns:
        Lista ordenada de arrays NumPy correspondendo a cada tensor LoRA.
    """
    lora_state = get_peft_model_state_dict(model)
    return [v.detach().float().cpu().numpy() for v in lora_state.values()]


def set_lora_parameters(model: PreTrainedModel, parameters: list[np.ndarray]) -> None:
    """
    Aplica pesos LoRA agregados (recebidos do servidor Flower) ao modelo local.

    Chamado em `NumPyClient.configure_fit()` ou `NumPyClient.evaluate()` antes
    de qualquer inferência ou round de treinamento subsequente.

    Args:
        model:      PeftModel com adaptadores LoRA.
        parameters: Lista de arrays NumPy no mesmo formato de `get_lora_parameters`.
    """
    lora_state = get_peft_model_state_dict(model)
    keys       = list(lora_state.keys())

    if len(keys) != len(parameters):
        raise ValueError(
            f"Incompatibilidade de parâmetros: modelo tem {len(keys)} tensores LoRA, "
            f"mas recebeu {len(parameters)}."
        )

    new_state = {
        k: torch.tensor(v, dtype=lora_state[k].dtype)
        for k, v in zip(keys, parameters)
    }
    # set_peft_model_state_dict requer o modelo unwrapped e o state dict
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, new_state)
    log.info("Pesos LoRA atualizados com parâmetros agregados do servidor (%d tensores).", len(keys))
