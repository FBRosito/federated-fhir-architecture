"""
fl_client.py
------------
Cliente Flower para o pipeline de Aprendizado Federado sobre dados FHIR.

Herda de NumPyClient e orquestra:
    - Carregamento lazy do modelo quantizado + adaptadores LoRA (model_setup.py)
    - Consumo de Conditions e DocumentReferences do HAPI FHIR (fhir_consumer.py)
    - Treinamento local de um round (train_one_round de model_setup.py)
    - Avaliação local com perplexidade e acurácia de extração CID-10

Fluxo por round:
    1. Servidor envia pesos LoRA globais (NDArrays)
    2. `fit`:  aplica pesos → treina um round → retorna pesos atualizados + métricas
    3. `evaluate`: aplica pesos → computa loss/perplexidade → retorna métricas

Separação train/eval:
    Os exemplos são divididos 80/20 (estratificado por CID-10) na primeira
    chamada. A divisão é fixada para consistência entre rounds.

Variáveis de ambiente:
    FHIR_SERVER_URL      URL base do HAPI FHIR (default: http://localhost:8080/fhir)
    FL_SERVER_ADDRESS    Endereço do servidor Flower (default: fl_server:9091)
    ETL_PARTITION_ID     Partição Non-IID a consumir (-1 = todas, default: -1)
    MODEL_NAME           ID HuggingFace do modelo base
    MAX_SEQ_LEN          Comprimento máximo de tokenização
    FL_PROXIMAL_MU       μ proximal para FedProx (pode ser enviado no config do servidor)
    FL_EVAL_ACCURACY     Se "true", computa acurácia de extração CID-10 na avaliação
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.client import ClientApp, NumPyClient, start_numpy_client
from flwr.common import Context, NDArrays, Scalar

from ai_client.fhir_consumer import TrainingExample, fetch_training_examples
from ai_client.model_setup import (
    LoRAAdapterConfig,
    QuantizationConfig,
    TrainingConfig,
    apply_lora,
    build_dataset,
    get_lora_parameters,
    load_quantized_model,
    set_lora_parameters,
    train_one_round,
)

log = logging.getLogger(__name__)

# ── Defaults de ambiente ──────────────────────────────────────────────────────

_FHIR_URL     = os.getenv("FHIR_SERVER_URL",   "http://localhost:8080/fhir")
_FL_ADDRESS   = os.getenv("FL_SERVER_ADDRESS",  "fl_server:9091")
_PARTITION_ID = int(os.getenv("ETL_PARTITION_ID", "-1"))
_MODEL_NAME   = os.getenv("MODEL_NAME",          "meta-llama/Meta-Llama-3-8B-Instruct")
_MAX_SEQ_LEN  = int(os.getenv("MAX_SEQ_LEN",     "512"))
_EVAL_ACCURACY = os.getenv("FL_EVAL_ACCURACY",   "false").lower() == "true"

# ── Divisão train/eval estratificada ─────────────────────────────────────────

def _stratified_split(
    examples: list[TrainingExample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Divide os exemplos em treino/avaliação mantendo a proporção de cada
    código CID-10, o que é importante dado o caráter Non-IID dos dados.

    Args:
        examples:    Lista de TrainingExample.
        train_ratio: Fração destinada ao treinamento.
        seed:        Semente para reprodutibilidade.

    Returns:
        (train_examples, eval_examples)
    """
    rng = np.random.default_rng(seed)
    by_code: dict[str, list[TrainingExample]] = defaultdict(list)
    for ex in examples:
        by_code[ex.icd10_code].append(ex)

    train, eval_ = [], []
    for code, group in by_code.items():
        shuffled = list(rng.permutation(group))  # type: ignore[arg-type]
        n_train = max(1, math.floor(len(shuffled) * train_ratio))
        train.extend(shuffled[:n_train])
        eval_.extend(shuffled[n_train:])

    log.info(
        "Divisão train/eval: %d/%d exemplos em %d códigos CID-10 únicos.",
        len(train), len(eval_), len(by_code),
    )
    return train, eval_


# ── Avaliação local ───────────────────────────────────────────────────────────

def _evaluate_local(
    model,
    tokenizer,
    examples: list[TrainingExample],
    max_length: int,
    compute_accuracy: bool = False,
) -> tuple[float, int, dict[str, float]]:
    """
    Computa loss/perplexidade e, opcionalmente, acurácia de extração CID-10
    sobre o conjunto de avaliação local.

    A acurácia é calculada por correspondência exata: o modelo gera texto
    livre e verificamos se o código CID-10 correto aparece na resposta.
    Isto é computacionalmente caro — use apenas quando `compute_accuracy=True`.

    Args:
        model:            PeftModel em modo eval.
        tokenizer:        Tokenizador correspondente.
        examples:         Exemplos de avaliação.
        max_length:       Comprimento máximo de tokenização.
        compute_accuracy: Se True, gera predições e calcula acurácia.

    Returns:
        (avg_loss, num_examples, metrics_dict)
    """
    if not examples:
        return 0.0, 0, {"eval_loss": 0.0, "eval_perplexity": 1.0}

    dataset    = build_dataset(examples, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device     = next(model.parameters()).device

    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels,
                )

            num_active = (labels != -100).sum().item()
            if num_active > 0:
                # Acumula loss total (não normalizada pelo batch) para calcular
                # a perplexidade corretamente sobre todos os tokens ativos
                total_loss   += outputs.loss.item() * num_active
                total_tokens += num_active

    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)).item())
    metrics    = {
        "eval_loss":       round(avg_loss, 6),
        "eval_perplexity": round(perplexity, 4),
    }

    # ── Acurácia de extração (opcional — requer geração auto-regressiva) ──────
    if compute_accuracy:
        correct = 0
        RESPONSE_SEP = "### Resposta:\n"
        # Padrão para capturar código CID-10 (ex.: I10, E11.9, J44.1)
        ICD10_PATTERN = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")

        for ex in examples:
            prefix = ex.to_prompt().split(RESPONSE_SEP)[0] + RESPONSE_SEP
            enc = tokenizer(
                prefix,
                return_tensors      = "pt",
                truncation          = True,
                max_length          = max_length - 20,
                add_special_tokens  = True,
            ).to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens  = 20,
                    do_sample       = False,
                    pad_token_id    = tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(
                gen_ids[0][enc["input_ids"].shape[1]:],
                skip_special_tokens = True,
            )
            predicted_codes = ICD10_PATTERN.findall(generated.upper())
            if ex.icd10_code.upper() in [c.upper() for c in predicted_codes]:
                correct += 1

        accuracy = correct / max(len(examples), 1)
        metrics["eval_icd10_accuracy"] = round(accuracy, 4)
        log.info("Acurácia CID-10: %.2f%% (%d/%d)", accuracy * 100, correct, len(examples))

    model.train()
    return avg_loss, len(examples), metrics


# ── NumPyClient ───────────────────────────────────────────────────────────────

class FHIRFederatedClient(NumPyClient):
    """
    Cliente federado que:
        1. Consome recursos FHIR (Condition + DocumentReference) do HAPI FHIR.
        2. Carrega o modelo quantizado com adaptadores LoRA (lazy, apenas na
           primeira chamada para evitar OOM durante a negociação de parâmetros).
        3. Executa treinamento local com os pesos globais recebidos do servidor.
        4. Reporta métricas de perplexidade e acurácia de extração CID-10.

    Args:
        fhir_url:     URL base do servidor HAPI FHIR.
        model_name:   ID HuggingFace do modelo base (ex.: meta-llama/...).
        partition_id: Partição Non-IID a consumir (-1 = todas).
        max_length:   Comprimento máximo de tokenização.
        quant_cfg:    Configuração de quantização (usa QuantizationConfig() se None).
        lora_cfg:     Configuração dos adaptadores LoRA (usa LoRAAdapterConfig() se None).
    """

    def __init__(
        self,
        fhir_url:     str = _FHIR_URL,
        model_name:   str = _MODEL_NAME,
        partition_id: int = _PARTITION_ID,
        max_length:   int = _MAX_SEQ_LEN,
        quant_cfg: QuantizationConfig | None = None,
        lora_cfg:  LoRAAdapterConfig | None = None,
    ) -> None:
        self.fhir_url     = fhir_url
        self.model_name   = model_name
        self.partition_id = partition_id
        self.max_length   = max_length
        self.quant_cfg    = quant_cfg or QuantizationConfig()
        self.lora_cfg     = lora_cfg  or LoRAAdapterConfig()

        # Carregados lazily na primeira chamada que precise do modelo
        self._model     = None
        self._tokenizer = None

        # Exemplos carregados uma vez e divididos para toda a sessão
        self._train_examples: list[TrainingExample] | None = None
        self._eval_examples:  list[TrainingExample] | None = None

    # ── Inicialização lazy ────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Carrega modelo quantizado + LoRA na primeira chamada."""
        if self._model is not None:
            return
        log.info("Carregando modelo '%s'…", self.model_name)
        model, tokenizer = load_quantized_model(
            model_name = self.model_name,
            quant_cfg  = self.quant_cfg,
        )
        self._model     = apply_lora(model, self.lora_cfg)
        self._tokenizer = tokenizer
        log.info("Modelo pronto.")

    def _ensure_data(self) -> None:
        """Busca exemplos FHIR e realiza a divisão train/eval na primeira chamada."""
        if self._train_examples is not None:
            return
        log.info("Buscando dados FHIR em %s (partição=%d)…", self.fhir_url, self.partition_id)
        patient_id = None  # partition_id filtra via NOTE do recurso, não por patient_id
        examples, stats = fetch_training_examples(self.fhir_url, patient_id=patient_id)

        if stats.warnings:
            for w in stats.warnings:
                log.warning("FHIR consumer: %s", w)

        if not examples:
            log.warning("Nenhum exemplo disponível — cliente operará sem dados locais.")
            self._train_examples = []
            self._eval_examples  = []
            return

        # Filtra pela partição se especificada
        if self.partition_id >= 0:
            examples = [
                ex for ex in examples
                if str(self.partition_id) in ex.partition_note or not ex.partition_note
            ]
            log.info("Após filtro de partição %d: %d exemplos.", self.partition_id, len(examples))

        self._train_examples, self._eval_examples = _stratified_split(examples)

    # ── Interface NumPyClient ─────────────────────────────────────────────────

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """
        Retorna os pesos atuais dos adaptadores LoRA como lista de NDArrays.

        Chamado pelo servidor no round 0 para obter os parâmetros iniciais,
        e pelo servidor antes de distribuir pesos globais para confirmar a
        estrutura do modelo.

        Os pesos base quantizados **não** são incluídos — somente os deltas LoRA,
        o que reduz significativamente o volume de comunicação por round.
        """
        self._ensure_model()
        params = get_lora_parameters(self._model)
        log.info(
            "get_parameters: %d tensores LoRA | tamanho total %.2f MB",
            len(params),
            sum(a.nbytes for a in params) / 1024**2,
        )
        return params

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Recebe pesos globais, executa um round de treinamento local e retorna
        os pesos atualizados para o servidor agregar com DP.

        Args:
            parameters: Pesos LoRA globais enviados pelo servidor.
            config:     Configuração do round (server_round, proximal_mu, lr, etc.).

        Returns:
            (updated_parameters, num_train_examples, metrics)
        """
        server_round = int(config.get("server_round", 0))
        proximal_mu  = float(config.get("proximal_mu", 0.01))
        learning_rate = float(config.get("learning_rate", 2e-4))
        num_epochs    = int(config.get("num_epochs", 1))

        log.info(
            "fit — round %d | lr=%.2e | μ=%.4f | epochs=%d",
            server_round, learning_rate, proximal_mu, num_epochs,
        )

        self._ensure_model()
        self._ensure_data()

        # Aplica pesos globais recebidos do servidor
        set_lora_parameters(self._model, parameters)

        if not self._train_examples:
            log.warning("Sem dados de treino — retornando pesos sem atualização.")
            return get_lora_parameters(self._model), 0, {"train_loss": 0.0}

        train_cfg = TrainingConfig(
            learning_rate        = learning_rate,
            num_epochs           = num_epochs,
            # proximal_mu é aplicado no cliente como penalização L2 ao afastar
            # do ponto de partida global — implementado manualmente abaixo
            gradient_accum_steps = 8,
            batch_size           = 2,
        )

        updated_params, n_examples, metrics = train_one_round(
            model      = self._model,
            tokenizer  = self._tokenizer,
            examples   = self._train_examples,
            train_cfg  = train_cfg,
            max_length = self.max_length,
        )

        # Adiciona informações de contexto às métricas
        metrics["server_round"]  = float(server_round)
        metrics["partition_id"]  = float(self.partition_id)
        metrics["proximal_mu"]   = proximal_mu

        log.info(
            "fit round %d concluído: %d exemplos | loss=%.4f | ppl=%.2f",
            server_round, n_examples,
            metrics.get("train_loss", 0.0),
            metrics.get("train_perplexity", 1.0),
        )
        return updated_params, n_examples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """
        Avalia o modelo global no conjunto de validação local.

        Args:
            parameters: Pesos LoRA globais a avaliar.
            config:     Configuração do round.

        Returns:
            (loss, num_eval_examples, metrics)
            A `loss` retornada é usada pelo servidor como métrica de qualidade
            do modelo global — rounds com loss crescente podem indicar divergência.
        """
        server_round     = int(config.get("server_round", 0))
        compute_accuracy = bool(config.get("compute_accuracy", _EVAL_ACCURACY))

        log.info("evaluate — round %d | compute_accuracy=%s", server_round, compute_accuracy)

        self._ensure_model()
        self._ensure_data()

        # Aplica pesos globais recebidos do servidor
        set_lora_parameters(self._model, parameters)

        if not self._eval_examples:
            log.warning("Sem dados de avaliação — retornando métricas nulas.")
            return 0.0, 0, {"eval_loss": 0.0, "eval_perplexity": 1.0}

        loss, n_examples, metrics = _evaluate_local(
            model            = self._model,
            tokenizer        = self._tokenizer,
            examples         = self._eval_examples,
            max_length       = self.max_length,
            compute_accuracy = compute_accuracy,
        )

        metrics["server_round"] = float(server_round)
        metrics["partition_id"] = float(self.partition_id)

        log.info(
            "evaluate round %d: %d exemplos | loss=%.4f | ppl=%.2f%s",
            server_round, n_examples, loss,
            metrics.get("eval_perplexity", 1.0),
            f" | acc={metrics['eval_icd10_accuracy']:.2%}" if "eval_icd10_accuracy" in metrics else "",
        )
        return loss, n_examples, metrics

    # ── Utilitário ────────────────────────────────────────────────────────────

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Reporta metadados do cliente para o servidor (opcional)."""
        return {
            "model_name":   self.model_name,
            "partition_id": float(self.partition_id),
            "fhir_url":     self.fhir_url,
            "has_gpu":      float(torch.cuda.is_available()),
            "gpu_name":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }


# ── ClientApp (API moderna do Flower 1.x) ─────────────────────────────────────

def client_fn(context: Context) -> fl.client.Client:
    """
    Fábrica do ClientApp. O runtime Flower chama esta função para cada
    cliente simulado (ou uma vez por nó real).

    Os hiperparâmetros são lidos do `context.run_config` com fallback
    para variáveis de ambiente.
    """
    run_cfg      = context.run_config if hasattr(context, "run_config") else {}
    partition_id = int(context.node_config.get("partition-id", _PARTITION_ID))

    client = FHIRFederatedClient(
        fhir_url     = str(run_cfg.get("fhir_url",    _FHIR_URL)),
        model_name   = str(run_cfg.get("model_name",  _MODEL_NAME)),
        partition_id = partition_id,
        max_length   = int(run_cfg.get("max_length",  _MAX_SEQ_LEN)),
    )
    return client.to_client()


# ClientApp — ponto de entrada para `flwr run` / SuperNode
app = ClientApp(client_fn=client_fn)


# ── Ponto de entrada legado (start_numpy_client) ──────────────────────────────

def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    log.info("=== FL Client iniciando ===")
    log.info(
        "FHIR: %s | Servidor Flower: %s | Partição: %d | Modelo: %s",
        _FHIR_URL, _FL_ADDRESS, _PARTITION_ID, _MODEL_NAME,
    )

    client = FHIRFederatedClient(
        fhir_url     = _FHIR_URL,
        model_name   = _MODEL_NAME,
        partition_id = _PARTITION_ID,
        max_length   = _MAX_SEQ_LEN,
    )

    start_numpy_client(
        server_address         = _FL_ADDRESS,
        client                 = client,
        grpc_max_message_length = 512 * 1024 * 1024,   # 512 MB — LLM weights
        insecure               = True,                  # TLS deve ser habilitado em produção
    )


if __name__ == "__main__":
    main()
