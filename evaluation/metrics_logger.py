"""
metrics_logger.py
-----------------
Módulo de avaliação e registro do pipeline de Aprendizado Federado sobre dados FHIR.

Responsabilidades:
    1. Métricas de classificação CID-10 via scikit-learn:
       precisão, revocação e F1-score micro, macro e ponderado por classe;
       acurácia top-1; relatório por código CID-10; erros mais frequentes.

    2. Logger operacional para CSV:
       - Rounds de comunicação federada e volume de dados transmitidos
       - Tempo de processamento local (GPU via torch.cuda.Event ou CPU via perf_counter)
       - Pico de memória GPU, utilização de CPU e RAM
       - Métricas de treino/avaliação recebidas do fl_client

    3. Utilitários de análise pós-treinamento:
       curva de convergência por round, comparação entre partições Non-IID.

Uso:
    # Registro por round durante execução federada
    logger = FederatedRunLogger(csv_path="evaluation/logs/run.csv", client_id="node-0")
    with GPUTimer() as t:
        params, n, metrics = client.fit(global_params, config)
    tracker.record_sent(global_params)
    tracker.record_received(params)
    logger.log_round(
        round_number     = 1,
        y_true           = ground_truth_codes,
        y_pred           = predicted_codes,
        gpu_timer        = t,
        data_tracker     = tracker,
        training_metrics = metrics,
    )

    # Análise pós-treino
    df = logger.load_history()
    print(logger.summary())

Variáveis de ambiente:
    FL_LOG_DIR       Diretório de saída dos CSVs (default: evaluation/logs)
    FL_LOG_CLIENT_ID Identificador do cliente/nó (default: "client-0")
    FL_PARTITION_ID  Partição Non-IID deste cliente (default: -1)
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_LOG_DIR    = Path(os.getenv("FL_LOG_DIR",   "evaluation/logs"))
_CLIENT_ID  = os.getenv("FL_LOG_CLIENT_ID",  "client-0")
_PARTITION  = int(os.getenv("FL_PARTITION_ID", "-1"))

# ── Estruturas de dados ───────────────────────────────────────────────────────

@dataclass
class MetricsReport:
    """
    Resultado completo de uma rodada de avaliação de classificação CID-10.

    Todos os campos de média são calculados pelo scikit-learn com
    `zero_division=0` para evitar warnings em classes sem predição.
    """
    # ── Micro (trata cada amostra igualmente — afetado por classes frequentes)
    precision_micro:    float = 0.0
    recall_micro:       float = 0.0
    f1_micro:           float = 0.0

    # ── Macro (média simples por classe — penaliza desbalanceamento)
    precision_macro:    float = 0.0
    recall_macro:       float = 0.0
    f1_macro:           float = 0.0

    # ── Weighted (média por classe ponderada pelo suporte)
    precision_weighted: float = 0.0
    recall_weighted:    float = 0.0
    f1_weighted:        float = 0.0

    # ── Amostras
    accuracy:           float = 0.0
    n_samples:          int   = 0
    n_correct:          int   = 0
    n_classes_true:     int   = 0   # classes únicas em y_true
    n_classes_pred:     int   = 0   # classes únicas em y_pred

    # ── Por classe (dict {icd10_code: {precision, recall, f1, support}})
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    # ── Erros mais frequentes (lista de (true, pred, count))
    top_errors: list[tuple[str, str, int]] = field(default_factory=list)


@dataclass
class OperationalRecord:
    """
    Uma linha do CSV operacional — captura o estado de um round federado.
    Os campos seguem a ordem das colunas no arquivo CSV.
    """
    # ── Contexto temporal e de identificação
    timestamp:              str   = ""
    round_number:           int   = 0
    client_id:              str   = ""
    partition_id:           int   = -1
    phase:                  str   = "fit"   # "fit" | "evaluate"

    # ── Métricas de classificação CID-10
    accuracy:               float = 0.0
    precision_micro:        float = 0.0
    recall_micro:           float = 0.0
    f1_micro:               float = 0.0
    precision_macro:        float = 0.0
    recall_macro:           float = 0.0
    f1_macro:               float = 0.0
    precision_weighted:     float = 0.0
    recall_weighted:        float = 0.0
    f1_weighted:            float = 0.0
    n_samples:              int   = 0
    n_correct:              int   = 0
    n_classes_true:         int   = 0
    n_classes_pred:         int   = 0

    # ── Volume de comunicação federada
    total_comm_rounds:      int   = 0     # rounds acumulados na sessão
    params_sent_mb:         float = 0.0   # pesos LoRA enviados neste round (MB)
    params_received_mb:     float = 0.0   # pesos LoRA recebidos neste round (MB)
    cumulative_data_mb:     float = 0.0   # total de dados na sessão (MB)

    # ── Tempo e recursos de processamento
    gpu_processing_time_s:  float = 0.0   # tempo do round (GPU Event ou perf_counter)
    gpu_memory_peak_mb:     float = 0.0   # pico de VRAM alocada no round (MB)
    gpu_memory_current_mb:  float = 0.0   # VRAM alocada ao fim do round (MB)
    gpu_utilization_pct:    float = 0.0   # % de utilização GPU (nvidia-smi, best-effort)
    cpu_usage_pct:          float = 0.0   # % CPU no momento do log
    ram_usage_gb:           float = 0.0   # RAM usada pelo processo (GB)

    # ── Métricas de treino / avaliação do modelo (repassadas pelo fl_client)
    train_loss:             float = 0.0
    train_perplexity:       float = 0.0
    eval_loss:              float = 0.0
    eval_perplexity:        float = 0.0

    # ── Ambiente
    has_gpu:                bool  = False
    gpu_name:               str   = "none"


# ── GPUTimer ──────────────────────────────────────────────────────────────────

class GPUTimer:
    """
    Cronômetro de precisão para operações GPU/CPU.

    Em GPU: usa `torch.cuda.Event` com sincronização explícita — mede o tempo
    real de execução no device, incluindo kernel launches que são assíncronos
    por padrão. Evita o viés de `time.perf_counter` que retorna antes que os
    kernels CUDA terminem.

    Em CPU: usa `time.perf_counter` como fallback.

    Uso como context manager:
        with GPUTimer() as t:
            model(input)
        print(t.elapsed_s)   # segundos
        print(t.elapsed_ms)  # milissegundos

    Uso manual:
        t = GPUTimer()
        t.start()
        ...
        t.stop()
        print(t.elapsed_s)
    """

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self.elapsed_s:  float = 0.0
        self._use_cuda = torch.cuda.is_available()
        self._start_event = None
        self._end_event   = None
        self._cpu_start:  float = 0.0

    def start(self) -> "GPUTimer":
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event   = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def stop(self) -> "GPUTimer":
        if self._use_cuda and self._start_event is not None:
            self._end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._cpu_start) * 1_000
        self.elapsed_s = self.elapsed_ms / 1_000
        return self

    def __enter__(self) -> "GPUTimer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    @property
    def peak_memory_mb(self) -> float:
        """Pico de memória GPU alocada durante o intervalo cronometrado (MB)."""
        if self._use_cuda:
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    @property
    def current_memory_mb(self) -> float:
        """Memória GPU atualmente alocada (MB)."""
        if self._use_cuda:
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0


# ── DataVolumeTracker ─────────────────────────────────────────────────────────

class DataVolumeTracker:
    """
    Rastreia o volume de dados (em bytes/MB) transmitidos durante uma sessão
    federada — enviados ao servidor e recebidos dele.

    Os pesos LoRA são representados como listas de NDArrays (float32);
    o volume é calculado como a soma dos `nbytes` de cada array.

    Thread-safe: múltiplos clientes podem registrar simultaneamente em
    simulações com múltiplos nós.
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._sent_bytes:   int = 0
        self._recv_bytes:   int = 0
        self._round_sent:   int = 0   # bytes enviados no round atual
        self._round_recv:   int = 0   # bytes recebidos no round atual

    def record_sent(self, parameters: list[np.ndarray]) -> float:
        """
        Registra os pesos LoRA enviados ao servidor neste round.

        Args:
            parameters: Lista de NDArrays (saída de `get_lora_parameters`).

        Returns:
            Volume enviado neste round em MB.
        """
        nbytes = sum(a.nbytes for a in parameters)
        with self._lock:
            self._sent_bytes  += nbytes
            self._round_sent   = nbytes
        mb = nbytes / 1024**2
        log.debug("Enviado: %.3f MB (%d tensores)", mb, len(parameters))
        return mb

    def record_received(self, parameters: list[np.ndarray]) -> float:
        """
        Registra os pesos LoRA globais recebidos do servidor neste round.

        Args:
            parameters: Lista de NDArrays (parâmetros globais pós-agregação).

        Returns:
            Volume recebido neste round em MB.
        """
        nbytes = sum(a.nbytes for a in parameters)
        with self._lock:
            self._recv_bytes  += nbytes
            self._round_recv   = nbytes
        mb = nbytes / 1024**2
        log.debug("Recebido: %.3f MB (%d tensores)", mb, len(parameters))
        return mb

    @property
    def round_sent_mb(self) -> float:
        return self._round_sent / 1024**2

    @property
    def round_received_mb(self) -> float:
        return self._round_recv / 1024**2

    @property
    def cumulative_mb(self) -> float:
        return (self._sent_bytes + self._recv_bytes) / 1024**2

    @property
    def total_sent_mb(self) -> float:
        return self._sent_bytes / 1024**2

    @property
    def total_received_mb(self) -> float:
        return self._recv_bytes / 1024**2

    def reset_round(self) -> None:
        """Zera os contadores do round atual (mantém acumulados)."""
        with self._lock:
            self._round_sent = 0
            self._round_recv = 0


# ── Funções de métricas de classificação ─────────────────────────────────────

def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    top_k_errors: int = 5,
) -> MetricsReport:
    """
    Calcula métricas de classificação CID-10 usando scikit-learn.

    Args:
        y_true:       Rótulos verdadeiros (códigos CID-10, ex.: ["I10", "J44.1"]).
        y_pred:       Predições do modelo (mesma estrutura).
        labels:       Conjunto fixo de classes. Se None, inferido dos dados.
        top_k_errors: Quantos pares (true, pred) errados mais frequentes reportar.

    Returns:
        MetricsReport com todas as métricas calculadas.

    Notes:
        - `zero_division=0`: evita warnings em classes sem predições — retorna
          0 nesses casos, o que é conservador mas não mascara o problema.
        - Micro-F1 é idêntico à acurácia em tarefas single-label (cada amostra
          pertence a exatamente um código CID-10).
    """
    if not y_true:
        log.warning("compute_classification_metrics chamado com y_true vazio.")
        return MetricsReport()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true e y_pred têm tamanhos diferentes: {len(y_true)} vs {len(y_pred)}"
        )

    report = MetricsReport()
    report.n_samples      = len(y_true)
    report.n_correct      = int(accuracy_score(y_true, y_pred, normalize=False))
    report.accuracy       = round(float(accuracy_score(y_true, y_pred)), 6)
    report.n_classes_true = len(set(y_true))
    report.n_classes_pred = len(set(y_pred))

    # ── Micro ─────────────────────────────────────────────────────────────────
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", labels=labels, zero_division=0
    )
    report.precision_micro = round(float(p), 6)
    report.recall_micro    = round(float(r), 6)
    report.f1_micro        = round(float(f), 6)

    # ── Macro ─────────────────────────────────────────────────────────────────
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    report.precision_macro = round(float(p), 6)
    report.recall_macro    = round(float(r), 6)
    report.f1_macro        = round(float(f), 6)

    # ── Weighted ──────────────────────────────────────────────────────────────
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    report.precision_weighted = round(float(p), 6)
    report.recall_weighted    = round(float(r), 6)
    report.f1_weighted        = round(float(f), 6)

    # ── Por classe ────────────────────────────────────────────────────────────
    full_report: dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    # Remove as entradas de média agregada — mantém apenas por código CID-10
    skip_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    report.per_class = {
        code: {
            "precision": round(float(vals["precision"]), 6),
            "recall":    round(float(vals["recall"]),    6),
            "f1":        round(float(vals["f1-score"]),  6),
            "support":   int(vals["support"]),
        }
        for code, vals in full_report.items()
        if code not in skip_keys
    }

    # ── Erros mais frequentes ─────────────────────────────────────────────────
    error_counts: dict[tuple[str, str], int] = {}
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            error_counts[(yt, yp)] = error_counts.get((yt, yp), 0) + 1
    report.top_errors = [
        (yt, yp, cnt)
        for (yt, yp), cnt in sorted(error_counts.items(), key=lambda x: -x[1])[:top_k_errors]
    ]

    return report


def print_classification_report(report: MetricsReport, title: str = "Avaliação CID-10") -> None:
    """
    Exibe um relatório formatado no terminal com as métricas de classificação.

    Args:
        report: MetricsReport retornado por `compute_classification_metrics`.
        title:  Título da seção exibida.
    """
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Amostras     : {report.n_samples:>6}  |  Corretas: {report.n_correct:>6}")
    print(f"  Acurácia     : {report.accuracy:>8.4f}")
    print(f"  Classes (GT) : {report.n_classes_true:>6}  |  Classes (pred): {report.n_classes_pred}")
    print(sep)
    print(f"  {'Métrica':<22} {'Micro':>8}  {'Macro':>8}  {'Weighted':>9}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*9}")
    print(f"  {'Precisão':<22} {report.precision_micro:>8.4f}  {report.precision_macro:>8.4f}  {report.precision_weighted:>9.4f}")
    print(f"  {'Revocação':<22} {report.recall_micro:>8.4f}  {report.recall_macro:>8.4f}  {report.recall_weighted:>9.4f}")
    print(f"  {'F1-Score':<22} {report.f1_micro:>8.4f}  {report.f1_macro:>8.4f}  {report.f1_weighted:>9.4f}")

    if report.per_class:
        print(f"\n  {'CID-10':<14} {'Prec':>7}  {'Rev':>7}  {'F1':>7}  {'Suporte':>8}")
        print(f"  {'─'*14} {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
        for code, m in sorted(report.per_class.items(), key=lambda x: -x[1]["support"]):
            print(f"  {code:<14} {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['support']:>8}")

    if report.top_errors:
        print(f"\n  Top-{len(report.top_errors)} erros mais frequentes:")
        print(f"  {'Real':<14} {'Predito':<14} {'Contagem':>8}")
        print(f"  {'─'*14} {'─'*14} {'─'*8}")
        for yt, yp, cnt in report.top_errors:
            print(f"  {yt:<14} {yp:<14} {cnt:>8}")

    print(f"{sep}\n")


# ── Coleta de métricas de sistema ─────────────────────────────────────────────

def _get_gpu_utilization() -> float:
    """
    Consulta a utilização da GPU (%) via nvidia-smi.
    Retorna 0.0 se nvidia-smi não estiver disponível ou falhar.
    Best-effort — não interrompe o fluxo em caso de erro.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if lines:
                return float(lines[0])
    except Exception:
        pass
    return 0.0


def _get_system_stats() -> dict[str, float]:
    """
    Coleta uso de CPU (%) e RAM (GB) do processo atual.
    Sempre disponível — não depende de CUDA.
    """
    proc = psutil.Process()
    return {
        "cpu_usage_pct": round(psutil.cpu_percent(interval=None), 1),
        "ram_usage_gb":  round(proc.memory_info().rss / 1024**3, 3),
    }


# ── FederatedRunLogger ────────────────────────────────────────────────────────

class FederatedRunLogger:
    """
    Logger thread-safe que registra métricas operacionais de cada round
    federado em um arquivo CSV incrementalmente.

    O arquivo é criado com cabeçalho na primeira escrita e aberto em modo
    append nas chamadas seguintes — sobrevive a reinicializações do cliente
    sem perder histórico da sessão atual.

    Args:
        csv_path:    Caminho do arquivo CSV de saída.
        client_id:   Identificador do nó cliente (ex.: "node-cardiology-0").
        partition_id: Partição Non-IID deste cliente.

    Exemplo:
        logger = FederatedRunLogger("evaluation/logs/run.csv", client_id="node-0")
        with GPUTimer() as t:
            params, n, metrics = client.fit(global_params, config)
        logger.log_round(
            round_number=1,
            phase="fit",
            y_true=ground_truth,
            y_pred=predictions,
            gpu_timer=t,
            data_tracker=tracker,
            training_metrics=metrics,
        )
    """

    _FIELDNAMES = list(OperationalRecord.__dataclass_fields__.keys())

    def __init__(
        self,
        csv_path:    Path | str = _LOG_DIR / "federated_run.csv",
        client_id:   str = _CLIENT_ID,
        partition_id: int = _PARTITION,
    ) -> None:
        self.csv_path    = Path(csv_path)
        self.client_id   = client_id
        self.partition_id = partition_id
        self._lock       = threading.Lock()
        self._comm_rounds = 0              # rounds acumulados nesta sessão
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        """Cria o arquivo CSV com cabeçalho se ainda não existir."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writeheader()
            log.info("CSV de métricas criado: %s", self.csv_path)

    def log_round(
        self,
        round_number:      int,
        phase:             str = "fit",
        y_true:            list[str] | None = None,
        y_pred:            list[str] | None = None,
        gpu_timer:         GPUTimer | None = None,
        data_tracker:      DataVolumeTracker | None = None,
        training_metrics:  dict[str, float] | None = None,
        labels:            list[str] | None = None,
    ) -> OperationalRecord:
        """
        Registra uma linha no CSV com as métricas do round.

        Args:
            round_number:     Número do round federado atual.
            phase:            "fit" (treinamento) ou "evaluate" (avaliação).
            y_true:           Rótulos CID-10 verdadeiros (opcional — se None,
                              as métricas de classificação ficam zeradas).
            y_pred:           Predições CID-10 do modelo (deve acompanhar y_true).
            gpu_timer:        GPUTimer já finalizado (.stop() chamado).
            data_tracker:     DataVolumeTracker com volumes do round atual.
            training_metrics: Dict com "train_loss", "train_perplexity",
                              "eval_loss", "eval_perplexity" (vindo do fl_client).
            labels:           Conjunto fixo de classes CID-10 para sklearn.

        Returns:
            OperationalRecord registrado (útil para testes e inspeção).
        """
        self._comm_rounds += 1
        tm = training_metrics or {}
        sys_stats = _get_system_stats()

        # ── Métricas de classificação ─────────────────────────────────────────
        clf = MetricsReport()
        if y_true and y_pred:
            try:
                clf = compute_classification_metrics(y_true, y_pred, labels=labels)
            except Exception as exc:
                log.warning("Erro ao calcular métricas de classificação: %s", exc)

        # ── Cronômetro e memória GPU ──────────────────────────────────────────
        timer = gpu_timer or GPUTimer()   # timer vazio se não fornecido
        gpu_time_s    = round(timer.elapsed_s,           4)
        gpu_peak_mb   = round(timer.peak_memory_mb,      2)
        gpu_curr_mb   = round(timer.current_memory_mb,   2)
        gpu_util      = _get_gpu_utilization()

        # ── Volume de comunicação ─────────────────────────────────────────────
        sent_mb     = round(data_tracker.round_sent_mb,     3) if data_tracker else 0.0
        recv_mb     = round(data_tracker.round_received_mb, 3) if data_tracker else 0.0
        cumul_mb    = round(data_tracker.cumulative_mb,     3) if data_tracker else 0.0

        record = OperationalRecord(
            timestamp             = datetime.now(timezone.utc).isoformat(timespec="seconds"),
            round_number          = round_number,
            client_id             = self.client_id,
            partition_id          = self.partition_id,
            phase                 = phase,
            accuracy              = clf.accuracy,
            precision_micro       = clf.precision_micro,
            recall_micro          = clf.recall_micro,
            f1_micro              = clf.f1_micro,
            precision_macro       = clf.precision_macro,
            recall_macro          = clf.recall_macro,
            f1_macro              = clf.f1_macro,
            precision_weighted    = clf.precision_weighted,
            recall_weighted       = clf.recall_weighted,
            f1_weighted           = clf.f1_weighted,
            n_samples             = clf.n_samples,
            n_correct             = clf.n_correct,
            n_classes_true        = clf.n_classes_true,
            n_classes_pred        = clf.n_classes_pred,
            total_comm_rounds     = self._comm_rounds,
            params_sent_mb        = sent_mb,
            params_received_mb    = recv_mb,
            cumulative_data_mb    = cumul_mb,
            gpu_processing_time_s = gpu_time_s,
            gpu_memory_peak_mb    = gpu_peak_mb,
            gpu_memory_current_mb = gpu_curr_mb,
            gpu_utilization_pct   = gpu_util,
            cpu_usage_pct         = sys_stats["cpu_usage_pct"],
            ram_usage_gb          = sys_stats["ram_usage_gb"],
            train_loss            = float(tm.get("train_loss",       0.0)),
            train_perplexity      = float(tm.get("train_perplexity", 0.0)),
            eval_loss             = float(tm.get("eval_loss",        0.0)),
            eval_perplexity       = float(tm.get("eval_perplexity",  0.0)),
            has_gpu               = torch.cuda.is_available(),
            gpu_name              = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        )

        self._write_row(record)

        log.info(
            "Round %d [%s] | F1-micro=%.4f | F1-macro=%.4f | "
            "tempo=%.2fs | GPU_peak=%.1fMB | dados=↑%.2fMB ↓%.2fMB | loss=%.4f",
            round_number, phase,
            clf.f1_micro, clf.f1_macro,
            gpu_time_s, gpu_peak_mb,
            sent_mb, recv_mb,
            float(tm.get("train_loss", tm.get("eval_loss", 0.0))),
        )

        if data_tracker:
            data_tracker.reset_round()

        return record

    def _write_row(self, record: OperationalRecord) -> None:
        """Escreve uma linha no CSV de forma thread-safe."""
        row = asdict(record)
        with self._lock:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writerow(row)

    # ── Análise pós-treinamento ───────────────────────────────────────────────

    def load_history(self) -> pd.DataFrame:
        """
        Carrega o histórico completo de rounds do CSV em um DataFrame pandas.

        Returns:
            DataFrame com todas as linhas registradas, `timestamp` como índice
            e tipos numéricos corretamente inferidos.
        """
        if not self.csv_path.exists():
            log.warning("CSV de histórico não encontrado: %s", self.csv_path)
            return pd.DataFrame(columns=self._FIELDNAMES)

        df = pd.read_csv(
            self.csv_path,
            parse_dates=["timestamp"],
            dtype={
                "round_number":   "int32",
                "partition_id":   "int32",
                "n_samples":      "int32",
                "n_correct":      "int32",
                "total_comm_rounds": "int32",
            },
        )
        return df

    def summary(self) -> str:
        """
        Gera um resumo textual do histórico de rounds.
        Útil para inspecionar convergência e custo de comunicação ao final
        de uma sessão federada.

        Returns:
            String multi-linha formatada com estatísticas por fase e round.
        """
        df = self.load_history()
        if df.empty:
            return "Histórico vazio — nenhum round registrado."

        lines = [
            "",
            "═" * 64,
            f"  Resumo da sessão federada — {self.client_id}",
            "═" * 64,
        ]

        for phase in df["phase"].unique():
            sub = df[df["phase"] == phase]
            lines += [
                f"\n  Fase: {phase.upper()}  ({len(sub)} rounds)",
                f"  {'─'*60}",
                f"  F1-micro  : {sub['f1_micro'].mean():.4f} ± {sub['f1_micro'].std():.4f}"
                f"  (último: {sub['f1_micro'].iloc[-1]:.4f})",
                f"  F1-macro  : {sub['f1_macro'].mean():.4f} ± {sub['f1_macro'].std():.4f}"
                f"  (último: {sub['f1_macro'].iloc[-1]:.4f})",
                f"  Acurácia  : {sub['accuracy'].mean():.4f} ± {sub['accuracy'].std():.4f}",
                f"  Tempo GPU : {sub['gpu_processing_time_s'].sum():.2f}s total"
                f"  (média: {sub['gpu_processing_time_s'].mean():.2f}s/round)",
                f"  GPU peak  : {sub['gpu_memory_peak_mb'].max():.1f} MB (máximo)",
            ]
            if phase == "fit":
                lines += [
                    f"  Train loss: {sub['train_loss'].iloc[-1]:.4f} (último round)",
                    f"  Train ppl : {sub['train_perplexity'].iloc[-1]:.2f} (último round)",
                ]
            else:
                lines += [
                    f"  Eval loss : {sub['eval_loss'].iloc[-1]:.4f} (último round)",
                    f"  Eval ppl  : {sub['eval_perplexity'].iloc[-1]:.2f} (último round)",
                ]

        comm = df[df["phase"] == "fit"]
        if not comm.empty:
            total_sent = comm["params_sent_mb"].sum()
            total_recv = comm["params_received_mb"].sum()
            total_comm = comm["cumulative_data_mb"].iloc[-1] if not comm.empty else 0.0
            lines += [
                f"\n  {'─'*60}",
                f"  Comunicação federada ({len(comm)} rounds de fit):",
                f"    Dados enviados     : {total_sent:.2f} MB",
                f"    Dados recebidos    : {total_recv:.2f} MB",
                f"    Total acumulado    : {total_comm:.2f} MB",
                f"    CPU médio          : {df['cpu_usage_pct'].mean():.1f}%",
                f"    RAM pico           : {df['ram_usage_gb'].max():.2f} GB",
            ]

        lines.append("═" * 64 + "\n")
        return "\n".join(lines)

    def convergence_table(self) -> pd.DataFrame:
        """
        Retorna um DataFrame resumindo a evolução das métricas por round —
        útil para plotar curvas de convergência.

        Returns:
            DataFrame indexado por `round_number` com F1 micro/macro, loss e
            volume de comunicação acumulado.
        """
        df = self.load_history()
        if df.empty:
            return df

        cols = [
            "round_number", "phase", "f1_micro", "f1_macro", "accuracy",
            "train_loss", "eval_loss", "train_perplexity", "eval_perplexity",
            "gpu_processing_time_s", "cumulative_data_mb",
        ]
        available = [c for c in cols if c in df.columns]
        return df[available].sort_values(["round_number", "phase"]).reset_index(drop=True)


# ── CLI / demonstração ────────────────────────────────────────────────────────

def _demo_run(csv_path: Path) -> None:
    """
    Simula 3 rounds federados com dados sintéticos para demonstrar o logger.
    Todos os valores são fictícios — nenhum modelo real é carregado.
    """
    import random

    CODES = ["I10", "I50.0", "J44.1", "J45.9", "E11.9", "E03.9", "M54.5", "F32.9"]
    rng   = random.Random(42)

    logger  = FederatedRunLogger(csv_path=csv_path, client_id="demo-node-0", partition_id=0)
    tracker = DataVolumeTracker()

    # Simula pesos LoRA: 50 tensores de tamanhos variados (≈ 8B × 0.3% params ≈ 24M params)
    def fake_lora_params(seed: int) -> list[np.ndarray]:
        rs = np.random.RandomState(seed)
        return [rs.randn(256, 64).astype(np.float32) for _ in range(50)]

    for rnd in range(1, 4):
        global_params  = fake_lora_params(seed=rnd * 10)
        updated_params = fake_lora_params(seed=rnd * 10 + 1)
        tracker.record_received(global_params)
        tracker.record_sent(updated_params)

        # ── Predições sintéticas com acurácia crescente ──────────────────────
        n = 20
        y_true = [rng.choice(CODES) for _ in range(n)]
        # Simula melhora gradual: probabilidade de acerto aumenta por round
        accuracy_target = 0.4 + rnd * 0.15
        y_pred = [
            yt if rng.random() < accuracy_target else rng.choice(CODES)
            for yt in y_true
        ]

        # ── Timer simulado (sem GPU real) ─────────────────────────────────────
        timer = GPUTimer()
        timer._cpu_start = 0.0
        timer.elapsed_ms = rng.uniform(800, 2500)   # 0.8–2.5 s simulados
        timer.elapsed_s  = timer.elapsed_ms / 1000

        training_metrics = {
            "train_loss":       round(1.5 - rnd * 0.3 + rng.uniform(-0.05, 0.05), 4),
            "train_perplexity": round(4.5 - rnd * 0.8 + rng.uniform(-0.1, 0.1),  2),
            "eval_loss":        round(1.6 - rnd * 0.25 + rng.uniform(-0.05, 0.05), 4),
            "eval_perplexity":  round(4.8 - rnd * 0.7 + rng.uniform(-0.1, 0.1),  2),
        }

        rec = logger.log_round(
            round_number     = rnd,
            phase            = "fit",
            y_true           = y_true,
            y_pred           = y_pred,
            gpu_timer        = timer,
            data_tracker     = tracker,
            training_metrics = training_metrics,
        )

        # Exibe relatório de classificação do último round
        if rnd == 3:
            clf = compute_classification_metrics(y_true, y_pred)
            print_classification_report(clf, title=f"Avaliação Round {rnd} — Demo")

    print(logger.summary())
    print(f"CSV salvo em: {csv_path}")
    print("\nTabela de convergência:")
    print(logger.convergence_table().to_string(index=False))


def main() -> None:
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Logger de métricas federadas — executa demo com dados sintéticos.",
    )
    parser.add_argument(
        "--csv",
        type    = Path,
        default = _LOG_DIR / "federated_run.csv",
        help    = "Caminho do arquivo CSV de saída.",
    )
    parser.add_argument(
        "--summary",
        action  = "store_true",
        help    = "Exibe resumo de um CSV existente sem gerar novos dados.",
    )
    args = parser.parse_args()

    if args.summary:
        logger = FederatedRunLogger(csv_path=args.csv)
        print(logger.summary())
        ct = logger.convergence_table()
        if not ct.empty:
            print("\nTabela de convergência:")
            print(ct.to_string(index=False))
    else:
        _demo_run(args.csv)


if __name__ == "__main__":
    main()
