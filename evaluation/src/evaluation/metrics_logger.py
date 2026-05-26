"""
metrics_logger.py
-----------------
Evaluation and logging module for the Federated Learning pipeline over FHIR data.

Responsibilities:
    1. ICD-10 classification metrics via scikit-learn:
       precision, recall, and F1-score (micro, macro, weighted per class);
       top-1 accuracy; per-ICD-10 code report; most frequent errors.

    2. Operational logger to CSV:
       - Federated communication rounds and data volume transmitted
       - Local processing time (GPU via torch.cuda.Event or CPU via perf_counter)
       - Peak GPU memory, CPU utilisation, and RAM
       - Training/evaluation metrics received from fl_client

    3. Post-training analysis utilities:
       convergence curve by round, comparison across Non-IID partitions.

Usage:
    # Per-round logging during federated execution
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

    # Post-training analysis
    df = logger.load_history()
    print(logger.summary())

Environment variables:
    FL_LOG_DIR       Output directory for CSVs (default: evaluation/logs)
    FL_LOG_CLIENT_ID Client/node identifier (default: "client-0")
    FL_PARTITION_ID  Non-IID partition for this client (default: -1)
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
    precision_recall_fscore_support,
)

log = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

_LOG_DIR    = Path(os.getenv("FL_LOG_DIR",   "evaluation/logs"))
_CLIENT_ID  = os.getenv("FL_LOG_CLIENT_ID",  "client-0")
_PARTITION  = int(os.getenv("FL_PARTITION_ID", "-1"))

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class MetricsReport:
    """
    Complete result from one ICD-10 classification evaluation round.

    All average fields are computed by scikit-learn with
    `zero_division=0` to avoid warnings for classes with no predictions.
    """
    # ── Micro (treats each sample equally — affected by frequent classes)
    precision_micro:    float = 0.0
    recall_micro:       float = 0.0
    f1_micro:           float = 0.0

    # ── Macro (simple mean per class — penalises imbalance)
    precision_macro:    float = 0.0
    recall_macro:       float = 0.0
    f1_macro:           float = 0.0

    # ── Weighted (class mean weighted by support)
    precision_weighted: float = 0.0
    recall_weighted:    float = 0.0
    f1_weighted:        float = 0.0

    # ── Samples
    accuracy:           float = 0.0
    n_samples:          int   = 0
    n_correct:          int   = 0
    n_classes_true:     int   = 0   # unique classes in y_true
    n_classes_pred:     int   = 0   # unique classes in y_pred

    # ── Per class (dict {icd10_code: {precision, recall, f1, support}})
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    # ── Most frequent errors (list of (true, pred, count))
    top_errors: list[tuple[str, str, int]] = field(default_factory=list)


@dataclass
class OperationalRecord:
    """
    One CSV row — captures the state of a federated round.
    Fields follow the column order in the CSV file.
    """
    # ── Temporal and identification context
    timestamp:              str   = ""
    round_number:           int   = 0
    client_id:              str   = ""
    partition_id:           int   = -1
    phase:                  str   = "fit"   # "fit" | "evaluate"

    # ── ICD-10 classification metrics
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

    # ── Federated communication volume
    total_comm_rounds:      int   = 0     # cumulative rounds in the session
    params_sent_mb:         float = 0.0   # LoRA weights sent this round (MB)
    params_received_mb:     float = 0.0   # LoRA weights received this round (MB)
    cumulative_data_mb:     float = 0.0   # total data in the session (MB)

    # ── Processing time and resources
    gpu_processing_time_s:  float = 0.0   # round time (GPU Event or perf_counter)
    gpu_memory_peak_mb:     float = 0.0   # peak VRAM allocated this round (MB)
    gpu_memory_current_mb:  float = 0.0   # VRAM allocated at end of round (MB)
    gpu_utilization_pct:    float = 0.0   # GPU utilisation % (nvidia-smi, best-effort)
    cpu_usage_pct:          float = 0.0   # CPU % at log time
    ram_usage_gb:           float = 0.0   # process RAM usage (GB)

    # ── Model training/evaluation metrics (forwarded from fl_client)
    train_loss:             float = 0.0
    train_perplexity:       float = 0.0
    eval_loss:              float = 0.0
    eval_perplexity:        float = 0.0

    # ── Environment
    has_gpu:                bool  = False
    gpu_name:               str   = "none"


# ── GPUTimer ───────────────────────────────────────────────────────────────────

class GPUTimer:
    """
    Precision timer for GPU/CPU operations.

    On GPU: uses `torch.cuda.Event` with explicit synchronisation — measures the
    actual execution time on the device, including asynchronous kernel launches.
    Avoids the bias of `time.perf_counter` which returns before CUDA kernels finish.

    On CPU: falls back to `time.perf_counter`.

    Context manager usage:
        with GPUTimer() as t:
            model(input)
        print(t.elapsed_s)   # seconds
        print(t.elapsed_ms)  # milliseconds

    Manual usage:
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
        """Peak GPU memory allocated during the timed interval (MB)."""
        if self._use_cuda:
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    @property
    def current_memory_mb(self) -> float:
        """GPU memory currently allocated (MB)."""
        if self._use_cuda:
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0


# ── DataVolumeTracker ──────────────────────────────────────────────────────────

class DataVolumeTracker:
    """
    Tracks data volume (bytes/MB) transmitted during a federated session —
    sent to the server and received from it.

    LoRA weights are represented as lists of NDArrays (float32);
    volume is computed as the sum of each array's `nbytes`.

    Thread-safe: multiple clients can record simultaneously in
    multi-node simulation scenarios.
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._sent_bytes:   int = 0
        self._recv_bytes:   int = 0
        self._round_sent:   int = 0   # bytes sent this round
        self._round_recv:   int = 0   # bytes received this round

    def record_sent(self, parameters: list[np.ndarray]) -> float:
        """
        Records LoRA weights sent to the server this round.

        Args:
            parameters: List of NDArrays (output of `get_lora_parameters`).

        Returns:
            Volume sent this round in MB.
        """
        nbytes = sum(a.nbytes for a in parameters)
        with self._lock:
            self._sent_bytes  += nbytes
            self._round_sent   = nbytes
        mb = nbytes / 1024**2
        log.debug("Sent: %.3f MB (%d tensors)", mb, len(parameters))
        return mb

    def record_received(self, parameters: list[np.ndarray]) -> float:
        """
        Records global LoRA weights received from the server this round.

        Args:
            parameters: List of NDArrays (global parameters post-aggregation).

        Returns:
            Volume received this round in MB.
        """
        nbytes = sum(a.nbytes for a in parameters)
        with self._lock:
            self._recv_bytes  += nbytes
            self._round_recv   = nbytes
        mb = nbytes / 1024**2
        log.debug("Received: %.3f MB (%d tensors)", mb, len(parameters))
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
        """Resets current-round counters (cumulative totals preserved)."""
        with self._lock:
            self._round_sent = 0
            self._round_recv = 0


# ── Classification metric functions ───────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    top_k_errors: int = 5,
) -> MetricsReport:
    """
    Computes ICD-10 classification metrics using scikit-learn.

    Args:
        y_true:       Ground-truth labels (ICD-10 codes, e.g. ["I10", "J44.1"]).
        y_pred:       Model predictions (same structure).
        labels:       Fixed set of classes. If None, inferred from the data.
        top_k_errors: Number of most frequent (true, pred) error pairs to report.

    Returns:
        MetricsReport with all metrics computed.

    Notes:
        - `zero_division=0`: avoids warnings for classes without predictions —
          returns 0 in those cases, which is conservative but not masking.
        - Micro-F1 equals accuracy for single-label tasks (each sample belongs
          to exactly one ICD-10 code).
    """
    if not y_true:
        log.warning("compute_classification_metrics called with empty y_true.")
        return MetricsReport()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}"
        )

    report = MetricsReport()
    report.n_samples      = len(y_true)
    report.n_correct      = int(accuracy_score(y_true, y_pred, normalize=False))
    report.accuracy       = round(float(accuracy_score(y_true, y_pred)), 6)
    report.n_classes_true = len(set(y_true))
    report.n_classes_pred = len(set(y_pred))

    # ── Micro ──────────────────────────────────────────────────────────────────
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", labels=labels, zero_division=0
    )
    report.precision_micro = round(float(p), 6)
    report.recall_micro    = round(float(r), 6)
    report.f1_micro        = round(float(f), 6)

    # ── Macro ──────────────────────────────────────────────────────────────────
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    report.precision_macro = round(float(p), 6)
    report.recall_macro    = round(float(r), 6)
    report.f1_macro        = round(float(f), 6)

    # ── Weighted ───────────────────────────────────────────────────────────────
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    report.precision_weighted = round(float(p), 6)
    report.recall_weighted    = round(float(r), 6)
    report.f1_weighted        = round(float(f), 6)

    # ── Per class ──────────────────────────────────────────────────────────────
    full_report: dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    # Remove aggregate average entries — keep only per-ICD-10-code entries
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

    # ── Most frequent errors ───────────────────────────────────────────────────
    error_counts: dict[tuple[str, str], int] = {}
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            error_counts[(yt, yp)] = error_counts.get((yt, yp), 0) + 1
    report.top_errors = [
        (yt, yp, cnt)
        for (yt, yp), cnt in sorted(error_counts.items(), key=lambda x: -x[1])[:top_k_errors]
    ]

    return report


def print_classification_report(report: MetricsReport, title: str = "ICD-10 Evaluation") -> None:
    """
    Prints a formatted report to the terminal with classification metrics.

    Args:
        report: MetricsReport returned by `compute_classification_metrics`.
        title:  Section title displayed.
    """
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Samples      : {report.n_samples:>6}  |  Correct: {report.n_correct:>6}")
    print(f"  Accuracy     : {report.accuracy:>8.4f}")
    print(f"  Classes (GT) : {report.n_classes_true:>6}  |  Classes (pred): {report.n_classes_pred}")
    print(sep)
    print(f"  {'Metric':<22} {'Micro':>8}  {'Macro':>8}  {'Weighted':>9}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*9}")
    print(f"  {'Precision':<22} {report.precision_micro:>8.4f}  {report.precision_macro:>8.4f}  {report.precision_weighted:>9.4f}")
    print(f"  {'Recall':<22} {report.recall_micro:>8.4f}  {report.recall_macro:>8.4f}  {report.recall_weighted:>9.4f}")
    print(f"  {'F1-Score':<22} {report.f1_micro:>8.4f}  {report.f1_macro:>8.4f}  {report.f1_weighted:>9.4f}")

    if report.per_class:
        print(f"\n  {'ICD-10':<14} {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Support':>8}")
        print(f"  {'─'*14} {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
        for code, m in sorted(report.per_class.items(), key=lambda x: -x[1]["support"]):
            print(f"  {code:<14} {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['support']:>8}")

    if report.top_errors:
        print(f"\n  Top-{len(report.top_errors)} most frequent errors:")
        print(f"  {'True':<14} {'Predicted':<14} {'Count':>8}")
        print(f"  {'─'*14} {'─'*14} {'─'*8}")
        for yt, yp, cnt in report.top_errors:
            print(f"  {yt:<14} {yp:<14} {cnt:>8}")

    print(f"{sep}\n")


# ── System metric collection ───────────────────────────────────────────────────

def _get_gpu_utilization() -> float:
    """
    Queries GPU utilisation (%) via nvidia-smi.
    Returns 0.0 if nvidia-smi is unavailable or fails.
    Best-effort — does not interrupt the main flow on error.
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
    Collects CPU (%) and RAM (GB) usage for the current process.
    Always available — does not depend on CUDA.
    """
    proc = psutil.Process()
    return {
        "cpu_usage_pct": round(psutil.cpu_percent(interval=None), 1),
        "ram_usage_gb":  round(proc.memory_info().rss / 1024**3, 3),
    }


# ── FederatedRunLogger ─────────────────────────────────────────────────────────

class FederatedRunLogger:
    """
    Thread-safe logger that records operational metrics for each federated round
    to a CSV file incrementally.

    The file is created with a header on first write and opened in append mode
    on subsequent calls — survives client restarts without losing session history.

    Args:
        csv_path:    Path to the CSV output file.
        client_id:   Node client identifier (e.g. "node-cardiology-0").
        partition_id: Non-IID partition for this client.

    Example:
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
        self._comm_rounds = 0              # cumulative rounds this session
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        """Creates the CSV file with a header if it does not yet exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writeheader()
            log.info("Metrics CSV created: %s", self.csv_path)

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
        Records one row in the CSV with round metrics.

        Args:
            round_number:     Current federated round number.
            phase:            "fit" (training) or "evaluate" (evaluation).
            y_true:           Ground-truth ICD-10 labels (optional — classification
                              metrics are zeroed if None).
            y_pred:           Model ICD-10 predictions (must accompany y_true).
            gpu_timer:        GPUTimer already stopped (.stop() called).
            data_tracker:     DataVolumeTracker with current-round volumes.
            training_metrics: Dict with "train_loss", "train_perplexity",
                              "eval_loss", "eval_perplexity" (from fl_client).
            labels:           Fixed set of ICD-10 classes for sklearn.

        Returns:
            OperationalRecord logged (useful for tests and inspection).
        """
        self._comm_rounds += 1
        tm = training_metrics or {}
        sys_stats = _get_system_stats()

        # ── Classification metrics ─────────────────────────────────────────────
        clf = MetricsReport()
        if y_true and y_pred:
            try:
                clf = compute_classification_metrics(y_true, y_pred, labels=labels)
            except Exception as exc:
                log.warning("Error computing classification metrics: %s", exc)

        # ── Timer and GPU memory ───────────────────────────────────────────────
        timer = gpu_timer or GPUTimer()   # empty timer if not provided
        gpu_time_s    = round(timer.elapsed_s,           4)
        gpu_peak_mb   = round(timer.peak_memory_mb,      2)
        gpu_curr_mb   = round(timer.current_memory_mb,   2)
        gpu_util      = _get_gpu_utilization()

        # ── Communication volume ───────────────────────────────────────────────
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
            "time=%.2fs | GPU_peak=%.1fMB | data=↑%.2fMB ↓%.2fMB | loss=%.4f",
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
        """Writes one row to the CSV in a thread-safe manner."""
        row = asdict(record)
        with self._lock:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writerow(row)

    # ── Post-training analysis ─────────────────────────────────────────────────

    def load_history(self) -> pd.DataFrame:
        """
        Loads the full round history from the CSV into a pandas DataFrame.

        Returns:
            DataFrame with all logged rows, `timestamp` as index
            and numeric types correctly inferred.
        """
        if not self.csv_path.exists():
            log.warning("History CSV not found: %s", self.csv_path)
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
        Generates a textual summary of the round history.
        Useful for inspecting convergence and communication cost at the end
        of a federated session.

        Returns:
            Multi-line formatted string with statistics by phase and round.
        """
        df = self.load_history()
        if df.empty:
            return "Empty history — no rounds recorded."

        lines = [
            "",
            "═" * 64,
            f"  Federated session summary — {self.client_id}",
            "═" * 64,
        ]

        for phase in df["phase"].unique():
            sub = df[df["phase"] == phase]
            lines += [
                f"\n  Phase: {phase.upper()}  ({len(sub)} rounds)",
                f"  {'─'*60}",
                f"  F1-micro  : {sub['f1_micro'].mean():.4f} ± {sub['f1_micro'].std():.4f}"
                f"  (last: {sub['f1_micro'].iloc[-1]:.4f})",
                f"  F1-macro  : {sub['f1_macro'].mean():.4f} ± {sub['f1_macro'].std():.4f}"
                f"  (last: {sub['f1_macro'].iloc[-1]:.4f})",
                f"  Accuracy  : {sub['accuracy'].mean():.4f} ± {sub['accuracy'].std():.4f}",
                f"  GPU time  : {sub['gpu_processing_time_s'].sum():.2f}s total"
                f"  (mean: {sub['gpu_processing_time_s'].mean():.2f}s/round)",
                f"  GPU peak  : {sub['gpu_memory_peak_mb'].max():.1f} MB (maximum)",
            ]
            if phase == "fit":
                lines += [
                    f"  Train loss: {sub['train_loss'].iloc[-1]:.4f} (last round)",
                    f"  Train ppl : {sub['train_perplexity'].iloc[-1]:.2f} (last round)",
                ]
            else:
                lines += [
                    f"  Eval loss : {sub['eval_loss'].iloc[-1]:.4f} (last round)",
                    f"  Eval ppl  : {sub['eval_perplexity'].iloc[-1]:.2f} (last round)",
                ]

        comm = df[df["phase"] == "fit"]
        if not comm.empty:
            total_sent = comm["params_sent_mb"].sum()
            total_recv = comm["params_received_mb"].sum()
            total_comm = comm["cumulative_data_mb"].iloc[-1] if not comm.empty else 0.0
            lines += [
                f"\n  {'─'*60}",
                f"  Federated communication ({len(comm)} fit rounds):",
                f"    Data sent          : {total_sent:.2f} MB",
                f"    Data received      : {total_recv:.2f} MB",
                f"    Cumulative total   : {total_comm:.2f} MB",
                f"    Mean CPU           : {df['cpu_usage_pct'].mean():.1f}%",
                f"    Peak RAM           : {df['ram_usage_gb'].max():.2f} GB",
            ]

        lines.append("═" * 64 + "\n")
        return "\n".join(lines)

    def convergence_table(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarising metric evolution by round —
        useful for plotting convergence curves.

        Returns:
            DataFrame indexed by `round_number` with F1 micro/macro, loss, and
            cumulative communication volume.
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


# ── CLI / demo ─────────────────────────────────────────────────────────────────

def _demo_run(csv_path: Path) -> None:
    """
    Simulates 3 federated rounds with synthetic data to demonstrate the logger.
    All values are fictional — no real model is loaded.
    """
    import random

    CODES = ["I10", "I50.0", "J44.1", "J45.9", "E11.9", "E03.9", "M54.5", "F32.9"]
    rng   = random.Random(42)

    logger  = FederatedRunLogger(csv_path=csv_path, client_id="demo-node-0", partition_id=0)
    tracker = DataVolumeTracker()

    # Simulate LoRA weights: 50 tensors of varying sizes (≈ 8B × 0.3% params ≈ 24M params)
    def fake_lora_params(seed: int) -> list[np.ndarray]:
        rs = np.random.RandomState(seed)
        return [rs.randn(256, 64).astype(np.float32) for _ in range(50)]

    for rnd in range(1, 4):
        global_params  = fake_lora_params(seed=rnd * 10)
        updated_params = fake_lora_params(seed=rnd * 10 + 1)
        tracker.record_received(global_params)
        tracker.record_sent(updated_params)

        # ── Synthetic predictions with increasing accuracy ─────────────────────
        n = 20
        y_true = [rng.choice(CODES) for _ in range(n)]
        # Simulate gradual improvement: hit probability increases per round
        accuracy_target = 0.4 + rnd * 0.15
        y_pred = [
            yt if rng.random() < accuracy_target else rng.choice(CODES)
            for yt in y_true
        ]

        # ── Simulated timer (no real GPU) ─────────────────────────────────────
        timer = GPUTimer()
        timer._cpu_start = 0.0
        timer.elapsed_ms = rng.uniform(800, 2500)   # 0.8–2.5 s simulated
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

        # Display classification report for the last round
        if rnd == 3:
            clf = compute_classification_metrics(y_true, y_pred)
            print_classification_report(clf, title=f"Round {rnd} Evaluation — Demo")

    print(logger.summary())
    print(f"CSV saved to: {csv_path}")
    print("\nConvergence table:")
    print(logger.convergence_table().to_string(index=False))


def main() -> None:
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Federated metrics logger — runs demo with synthetic data.",
    )
    parser.add_argument(
        "--csv",
        type    = Path,
        default = _LOG_DIR / "federated_run.csv",
        help    = "Path to the CSV output file.",
    )
    parser.add_argument(
        "--summary",
        action  = "store_true",
        help    = "Display summary of an existing CSV without generating new data.",
    )
    args = parser.parse_args()

    if args.summary:
        logger = FederatedRunLogger(csv_path=args.csv)
        print(logger.summary())
        ct = logger.convergence_table()
        if not ct.empty:
            print("\nConvergence table:")
            print(ct.to_string(index=False))
    else:
        _demo_run(args.csv)


if __name__ == "__main__":
    main()
