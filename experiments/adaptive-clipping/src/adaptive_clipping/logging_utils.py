"""
logging_utils.py
-----------------
Per-round JSONL logging for the adaptive-clipping experiment.

One file per round, per silo: logs/{experiment_tag}/{silo_id}/round_{r:03d}.jsonl,
each file containing exactly one JSON line (satisfies both the file-naming
pattern and "one line per entry" literally).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RoundLogRecord:
    round: int
    silo_id: int
    sigma: float
    clipping_strategy: str
    global_c0: float
    per_layer_thresholds: dict[str, float] = field(default_factory=dict)
    per_layer_norms: dict[str, dict] = field(default_factory=dict)
    micro_f1: float = 0.0
    epsilon: float = 0.0
    delta: float = 0.0
    wall_clock_seconds: float = 0.0


class GradientNormLogger:
    def __init__(
        self, experiment_tag: str, silo_id: int, logs_root: Path = Path("logs")
    ) -> None:
        self.path = Path(logs_root) / experiment_tag / str(silo_id)
        self.path.mkdir(parents=True, exist_ok=True)

    def log_round(self, record: RoundLogRecord) -> None:
        """Persist one round's record as a single-line JSON file."""
        out_file = self.path / f"round_{record.round:03d}.jsonl"
        with open(out_file, "w") as f:
            f.write(json.dumps(asdict(record)))
            f.write("\n")
