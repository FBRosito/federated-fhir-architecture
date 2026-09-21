"""Persist the operator E's original -> equalized config mapping."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dp_equalization.accountant import MethodConfig


def save_equalization_map(entries: list[dict], path: Path) -> Path:
    """Write the mapping produced by ``equalize()`` calls to ``path`` as JSON.

    Each entry: {"original": MethodConfig, "eps_star": float,
    "equalized": MethodConfig, "epsilon_achieved": float}.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "original": asdict(_as_method_config(e["original"])),
            "eps_star": e["eps_star"],
            "equalized": asdict(_as_method_config(e["equalized"])),
            "epsilon_achieved": e["epsilon_achieved"],
        }
        for e in entries
    ]
    path.write_text(json.dumps(serializable, indent=2) + "\n")
    return path


def _as_method_config(value: MethodConfig | dict) -> MethodConfig:
    return value if isinstance(value, MethodConfig) else MethodConfig(**value)
