"""Tests for dp_equalization.registry.save_equalization_map()."""

from __future__ import annotations

import json
from pathlib import Path

from dp_equalization.accountant import MethodConfig
from dp_equalization.equalize import equalize
from dp_equalization.registry import save_equalization_map

RECORD_LEVEL_TEMPLATE = MethodConfig(
    method="fedprox_dpsgd_record_level",
    sigma=1.0,
    rounds=20,
    delta=1e-5,
    silos=((155, 1550), (150, 1508), (151, 1518), (152, 1527), (149, 1492)),
    accountant="rdp",
)


def test_save_equalization_map_writes_readable_json(tmp_path: Path):
    cfg, eps_achieved = equalize(RECORD_LEVEL_TEMPLATE, eps_star=4.0)
    out_path = tmp_path / "equalization_map.json"

    written = save_equalization_map(
        [
            {
                "original": RECORD_LEVEL_TEMPLATE,
                "eps_star": 4.0,
                "equalized": cfg,
                "epsilon_achieved": eps_achieved,
            }
        ],
        out_path,
    )

    assert written == out_path
    data = json.loads(out_path.read_text())
    assert len(data) == 1
    entry = data[0]
    assert entry["eps_star"] == 4.0
    assert entry["original"]["method"] == "fedprox_dpsgd_record_level"
    assert entry["equalized"]["sigma"] != RECORD_LEVEL_TEMPLATE.sigma
    assert abs(entry["epsilon_achieved"] - 4.0) / 4.0 < 0.01
