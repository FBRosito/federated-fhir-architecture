"""
mimic_builder.py is a top-level script in etl_worker/ (not part of the
etl_worker.etl_pipeline installable package), so it is loaded here via
importlib from its file path rather than a normal package import.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MIMIC_BUILDER_PATH = Path(__file__).resolve().parents[1] / "mimic_builder.py"


@pytest.fixture(scope="session")
def mimic_builder():
    spec = importlib.util.spec_from_file_location(
        "mimic_builder_under_test", _MIMIC_BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mimic_builder_under_test"] = module
    spec.loader.exec_module(module)
    return module
