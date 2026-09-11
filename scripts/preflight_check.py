#!/usr/bin/env python3
"""
preflight_check.py
-------------------
Fase 0 environment guard for the HERALD DP research line (client-level DP,
FedSVD, federated distillation — Fases 1-3).

Validates that the experimental environment matches the canonical
configuration documented in docs/validated_environment.md, and aborts
(exit 1) with an actionable message on the FIRST failing check, before any
GPU time is spent training.

Historical context (see docs/validated_environment.md for full detail):
three silent-failure bugs invalidated months of DP-SGD experiments before
this script existed, and none of them raised an exception — the
RDPAccountant kept reporting a valid-looking epsilon throughout:
  1. PerLayerClipper cached nn.Parameter references at construction time;
     HERALD reloads a brand-new model every round, so those references
     went stale from round 2 onward and training silently ran as plain
     SGD with no clipping or noise.
  2. FL_LEARNING_RATE was not exported by the experiment scripts, so the
     server's silent default (5e-5) overrode the correct BERT value
     (2e-4) for most of every run.
  3. MAX_ADMISSIONS=2000 was loaded into HAPI FHIR instead of 10000,
     giving ~295 examples/silo instead of ~1200 and an epsilon computed
     against the wrong subsampling rate.

This script only covers checks that can run BEFORE training starts (FHIR
dataset size, FL_LEARNING_RATE presence/validity, per-silo example
counts). The round-1 "effective LR == env var" assertion (task 1c) cannot
run here — Flower has no server config HTTP endpoint to query out of
band — so it is implemented instead as a log + assert inside each client's
fit() (see ai_client/fl_client.py, experiments/adaptive-clipping/src/adaptive_clipping/client.py,
experiments/article3/src/article3/client.py), which fires on the client's
first real round.

Usage:
    uv run python scripts/preflight_check.py          # from the repo root
    (cd "$REPO_ROOT" && uv run python scripts/preflight_check.py)   # from any experiment script

Exit 0: environment OK. Exit 1: at least one check failed.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict


class PreflightError(RuntimeError):
    """Carries an actionable message; caught once per check in main()."""


def _fhir_url() -> str:
    return os.environ.get("FHIR_SERVER_URL", "http://localhost:8080/fhir").rstrip("/")


def check_fhir_dataset_size() -> None:
    """(1a) HAPI FHIR's Patient count must match MAX_ADMISSIONS.

    Also the connectivity check: a stopped FHIR/Flower server surfaces here
    as a URLError, satisfying the "servidor parado" acceptance scenario.
    """
    expected = int(os.environ.get("MAX_ADMISSIONS", "10000"))
    url = f"{_fhir_url()}/Patient?_summary=count"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise PreflightError(
            f"[1a] Could not reach FHIR server at {url}: {exc}\n"
            f"  Fix: make sure the HAPI FHIR container is running — `make up-infra`."
        ) from exc
    total = data.get("total")
    if total is None:
        raise PreflightError(f"[1a] FHIR responded with no 'total' field: {data}")
    if total != expected:
        raise PreflightError(
            f"[1a] FHIR has {total} Patient(s), expected {expected} (from MAX_ADMISSIONS).\n"
            f"  This is the exact dataset-volume bug that produced ~295 examples/silo "
            f"instead of ~1200 (MAX_ADMISSIONS=2000 loaded instead of 10000).\n"
            f"  Fix:\n"
            f"    docker compose rm -sf hapi_fhir && "
            f"make up-mimic MAX_ADMISSIONS={expected} N_SILOS=5 DIRICHLET_ALPHA=0.5 "
            f"BENCHMARK=top50 ICD_VERSION=icd10"
        )
    print(
        f"[1a] OK — FHIR Patient count = {total} (matches MAX_ADMISSIONS={expected})."
    )


def check_learning_rate_exported() -> None:
    """(1b) FL_LEARNING_RATE must be explicitly exported — never a silent default."""
    raw = os.environ.get("FL_LEARNING_RATE")
    if raw is None:
        raise PreflightError(
            "[1b] FL_LEARNING_RATE is not set in the environment.\n"
            "  This is the exact bug that let the server's silent default (5e-5) "
            "override the correct BERT LR (2e-4) for most of every round.\n"
            "  Fix: export FL_LEARNING_RATE=2e-4 before starting the server/clients."
        )
    try:
        lr = float(raw)
    except ValueError as exc:
        raise PreflightError(
            f"[1b] FL_LEARNING_RATE={raw!r} is not a valid float."
        ) from exc

    backend = os.environ.get("MODEL_BACKEND", "bert").strip().lower()
    if backend == "bert" and lr < 1e-4:
        raise PreflightError(
            f"[1b] FL_LEARNING_RATE={lr:.2e} is below the minimum valid value for the "
            f"BERT backend (>= 1e-4; canonical value is 2e-4).\n"
            f"  Fix: export FL_LEARNING_RATE=2e-4"
        )
    print(f"[1b] OK — FL_LEARNING_RATE={lr:.2e} (backend={backend}).")


def check_examples_per_silo() -> None:
    """(1d) Every silo must have >= 50% of the mean per-silo example count.

    Uses the mean of the OBSERVED counts rather than a hardcoded target: this
    catches both the MAX_ADMISSIONS volume bug (all silos low together) and a
    degenerate Dirichlet partition (one silo starved while others are fine),
    without hardcoding an exact per-silo number that shifts with
    MAX_ADMISSIONS/N_SILOS/DIRICHLET_ALPHA.
    """
    try:
        from ai_client.fhir_consumer import fetch_training_examples
    except ImportError as exc:
        raise PreflightError(
            "[1d] Could not import ai_client.fhir_consumer. This check must run via "
            "`uv run python scripts/preflight_check.py` from the repo root — ai_client "
            "is a uv workspace member there (see pyproject.toml)."
        ) from exc

    n_silos = int(os.environ.get("FL_MIN_CLIENTS", "5"))
    examples, stats = fetch_training_examples(_fhir_url())
    if not examples:
        raise PreflightError(
            f"[1d] fetch_training_examples() returned 0 examples from {_fhir_url()}. "
            f"Warnings: {stats.warnings}"
        )

    per_silo: dict[int, int] = defaultdict(int)
    untagged = 0
    for ex in examples:
        note = ex.partition_note or ""
        for pid in range(n_silos):
            if f"partition_id={pid}" in note:
                per_silo[pid] += 1
                break
        else:
            untagged += 1

    if not per_silo:
        raise PreflightError(
            f"[1d] None of the {len(examples)} examples carry a 'partition_id=N' tag "
            f"for N in [0, {n_silos}). Either the FHIR data predates Dirichlet "
            f"partitioning, or FL_MIN_CLIENTS={n_silos} does not match the N_SILOS "
            f"used when the dataset was built."
        )

    counts = [per_silo.get(i, 0) for i in range(n_silos)]
    mean = sum(counts) / n_silos
    threshold = mean * 0.5
    starved = [(i, c) for i, c in enumerate(counts) if c < threshold]

    print(
        f"[1d] Per-silo example counts: {dict(enumerate(counts))} "
        f"(mean={mean:.0f}, untagged={untagged})"
    )
    if starved:
        raise PreflightError(
            f"[1d] Silo(s) {[i for i, _ in starved]} have fewer than 50% of the mean "
            f"per-silo example count ({threshold:.0f}): {starved}.\n"
            f"  This detects both the dataset-volume bug and a degenerate Dirichlet "
            f"partition (some silos starved of data at low alpha).\n"
            f"  Fix: check MAX_ADMISSIONS/DIRICHLET_ALPHA used to build the dataset, "
            f"or rebuild with a higher alpha via `make up-mimic ...`."
        )
    print(f"[1d] OK — all {n_silos} silos have >= 50% of the mean example count.")


CHECKS: list[tuple[str, "callable"]] = [
    ("1a: FHIR dataset size", check_fhir_dataset_size),
    ("1b: FL_LEARNING_RATE exported", check_learning_rate_exported),
    ("1d: per-silo example count", check_examples_per_silo),
]


def main() -> int:
    """Run all preflight checks in order.

    Returns:
        0 when every check passes, 1 on the first failure.
    """
    print("=== HERALD Fase 0 preflight check ===")
    for label, fn in CHECKS:
        try:
            fn()
        except PreflightError as exc:
            print(f"\nPREFLIGHT FAILED at {label}:\n{exc}\n", file=sys.stderr)
            return 1
    print("=== preflight OK — environment matches docs/validated_environment.md ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
