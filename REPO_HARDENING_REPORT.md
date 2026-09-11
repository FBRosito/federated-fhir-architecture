# Repository Hardening Report — HERALD (federated-fhir-architecture)

**Date:** 2026-09-11
**Branch:** `main` (backup: `backup-pre-hardening`, created before any change)
**Scope:** Security audit (working tree + full git history) → code cleanup →
structure/documentation → final verification. **Nothing was committed or pushed** —
all changes are in the working tree for maintainer review.

---

## 1. Security status (summary)

| Category | Status | Evidence |
|----------|--------|----------|
| MIMIC-IV data in working tree (tracked) | ✅ zero | `git ls-files` grep for `*.csv/*.parquet/*.pkl/*.pt/...` → empty |
| MIMIC-IV data in git history | ✅ zero | `git log --all --full-history -- "*.csv" ...` → no data files ever committed |
| MIMIC-IV on local disk | ⚠️ present but safe | `physionet.org/` (multi-GB) exists locally, **gitignored** (`.gitignore:71`), never tracked — see AUDIT_SECURITY.md |
| Credentials (tracked + history) | ✅ zero | patterns `sk-…`, `AKIA…`, `ghp_…`, `xox-…`, private keys, credentialed URLs, `password/secret/token/api_key` → only placeholders (`sk-or-your_key_here`) and `os.getenv(...)` reads |
| `.env` files | ✅ none | do not exist on disk; never committed; `.env.example` has placeholders only |
| Large files in history | ✅ clean | only blob > 1 MB is `graphify-out/graph.json` (tooling artifact, not sensitive); `.git` = 11 MB |
| Checkpoints/logs on disk | ⚠️ present but safe | `experiments/herald-pate/logs/*.pt` (3.3 GB) etc. — gitignored, never tracked |
| Notebooks | ✅ none tracked | `*.ipynb` gitignored |

> **Zero dados sensíveis encontrados no working tree rastreado e no git history.**
> Real MIMIC-IV data, derived FHIR bundles, experiment logs and checkpoints exist
> only as *untracked, gitignored* local files. No history rewrite is required
> for safety. Full details: [AUDIT_SECURITY.md](AUDIT_SECURITY.md).

## 2. Changes by phase

### Phase 2 — Code cleanup

**Portuguese → English (comments/UI strings):**
- `etl_worker/generate_dataset.py` — 3 comments + CLI output strings + argparse description (the Portuguese *clinical-note template strings* were **kept**: they are synthetic data, not comments).
- `experiments/article3/configs/*.yaml` — 3 header comments.
- `.gitignore` — section comments.
- **Kept intentionally (domain content):** Portuguese clinical-term regexes/prompts in `ai_client/fhir_consumer*.py` (they parse/generate Portuguese clinical text).

**TODO/FIXME/HACK:** none existed in code (the single `TODO`-ish match in `run_experiments.sh` is a log-format example). Added `TODO(maintainer)` markers in `CITATION.cff` for the author list — deliberate, outside merged-code scope.

**Dead code removed (verified unused via vulture + grep + flake8):**
- `main.py` — deleted (uv scaffold `"Hello from…"`, referenced nowhere).
- `evaluation/__init__.py` — scaffold `main()` replaced by a package docstring; console script repointed to the real CLI (`evaluation.post_eval:main`) in `evaluation/pyproject.toml`.
- Unused imports: `etl_pipeline.py` (`BundleEntry`, `BundleEntryRequest`), `icd_metrics.py` (`average_precision_score`, `asdict`), `fl_server/server.py` (`ndarrays_to_parameters`, `DPFixedClipping`, `fl`, `NDArrays`), `fl_client.py` (`Any`, local `torch`, `BertLoRAConfig`), `fhir_consumer_bert.py` (`Any`), `generate_dataset.py` (`dataclass`, `field`), `mimic_builder.py` (`math`, local `numpy`), `plots.py` (`numpy`), `gradient_inversion.py` (`torch.nn.functional as F`), `compare_strategies.py` (`json`), `article3/client.py` (`_CALIBRATE_GRAD_NORM`), `article3/dual_training.py` (`build_bert_dataset`), PoC scripts (`os`, module-level `torch` ×3).
- Unused variables: `_compute_qags(n_questions=…)` parameter (never used by body or caller — removed), `total_norm` (clip_grad_norm_ called for side effect), `ratings_dim` (llm_judge), `rec` (metrics_logger demo), `global_lora`, `tokenizer` (PoCs).
- `_FHIRHandler.log_message(fmt, *args)` renamed to match the stdlib base signature (`format`); fhir_server `p()` → `_param()`.

**prints → logging:** `fhir_server.py` startup message → `logging` with `basicConfig`. All other `print()` calls kept deliberately: they are CLI deliverable output (benchmarks, preflight report, PoC progress, JSON results) or docstring examples.

**Commented-out code blocks:** none found (scan for commented code patterns came back clean).

**Docstrings & type hints:** all public functions/classes outside test files now have Google-style docstrings and full parameter/return annotations (AST-verified: 0 gaps). ~40 functions updated across `fl_server`, `ai_client`, `etl_worker`, `evaluation`, `fhir_server.py`, `scripts/preflight_check.py`, `experiments/adaptive-clipping`, `experiments/article3`, `experiments/article3-poc`. Heavy framework types use `TYPE_CHECKING` imports (`PreTrainedModel`, `PreTrainedTokenizerBase`, `DataLoader`, `torch`).

**Formatting:** `black` (88) + `isort --profile black` applied repo-wide (47 files reformatted); `flake8` clean with `.flake8` config (max-line-length 120, E203/W503 ignored, per-file ignores only for the synthetic-data string file and PoC import-order idiom).

### Phase 3 — Structure & documentation

**Created:**
- `CITATION.cff` — citation metadata (author = "Anonymous", matching the paper-under-review state; marked with a maintainer TODO).
- `CONTRIBUTING.md` — code standards, dev setup, PR process.
- `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1.
- `SECURITY.md` — responsible disclosure via security@ufcspa.edu.br + data-handling policy.
- `.github/workflows/ci.yml` — `lint` job (black/flake8/isort) + `test` job (adaptive-clipping unit tests on CPU-only torch).
- `.pre-commit-config.yaml` — black 24.1.0, flake8 7.0.0, isort 5.13.2.
- `.flake8` — lint configuration.
- `requirements.txt` — 105 packages, fully pinned, generated from `uv.lock` (`uv export`).
- `requirements-dev.txt` — pinned dev tools (black/flake8/isort/pytest/pre-commit).
- `AUDIT_SECURITY.md` — phase-1 audit.

**Modified:**
- `README.md` — HERALD title, CI + license badges, research-software warning, License/Contributing/Security sections, updated ToC. (Architecture, Quickstart, Data Preparation, Configuration, Citation sections already existed and meet the required structure.)
- `pyproject.toml` (root) — added `[tool.black]`, `[tool.isort]`, `[tool.pytest.ini_options]`.
- `*/pyproject.toml` — placeholder descriptions replaced with real ones.

**LICENSE:** MIT — already present. ⚠️ Copyright holder is `"Anonymous"` — see §3.

## 3. Decisions required from the maintainer (before going public)

1. **LICENSE + CITATION.cff authorship** — replace `"Anonymous"` with the real author(s)/institution (or keep anonymous for double-blind review and update later).
2. **Publication scope (`.gitignore`)** — the following are currently gitignored entirely, *including their source code*: `experiments/herald-pate/` (except `MANIFEST_M0.md`), `experiments/client-level-dp/`, `papers/*/`, `docs/fase*_resultados.md` and similar notes. Decide whether the source of those experiments should be published; if yes, narrow the ignore rules to exclude only `.venv/`, `logs/`, and data outputs.
3. **Portuguese commit messages in history** (e.g. `15c077c`) — cosmetic only; rewriting history changes all hashes. Optional.
4. **`graphify-out/`** is tracked but also gitignored (added before the rule). Recommended: `git rm -r --cached graphify-out/` so the ignore rule takes effect (it is regenerated locally).
5. **Untracked source files to review and add**: `.gitattributes` (graphify merge driver), `experiments/adaptive-clipping/{analysis/renyi_group_composition.py, scripts/run_baseline_sigma1_n10.sh, tests/test_renyi_group_composition.py}` — all lint-clean and test-covered.

## 4. Final verification results (all run on the cleaned tree)

| Check | Result |
|-------|--------|
| Tracked data files (`*.csv/parquet/pkl/pt/pth/ckpt/safetensors`, `.env`) | ✅ none |
| Credentials in full history | ✅ none (10 grep hits = placeholders/`os.getenv` only) |
| `.git` size | ✅ 11 MB (no data blobs) |
| `.gitignore` effective on `data/`, `checkpoints/`, `etl_worker/data`, `physionet.org` | ✅ rules match |
| `black --check .` | ✅ clean |
| `isort --check-only .` | ✅ clean |
| `flake8 .` | ✅ clean (0 findings) |
| `pytest experiments/adaptive-clipping/tests/` | ✅ 12 passed |
| `python -m py_compile` on every tracked `.py` | ✅ all compile |
| Public-function docstring/type-hint coverage (AST scan, excl. tests) | ✅ 0 gaps |

## 5. How to review and publish

```bash
git status                                  # review the change list
git diff                                    # review code changes
git add -A                                  # stages only non-ignored files
git status --porcelain | grep '^[AM]'       # confirm no data/logs staged
git commit -m "chore: harden repository for public release"
```

The CI workflow (`.github/workflows/ci.yml`) will run on the first push.
