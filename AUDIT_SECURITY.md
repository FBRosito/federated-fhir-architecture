# Security Audit — federated-fhir-architecture (HERALD)

**Date:** 2026-09-10
**Scope:** working tree + full git history (`git log --all --full-history`), all branches (`main`, `origin/main`; 30 commits, `.git` = 11 MB).
**Backup:** branch `backup-pre-hardening` created before any modification.

---

## 1. Sensitive data scan

| # | Path | Risk type | Tracked? | In git history? | Gitignored? | Recommendation |
|---|------|-----------|----------|-----------------|-------------|----------------|
| 1 | `physionet.org/files/mimiciv/3.1/` + `mimic-iv-note/2.2/` (~20 `.csv.gz`, multi-GB) | **Real clinical data (MIMIC-IV)** | No | **No** (verified via `git log --all --full-history -- "*mimic*" "*.csv" "*.csv.gz"`) | Yes (`.gitignore:71`) | Keep on disk for local ETL only. Never `git add -f`. Optionally move outside the repo root. **No history rewrite needed.** |
| 2 | `etl_worker/data/bundles/*.json`, `label_index.json`, `temporal_split.json` | Derived FHIR bundles (from MIMIC) | No | No | Yes (`.gitignore:78`) | Nothing. Correctly excluded. |
| 3 | `experiment_logs/*.json`, `experiment_logs/figures/` | Unpublished experiment results (Artigo 2 baselines) | No | No | Yes (`.gitignore:108-112`) | Nothing (publication-scope decision, see §5). |
| 4 | `experiment_results_vastai/` | Unpublished results mirror | No | No | Yes (`.gitignore:154`) | Nothing. |
| 5 | `experiments/herald-pate/` — 9.3 GB total: `.venv/`, `logs/` (3.3 GB of `teacher_*.pt` checkpoints + `*.npy`), `proxy_corpus/` | Checkpoints / experiment outputs / venv | No | No | Yes (`.gitignore:169`, whole dir except `MANIFEST_M0.md`) | Nothing for safety. **Publication-scope flag:** the herald-pate *source code* (`src/`, `tests/`, `scripts/`) is also ignored — decide whether it should be published (see §5). |
| 6 | `experiments/adaptive-clipping/.venv/` | Virtualenv (CUDA libs, multi-GB) | No | No | Yes (`.gitignore:21`) | Nothing. |
| 7 | `graphify-out/graph.json` (~1.1 MB) | Tooling artifact (knowledge graph) | **Yes** (only blob > 1 MB in history) | Yes | Yes (`.gitignore:157`) — but tracked before the rule existed | Not sensitive. Recommend `git rm -r --cached graphify-out/` so the gitignore rule takes effect (regenerated locally by `graphify update .`). |

**History verification commands run:**

```bash
git log --all --full-history --oneline -- "*patient*" "*subject_id*" "*hadm_id*" "*mimic*" \
    "*.csv" "*.parquet" "*.pkl" "*.pt" "*.pth" "*.ckpt"
# → only code files (etl_worker/mimic_builder.py etc.). Zero data files ever committed.

git rev-list --objects --all | git cat-file --batch-check | awk '$1=="blob" && $3>1000000'
# → only graphify-out/graph.json (2 revisions). No data blobs in history.
```

## 2. MIMIC-IV verification (item 1.2 of the request)

- **No MIMIC-IV data file is tracked in the repository or present in any commit of any branch.** ✅
- Raw MIMIC-IV exists only on local disk under `physionet.org/` (gitignored), downloaded by the maintainer via PhysioNet credentialed access. The ETL (`etl_worker/mimic_builder.py`, `etl_worker/src/etl_worker/etl_pipeline.py`) is a script that processes user-downloaded data — by design.
- Note: Portuguese clinical-note templates in `etl_worker/generate_dataset.py` are **synthetic data strings (domain content), not comments** — they generate the synthetic dataset and must not be "translated" as comments. Flagged so the cleanup phase does not touch them. The same applies to the Portuguese clinical-term regex in `ai_client/src/ai_client/fhir_consumer.py:305`.

## 3. Credentials scan (item 1.3)

Searched tracked files **and** full history (`git log --all -p`) for: `password`, `secret`, `token`, `api_key`, `aws_access`, `private_key`, `sk-…`, `AKIA…`, `ghp_…`, `gho_…`, `xox[baprs]-…`, `-----BEGIN … PRIVATE KEY-----`, and credentialed URLs (`https://user:pass@host`).

- **Zero real credentials found** in tracked files or history. ✅
- `.env`, `.env.local`, `.env.production`: do not exist on disk; never committed. ✅
- `.env.example` contains only placeholders (`HF_TOKEN=hf_YOUR_TOKEN_HERE`, commented `# OPENROUTER_API_KEY=sk-or-...`). ✅
- No config YAML with embedded secrets; the only `http://…` strings are example identifiers (`http://hospital.example.org/patients`, a FHIR system URI, not a credential). ✅

## 4. Other findings

| Item | Status | Notes |
|------|--------|-------|
| Jupyter notebooks | ✅ none tracked | `*.ipynb` is gitignored anyway. |
| TODO/FIXME/HACK | ✅ none real | Only one match: `run_experiments.sh:273`, a log-format example inside a comment, not an action item. |
| Portuguese comments in code | ⚠️ 1 found | `etl_worker/generate_dataset.py:273` + several `.gitignore` comments (lines 69, 104-107, 116). Fixed in Phase 2. |
| Portuguese **commit messages** in history | ⚠️ | e.g. `15c077c feat: arquitetura completa — …`. History rewrite is destructive and optional — **maintainer decision**, flagged here. |
| `LICENSE` | ⚠️ | MIT, but copyright holder is `"Anonymous"` — decide the real author/institution before going public. |
| Untracked files pending review | ⚠️ | `.gitattributes` (graphify merge driver), `experiments/adaptive-clipping/{analysis/renyi_group_composition.py, scripts/run_baseline_sigma1_n10.sh, tests/test_renyi_group_composition.py}`. Legitimate source files; include in the cleanup scope before committing. |

## 5. Publication-scope decisions for the maintainer (not blockers)

The `.gitignore` currently hides *source code* of unpublished work. Before flipping the repo to public, decide:

1. `experiments/herald-pate/` — fully ignored except `MANIFEST_M0.md`. If the Fase 3 / PATE paper will be public, narrow the rule to ignore only `.venv/`, `logs/`, `proxy_corpus/`, `analysis/` outputs and publish `src/`, `tests/`, `scripts/`, `pyproject.toml`, `README.md`.
2. `experiments/client-level-dp/` — fully ignored (code included). Same decision.
3. `papers/artigo2/`, `papers/artigo3/`, `papers/eramia2026/` — fully ignored (LaTeX sources). Fine if intentional.
4. `docs/fase*_resultados.md`, `docs/artigo2_errata_ci.md`, etc. — ignored internal result notes. Fine if intentional.
5. Portuguese commit messages — keep or rewrite history (`git rebase -i` / `git-filter-repo`) before making public. Rewriting changes all commit hashes.

## 6. Conclusion

> **Zero dados sensíveis encontrados no working tree rastreado e no git history.**
> Real MIMIC-IV data and derived artifacts exist only as *untracked, gitignored* local files. No credentials anywhere. No history rewrite is required for safety. The only history-touching item is the optional cosmetic rewrite of Portuguese commit messages (§5.5).
