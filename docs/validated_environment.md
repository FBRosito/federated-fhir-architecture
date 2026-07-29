# Validated Environment — HERALD DP Research Line (Fase 0)

This document is the single source of truth for what "a correctly configured
DP-SGD experiment" means in this repository. It exists because three
independent, silent-failure bugs invalidated months of prior DP-SGD
experiments — none of them raised an exception, and the RDPAccountant kept
reporting a valid-looking epsilon throughout each one. Any deviation from the
canonical configuration below requires an explicit justification recorded in
the experiment's own log — do not change these values informally.

## Canonical validated configuration

| Parameter | Value |
|---|---|
| Base model | PubMedBERT-base (110M) |
| LoRA | r=8, α=16, target=[query, value] |
| FL algorithm | FedProx (μ=0.01) |
| Dataset | MIMIC-IV v3.1, MAX_ADMISSIONS=10000, top-50 ICD-10-CM |
| Silos | K=5, Dirichlet α=0.5, ~1900 examples/silo (observed; see note below) |
| LR (BERT) | 2e-4 rounds 1-2, server decays to 40% (≈8e-5) from round 3 |
| δ | 1e-5 |
| Accountant | RDPAccountant (Opacus v1.6.0) |

**Note on the per-silo example count**: the original investigation estimated
~1200 examples/silo at MAX_ADMISSIONS=10000. The actual measured count in
this environment (via `scripts/preflight_check.py`, task 1d) is **~1900**
per silo (range 1885-1959 across 5 silos, α=0.5). `preflight_check.py`
therefore checks against the *observed mean* rather than a hardcoded target —
this also means it correctly catches a degenerate Dirichlet split (one silo
starved while others are fine), not just a global volume regression.

## The three historical bugs, and how each is detected today

### Bug 1 — Stale `nn.Parameter` references in `PerLayerClipper`

**What happened**: `PerLayerClipper` (Article 2's per-layer adaptive clipping,
`experiments/adaptive-clipping/src/adaptive_clipping/clipping.py`) cached
`nn.Parameter` references at construction time. HERALD reloads a brand-new
model object every round via `ai_client.fl_client._load_model()`
(`FL_KEEP_MODEL_IN_VRAM` defaults to `false`), so those cached references
belonged to an already-deleted model from round 2 onward. Their `.grad` was
permanently `None`, so clipping/noise silently stopped applying while
`RDPAccountant` kept accumulating epsilon as if training were still
DP-protected — rounds 2-20 ran as plain SGD with no privacy guarantee.

**How it's detected today**:
- `PerLayerClipper` no longer caches parameter references at all — see the
  class docstring in `clipping.py`. `update_history()` takes the current
  round's live `named_parameters()` explicitly, every call, and recomputes
  grouping fresh — there is no cached state to go stale.
- **Grad-presence guard** (`clipping.py`, `update_history()`): raises
  `RuntimeError` if a parameter group has zero gradients — the direct,
  observable symptom of this bug, regardless of *why* the gradients are
  missing.
- **History-length integrity guard** (`clipping.py`, `update_history()`):
  raises if a group's history length doesn't match the internal call
  counter — catches a silently skipped or double-counted round.
- **Model-reload identity guard** (`clipping.py`:
  `param_identity_fingerprint` / `assert_model_reloaded`, wired into
  `AdaptiveClippingClient.fit()` in `client.py`): fingerprints the model's
  parameter object identities right after `_load_model()` and raises if
  they're identical to the previous round's — unless
  `FL_KEEP_MODEL_IN_VRAM=true` opts into intentional reuse. This is a
  caller-side check (the reload itself is the caller's responsibility, not
  the clipper's — see the design note below).
- **Regression test**: `experiments/adaptive-clipping/tests/test_stale_parameters.py`
  (9 tests; run via `cd experiments/adaptive-clipping && uv run pytest
  tests/test_stale_parameters.py -v`). Includes a self-contained
  reproduction of the pre-fix caching pattern
  (`_BuggyClipperCachesParamsAtInit`) proving the test suite would have
  failed against it, and was additionally verified by temporarily
  reintroducing the real bug pattern into `clipping.py` and confirming 2 of
  9 tests fail as expected (then reverting).

**Design note**: the identity-fingerprint check is deliberately *not* inside
`PerLayerClipper` itself. `clipping.py`'s own module docstring states it is
designed to be "importable and unit-testable without CUDA/model weights,"
and its existing tests legitimately reuse the same fake model object across
simulated rounds to isolate the grouping/threshold math from reload
behavior — putting a "the model must be a new object" assertion inside the
class would have broken that. The check instead lives where the reload
actually happens (`AdaptiveClippingClient.fit()`), and is exposed as a
reusable pair of functions (`param_identity_fingerprint`,
`assert_model_reloaded`) any future component (Fases 1-3: FedSVD, federated
distillation) can call the same way.

**Scope of this fix**: only `PerLayerClipper` (Article 2) held cross-round
state built from live parameter objects. A repo-wide search found no other
vulnerable component: FedProx's global-weight reference and the
per-round `RDPAccountant` instance are both recreated fresh inside
`train_bert_one_round()` on every call (which itself only runs once per
round, after a fresh model load) — nothing there persists across a reload
boundary. Article 3's dual-LoRA local-adapter checkpoint
(`dual_training.py`) round-trips through `torch.save`/`torch.load` of a
plain state dict (values, not live object references), so it isn't
susceptible to this failure mode either.

### Bug 2 — `FL_LEARNING_RATE` not exported

**What happened**: the server reads `FL_LEARNING_RATE` with a silent default
of `5e-5` (`fl_server/src/fl_server/server.py`). Experiment scripts that
didn't explicitly `export FL_LEARNING_RATE=2e-4` let the server silently use
its own default instead of the correct BERT value, for most of every run.
Note the channel: the LR the **client** actually trains with comes from the
server's per-round `fit_config` (`config["learning_rate"]`) — a client
process's own `FL_LEARNING_RATE` env var is never read directly inside
`fit()`. Every `run_*.sh` script exports the variable once in a shell block
inherited by both the server and client processes it launches, which is why
checking presence in the current process (task 1b) is a valid check for this
repo's actual script structure — but it is the *server's* copy of the
variable that determines what clients actually train with.

**How it's detected today**:
- `scripts/preflight_check.py` (1b): aborts if `FL_LEARNING_RATE` is unset,
  or below `1e-4` for the BERT backend — before any GPU time is spent.
- **Effective-LR verification** (`ai_client.fl_client.verify_effective_learning_rate`,
  called from all three `fit()` implementations — base `FHIRFederatedClient`,
  `AdaptiveClippingClient`, `DualLoraClient`): on rounds 1-2 (before the
  server's 40% decay kicks in for round 3+), compares the `learning_rate`
  actually received via the server's `config` dict against the *current
  process's own* `FL_LEARNING_RATE`. A mismatch here is the specific
  fingerprint of this bug: it means the **server** process didn't have
  `FL_LEARNING_RATE` exported (this client's own copy is not proof the
  server's is set too), and raises immediately rather than training a whole
  run at the wrong LR.

### Bug 3 — MAX_ADMISSIONS=2000 loaded instead of 10000

**What happened**: HAPI FHIR was loaded with `MAX_ADMISSIONS=2000` (the
Makefile's default) instead of the intended `10000`, giving ~295
examples/silo instead of the expected volume, and an epsilon computed
against the wrong effective subsampling rate (`q = batch_size / n_total`
depends on `n_total`, which was ~5x too small).

**How it's detected today**:
- `scripts/preflight_check.py` (1a): queries
  `GET /fhir/Patient?_summary=count` and aborts if it doesn't match
  `MAX_ADMISSIONS` (env var, default 10000), with the exact `make up-mimic`
  fix command in the error message.
- `scripts/preflight_check.py` (1d): independently cross-checks per-silo
  example counts via `ai_client.fhir_consumer.fetch_training_examples` and
  the `partition_id=N` tag on each example — aborts if any silo has fewer
  than 50% of the observed mean. This also catches a degenerate Dirichlet
  partition even when the *total* dataset size is correct.

## Known open item — per-layer clipping privacy accounting

`clipping_strategy=per_layer` (Article 2's adaptive clipping) has a privacy
accounting question under internal review — see internal docs (not part of
this public checklist). **Do not treat `epsilon_cumulative` from the
per_layer path as an accurate privacy guarantee** until that review
concludes. This does not affect the `baseline` (global-clip) path.

## Checklist — before running any DP experiment

1. Run `docker ps` — confirm `hapi_fhir` and `fl_server` (or the experiment's
   own server binary) are up.
2. `uv run python scripts/preflight_check.py` from the repo root (every
   `run_*.sh` script in `experiments/*/scripts/` already does this
   automatically — see below) — must print `preflight OK` before anything
   else runs.
3. Confirm `FL_LEARNING_RATE=2e-4` is in the script you're about to run
   (`grep FL_LEARNING_RATE experiments/.../scripts/run_*.sh`).
4. If running `clipping_strategy=per_layer`: read the epsilon-accounting
   caveat above before trusting its `epsilon_cumulative`.
5. If you deviate from the canonical configuration table above for a
   specific run, write the deviation and its justification into that run's
   own log directory (not just this document) so it survives independently
   of future edits here.

## Preflight wiring

`scripts/preflight_check.py` is called automatically at the start of every
single-run experiment script:
`experiments/adaptive-clipping/scripts/{run_baseline,run_per_layer,run_sigma_sweep,run_c0_sweep}.sh`
and `experiments/article3/scripts/{run_dp_lora,run_ffa_lora,run_dual_lora}.sh`.

It is intentionally **not** called directly inside the two matrix wrapper
scripts (`run_matrix.sh`, `run_matrix_a3.sh`): those scripts never export
`FL_LEARNING_RATE` themselves (each sub-run script they invoke does), so
running preflight at the matrix level would false-fail check 1b before any
sub-run's own exports are in effect. The first sub-run's own preflight call
still fails the whole matrix fast if the environment is wrong — just after
run 1 of N starts rather than before run 1 begins.

## Environment incident — root `uv sync` silently no-ops

During this work, `uv sync` run from the repo root (without `--all-packages`)
resolved successfully ("Resolved N packages... Checked in 0.00ms") but left
`.venv/` essentially empty — none of the workspace member packages (ai_client,
evaluation, etc.) were actually installed, despite no error being reported.
`uv run` against that environment appeared to work for already-cached
imports but failed on new ones. Fix: `uv sync --all-packages` from the repo
root reliably installs every workspace member. If a workspace package fails
to import unexpectedly after a plain `uv sync`, re-run with
`--all-packages` before assuming a code problem.
