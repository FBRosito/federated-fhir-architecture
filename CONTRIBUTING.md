# Contributing to HERALD

Thank you for your interest in contributing! This document describes the
code standards and the pull-request process.

## Ground rules

- **Never commit data or credentials.** MIMIC-IV files, FHIR bundles,
  checkpoints, `.env` files and API keys are all covered by `.gitignore` —
  do not force-add them. See [SECURITY.md](SECURITY.md).
- All code, comments, docstrings, and commit messages are written in **English**.
- Keep changes minimal and focused; do not mix refactors with features.

## Development setup

```bash
git clone https://github.com/FBRosito/federated-fhir-architecture.git
cd federated-fhir-architecture
uv sync                      # installs the workspace (see pyproject.toml)
pip install -r requirements-dev.txt
pre-commit install           # optional but recommended
```

## Code style

Enforced in CI (`.github/workflows/ci.yml`) and via pre-commit:

- **black** (line length 88) — `black .`
- **isort** (black profile) — `isort --profile black .`
- **flake8** (`.flake8`: max line length 120, E203/W503 ignored) — `flake8 .`
- Every public function/class must have a Google-style docstring
  (description, `Args:`, `Returns:`, `Raises:` where applicable) and full
  type hints on parameters and return values.
- Use `logging` for diagnostics; `print()` is reserved for CLI output that
  is the program's deliverable.
- No `TODO`/`FIXME`/`HACK` markers in merged code — open an issue instead.

## Tests

- The offline test suite lives in `experiments/adaptive-clipping/tests/`
  and runs in CI. Run it locally:

  ```bash
  cd experiments/adaptive-clipping
  uv run pytest tests/
  ```

- Add tests for any behavior change in a package that already has tests.

## Pull requests

1. Branch from `main`; keep the branch small and topical.
2. Ensure `black --check .`, `isort --check-only .`, `flake8 .`, and the
   test suite pass locally.
3. Describe *what* changed and *why*; link related issues.
4. Maintainers review; squash-merge is preferred.

By contributing, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md) and
that your contributions are licensed under the [MIT License](LICENSE).
