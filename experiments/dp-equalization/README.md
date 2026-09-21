# dp-equalization

Article 2 (privacy-budget equalization) experiment: the equalization
operator **E**. Given a target privacy budget `eps*`, `E` reparameterizes a
DP-FL method's `sigma` so it spends exactly `eps*` (within tolerance),
holding every other DP-relevant knob (rounds, sample rate / silo sizes,
delta) fixed — so methods can be compared at equal privacy cost instead of
equal `sigma`.

Built per `docs/EQUALIZATION_READINESS.md` (answers C, D.4, E): the
accountants are reused as plain library calls, not reimplemented, so paper
`eps` numbers never drift from what the training code itself reports.

## Contents

- `src/dp_equalization/accountant.py` — `epsilon(config) -> float`. Wraps
  `ai_client.fl_client._compute_cumulative_epsilon` (record-level RDP) and
  `client_level_dp.accountant.client_level_epsilon` (client-level RDP).
  No training is run; both are pure post-hoc accounting calls.
- `src/dp_equalization/equalize.py` — `equalize(template, eps_star) ->
  (config, epsilon_achieved)`. Bisects on `sigma`.
- `src/dp_equalization/registry.py` — `save_equalization_map(entries, path)`
  writes the original→equalized mapping to JSON.
- `scripts/build_equalization_map.py` — builds `results/equalization_map.json`
  for the methods currently equalizable without new code (record-level
  FedProx/FedAvg DP-SGD, client-level DP), at the provisional `eps* in {1, 4,
  8}` grid.
- `tests/` — accountant regression test against HERALD's known logged
  epsilons (305.92 / 55.70 / 17.37 for sigma 0.5/1.0/2.0, R=20), a
  monotonicity test, and equalize() tolerance tests.

## Usage

```bash
cd experiments/dp-equalization
uv sync
uv run pytest
uv run python scripts/build_equalization_map.py
```

```python
from dp_equalization import MethodConfig, epsilon, equalize

template = MethodConfig(
    method="fedprox_dpsgd_record_level",
    sigma=1.0,  # placeholder; equalize() replaces it
    rounds=20,
    delta=1e-5,
    silos=((155, 1550), (150, 1508), (151, 1518), (152, 1527), (149, 1492)),
    accountant="rdp",
)
cfg, eps_achieved = equalize(template, eps_star=4.0)
```

## Known gaps (see `docs/EQUALIZATION_READINESS.md` for detail)

- Per-layer adaptive clipping's group-composed accountant
  (`experiments/adaptive-clipping/analysis/renyi_group_composition.py`)
  is not wired in here — it lives outside that experiment's installable
  package (`analysis/`, not `src/adaptive_clipping/`), so pulling it in
  would mean repackaging code outside this task's scope. Adding it is a
  small follow-up (new `accountant="rdp_group_composed"` branch) once that
  method enters the equalization matrix.
- PATE's Confident-GNMax accountant (`herald_pate.gnmax_accountant`) is a
  materially different mechanism (data-dependent, not (sigma, q, R)-shaped)
  and is out of scope for this operator as-is.
