"""Equalization operator E.

Given a target privacy budget eps*, find the sigma that makes a method
spend exactly eps* (within tolerance), holding every other DP-relevant knob
(R, q or silos, delta, accountant) fixed at what the caller supplies via
``template``. This is the Fair-Budget Comparison primitive: run each method
under E(method, eps*) instead of its native sigma, so the utility
comparison is made at equal privacy cost.
"""

from __future__ import annotations

from dataclasses import replace

from dp_equalization.accountant import MethodConfig, epsilon

DEFAULT_TOLERANCE = 0.01  # |eps - eps*| / eps*
DEFAULT_SIGMA_BOUNDS = (1e-3, 100.0)


def equalize(
    template: MethodConfig,
    eps_star: float,
    tol: float = DEFAULT_TOLERANCE,
    sigma_bounds: tuple[float, float] = DEFAULT_SIGMA_BOUNDS,
    max_iter: int = 100,
) -> tuple[MethodConfig, float]:
    """Bisect on sigma so that ``epsilon(template with this sigma) == eps_star``.

    ``template`` supplies every DP-accounting input except sigma (method,
    rounds, delta, sample_rate/steps_per_round or silos, accountant); its
    own ``sigma`` field is ignored. Relies on epsilon() being non-increasing
    in sigma for fixed (q/silos, R, delta) -- verified for both accountants
    in tests/test_accountant.py::test_monotonic_in_sigma.

    Returns (equalized_config, epsilon_achieved). Raises ValueError if
    eps_star is not reachable within sigma_bounds, or RuntimeError if
    bisection does not converge to `tol` within `max_iter` steps.
    """
    if eps_star <= 0:
        raise ValueError(f"eps_star must be positive, got {eps_star}")

    lo, hi = sigma_bounds
    eps_lo = epsilon(replace(template, sigma=lo))
    eps_hi = epsilon(replace(template, sigma=hi))
    if eps_lo < eps_star:
        raise ValueError(
            f"sigma_bounds[0]={lo} already only spends eps={eps_lo:.4f} < "
            f"eps*={eps_star} -- lower sigma_bounds[0] to reach this budget"
        )
    if eps_hi > eps_star:
        raise ValueError(
            f"sigma_bounds[1]={hi} still spends eps={eps_hi:.4f} > "
            f"eps*={eps_star} -- raise sigma_bounds[1] to reach this budget"
        )

    eps_mid = eps_lo
    mid = lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        eps_mid = epsilon(replace(template, sigma=mid))
        if abs(eps_mid - eps_star) / eps_star < tol:
            return replace(template, sigma=mid), eps_mid
        if eps_mid > eps_star:
            lo = mid
        else:
            hi = mid
    raise RuntimeError(
        f"equalize did not converge to within {tol:.1%} of eps*={eps_star} "
        f"after {max_iter} iterations (last sigma={mid:.6f}, eps={eps_mid:.4f})"
    )
