#!/usr/bin/env python3
"""
renyi_group_composition.py
---------------------------
Corrected per-layer epsilon accounting for G independent per-group Gaussian
releases sharing the same subsampled batch each step.

Renyi divergence adds across independent mechanisms: RDP_total(alpha) =
G * RDP_single(alpha), where RDP_single is Opacus's own subsampled-Gaussian
RDP curve (opacus.accountants.analysis.rdp.compute_rdp) at the REAL sigma —
not a substitution of sigma -> sigma/sqrt(G) into compute_rdp's sigma
argument. That substitution is only exact for the non-subsampled Gaussian
mechanism (RDP = alpha/(2*sigma^2)); Opacus's subsampled RDP is a
non-linear function of (q, sigma) jointly, so substituting sigma_eff does
not correctly propagate the G-fold composition through the subsampling
term. Verified numerically: substituting sigma_eff gives ~4x the value
produced by multiplying the RDP curve directly (see docs/per_layer_epsilon_impact.md).

Usage:
    cd experiments/adaptive-clipping && uv run python -m analysis.renyi_group_composition
"""

from __future__ import annotations

from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

DEFAULT_ALPHAS = (
    [1 + x / 100.0 for x in range(1, 100)] + list(range(2, 64)) + [128, 256, 512, 1024]
)


def group_composed_epsilon(
    q: float,
    noise_multiplier: float,
    steps: int,
    delta: float,
    n_groups: int = 1,
    alphas: list[float] | None = None,
) -> tuple[float, float]:
    """Corrected (epsilon, best_alpha) for n_groups independent Gaussian
    releases per step, each with the same (q, noise_multiplier, steps).

    n_groups=1 reduces exactly to Opacus's own single-mechanism accounting
    (sanity-checked in tests/test_renyi_group_composition.py).
    """
    orders = alphas if alphas is not None else DEFAULT_ALPHAS
    rdp_single = compute_rdp(
        q=q, noise_multiplier=noise_multiplier, steps=steps, orders=orders
    )
    rdp_total = [n_groups * r for r in rdp_single]
    eps, alpha = get_privacy_spent(orders=orders, rdp=rdp_total, delta=delta)
    return eps, alpha


def main() -> None:
    """Print the RDP group-composition privacy analysis for the HERALD silo table."""
    G = 13
    delta = 1e-5
    silos = [
        (0, 155, 1550),
        (1, 150, 1508),
        (2, 151, 1518),
        (3, 152, 1527),
        (4, 149, 1492),
    ]
    R = 20

    for sigma in (1.0, 2.0):
        print(f"=== sigma={sigma} (G={G}) ===")
        epsilons = []
        for silo, k, n in silos:
            q = k / n
            steps = k * R
            eps, alpha = group_composed_epsilon(q, sigma, steps, delta, n_groups=G)
            epsilons.append(eps)
            print(
                f"  silo{silo}: q={q:.4f} steps={steps} -> eps={eps:.4f} (alpha={alpha})"
            )
        mean_eps = sum(epsilons) / len(epsilons)
        print(
            f"  mean={mean_eps:.4f}  range=[{min(epsilons):.4f}, {max(epsilons):.4f}]"
        )
        print()


if __name__ == "__main__":
    main()
