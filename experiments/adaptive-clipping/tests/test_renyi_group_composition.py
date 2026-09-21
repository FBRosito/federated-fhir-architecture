from analysis.renyi_group_composition import group_composed_epsilon
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

Q = 0.0995
SIGMA = 1.0
STEPS = 3020
DELTA = 1e-5


def test_g1_matches_plain_opacus_accounting():
    """n_groups=1 must reduce exactly to Opacus's own single-mechanism result."""
    eps_group, _ = group_composed_epsilon(Q, SIGMA, STEPS, DELTA, n_groups=1)

    from analysis.renyi_group_composition import DEFAULT_ALPHAS

    rdp = compute_rdp(q=Q, noise_multiplier=SIGMA, steps=STEPS, orders=DEFAULT_ALPHAS)
    eps_plain, _ = get_privacy_spent(orders=DEFAULT_ALPHAS, rdp=rdp, delta=DELTA)

    assert eps_group == eps_plain


def test_g13_is_less_than_naive_sigma_substitution():
    """Regression guard against the sigma_eff=sigma/sqrt(G) substitution bug:
    true G-fold RDP additivity must give a smaller (correct) epsilon than
    the invalid substitution for this subsampled regime."""
    import math

    eps_correct, _ = group_composed_epsilon(Q, SIGMA, STEPS, DELTA, n_groups=13)

    from analysis.renyi_group_composition import DEFAULT_ALPHAS

    sigma_eff = SIGMA / math.sqrt(13)
    rdp_eff = compute_rdp(
        q=Q, noise_multiplier=sigma_eff, steps=STEPS, orders=DEFAULT_ALPHAS
    )
    eps_invalid_substitution, _ = get_privacy_spent(
        orders=DEFAULT_ALPHAS, rdp=rdp_eff, delta=DELTA
    )

    assert eps_correct < eps_invalid_substitution


def test_g13_matches_known_corrected_value_sigma1():
    """Pins the corrected sigma=1.0 per-layer epsilon (representative silo,
    q=0.0995, steps=3020) to the value documented in
    docs/per_layer_epsilon_impact.md after the Method A correction."""
    eps, _ = group_composed_epsilon(Q, SIGMA, STEPS, DELTA, n_groups=13)
    assert abs(eps - 402.80) < 0.5
