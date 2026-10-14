"""The two Wasserstein quantities tracked in the depth analysis of Section 5.3.

Propositions 2 and 3 bound how :math:`W_1` between the feature distributions of
arbitrary node subsets evolves with depth.  Theorem 1 then turns that into a trade-off,
because depth acts on two Wasserstein quantities with opposite effects on the bound:

:math:`W_C` -- intra-class concentration
    .. math::
        W_C := \\frac{1}{K} \\sum_{c=1}^{K} \\mathbb{E}_{\\pi'}\\!\\left[
            \\frac{m_c^{(\\pi')}}{m}
            W_1\\big(\\phi_\\# \\mu_{I^{(\\pi')}_{\\mathrm{train},c}},
                     \\phi_\\# \\mu_{I^{(\\pi')}_{\\mathrm{test},c}}\\big)\\right]

    This is the Theorem 1 Wasserstein term averaged over classes, so a smaller
    :math:`W_C` tightens the bound directly.

:math:`W_S` -- inter-class separation
    .. math::
        W_S := \\frac{1}{K(K-1)} \\sum_{c_1 \\neq c_2}
            W_1\\big(\\phi_\\# \\mu_{I_{c_1}}, \\phi_\\# \\mu_{I_{c_2}}\\big)

    A proxy for how far apart the classes sit.  It enters the bound indirectly: better
    separation admits a larger :math:`\\gamma` at the same :math:`R_{m,\\gamma}`, and the
    :math:`1/\\gamma` factor then tightens the bound.

Message passing shrinks both.  Figure 1 plots the two against depth alongside the
empirical error, and the opposing pulls are what make the error non-monotone in depth.
"""

from .bound import expected_wasserstein_per_class, inter_class_separation
from .wasserstein import w1_exact

__all__ = ["intra_class_concentration", "inter_class_separation"]


def intra_class_concentration(features, y, train_mask, test_mask, num_classes,
                              n_permutations=4, test_blind=False, w1_fn=w1_exact):
    """:math:`W_C`, the class-averaged expected Wasserstein term.

    Defaults to ``test_blind=False`` because the depth analysis is diagnostic rather
    than a practical estimator, so it may use every label.

    Returns:
        tuple: ``(mean, min, max)`` over classes of the per-class expectations.
    """
    per_class = expected_wasserstein_per_class(
        features, y, train_mask, test_mask, num_classes,
        n_permutations=n_permutations, test_blind=test_blind, w1_fn=w1_fn,
    )
    return float(per_class.mean()), float(per_class.min()), float(per_class.max())
