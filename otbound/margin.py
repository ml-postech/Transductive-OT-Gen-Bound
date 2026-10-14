"""Classifier margins and the change-rate factor :math:`M_c(f, \\phi)` of Theorem 1.

The margin of :math:`f` at :math:`(x, y)` is

.. math:: \\rho_f(\\phi(x), y) := f_y(\\phi(x)) - \\max_{y' \\neq y} f_{y'}(\\phi(x)),

so :math:`x` is misclassified exactly when :math:`\\rho_f(\\phi(x), y) \\le 0`.

Theorem 1 weights each class-``c`` Wasserstein term by

.. math::

    M_c(f, \\phi) := \\max_{i \\neq j}
        \\frac{|\\rho_f(\\phi(x_i), c) - \\rho_f(\\phi(x_j), c)|}{\\|\\phi(x_i) - \\phi(x_j)\\|},
    \\qquad i, j \\in I^{(\\pi)}_{\\mathrm{train},c} \\cup I^{(\\pi)}_{\\mathrm{test}},

the largest rate at which the class-``c`` margin can change per unit of movement in
feature space -- the transductive analogue of a Lipschitz constant of the margin.  Note
the asymmetric index set: class-``c`` *training* nodes (whose labels are observed)
together with *all* test nodes (whose labels are not), so computing :math:`M_c` never
requires a test label.

As with :mod:`otbound.wasserstein`, the quantity comes in two forms:

* :func:`max_change_class` -- the exact maximum over all pairs, for evaluation.
* :func:`soft_max_change_class` -- a temperature-smoothed log-sum-exp over a random
  subsample of pairs, differentiable, for the Section 5.2 regularizer.

The smooth form is an upper bound on the hard maximum
(:math:`\\operatorname{logsumexp} \\ge \\max`), so the surrogate never understates the
factor it replaces.
"""

import torch

__all__ = [
    "margins_true",
    "margins_all",
    "max_change_class",
    "soft_max_change_class",
    "change_rate_support",
]

# Pairs closer than this in feature space are dropped: the ratio is numerically
# meaningless there, and a node paired with itself would divide by zero.
_MIN_FEATURE_DIST = 1e-6


def margins_true(score, y):
    """Margin at the true label: ``[N]`` from scores ``[N, K]`` and labels ``[N]``."""
    N = score.shape[0]
    device = score.device

    true_scores = score.gather(1, y.unsqueeze(1)).squeeze(1)
    score_copy = score.clone()
    score_copy[torch.arange(N, device=device), y] = float("-inf")
    max_other_scores = torch.max(score_copy, dim=1)[0]
    return true_scores - max_other_scores


def margins_all(score):
    """Margin under *every* hypothetical label.

    Returns ``[N, K]`` where entry ``[i, c]`` is :math:`\\rho_f(\\phi(x_i), c)`, the
    margin node ``i`` would have if its label were ``c``.  Theorem 1 needs the class-``c``
    margin at nodes whose true label may be something else (and at test nodes, whose
    label is unknown), which is exactly what this provides.
    """
    N, K = score.shape
    device = score.device

    score_expanded = score.unsqueeze(2).expand(N, K, K)
    mask = torch.eye(K, device=device).bool().unsqueeze(0).expand(N, K, K)
    score_masked = score_expanded.clone()
    score_masked[mask] = float("-inf")

    max_other_scores = torch.max(score_masked, dim=1)[0]  # [N, K]
    return score - max_other_scores


def change_rate_support(class_mask, train_mask, test_mask):
    """The index set :math:`I^{(\\pi)}_{\\mathrm{train},c} \\cup I^{(\\pi)}_{\\mathrm{test}}`
    over which :math:`M_c` maximizes.  Defined here so that the bound and the
    regularizer cannot disagree about it.
    """
    return (class_mask & train_mask) | test_mask


def max_change_class(features, margin_all, c, batch_size=None):
    """Exact :math:`M_c(f, \\phi)` over all pairs in the given point set.

    ``features`` and ``margin_all`` must already be restricted to
    :func:`change_rate_support`; every row is treated as a candidate for both ``i`` and
    ``j``.  Evaluated in row blocks so the ``[N, N]`` pairwise matrices stay within
    memory on large graphs.

    Args:
        features: ``[N, d]`` encoded features.
        margin_all: ``[N, K]`` all-class margins.
        c: class index.
        batch_size: rows per block; a memory-aware default is chosen when ``None``.

    Returns:
        float: the maximum change rate.
    """
    N, d = features.shape
    device = features.device

    if batch_size is None:
        if device.type == "cuda":
            free_memory = (
                torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_allocated()
            )
            memory_per_row = N * d * 4  # float32
            batch_size = max(10, min(N, int(free_memory * 0.3 / max(memory_per_row, 1))))
        else:
            batch_size = min(50, N)

    margins_c = margin_all[:, c]  # [N]
    max_change = 0.0

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        block_features = features[i:end_i]
        block_margins = margins_c[i:end_i]

        feature_dists = torch.norm(
            block_features.unsqueeze(1) - features.unsqueeze(0), dim=2
        )  # [b, N]
        margin_diffs = torch.abs(
            block_margins.unsqueeze(1) - margins_c.unsqueeze(0)
        )  # [b, N]

        # Sending near-zero distances to +inf makes their rate 0, which drops them
        # from the maximum without needing a separate mask.
        feature_dists = feature_dists.masked_fill(
            feature_dists < _MIN_FEATURE_DIST, float("inf")
        )
        rates = margin_diffs / feature_dists
        rates = torch.nan_to_num(rates, nan=0.0, posinf=0.0)

        max_change = max(max_change, float(rates.max()))

        del feature_dists, margin_diffs, rates
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return max_change


def _soft_max(values, temperature):
    """Temperature-smoothed maximum, :math:`\\tau \\log \\sum \\exp(v / \\tau)`.

    Shifted by the running maximum for numerical stability.  Upper-bounds ``max``.
    """
    v_max = values.detach().max()
    return v_max + temperature * torch.logsumexp((values - v_max) / temperature, dim=0)


def soft_max_change_class(features, margin_all, c, temperature=1.0, n_sample=200,
                          generator=None):
    """Differentiable surrogate for :math:`M_c`, as used by the Section 5.2 regularizer.

    Two approximations relative to :func:`max_change_class`:

    1. the maximum is replaced by a temperature-smoothed log-sum-exp so that gradient
       reaches more than the single arg-max pair, and
    2. the pairs are drawn from a random subsample of at most ``n_sample`` nodes, since
       the full :math:`O(N^2)` set is far too large to touch every step.

    ``features`` and ``margin_all`` must already be restricted to
    :func:`change_rate_support`.

    Returns:
        torch.Tensor: a scalar, differentiable with respect to ``features`` and
        ``margin_all``.
    """
    N = features.shape[0]
    if N < 2:
        return features.new_zeros(())

    n_samp = min(n_sample, N)
    perm = torch.randperm(N, device=features.device, generator=generator)[:n_samp]
    f = features[perm]  # [n, d]
    m_c = margin_all[perm, c]  # [n]

    feature_dists = torch.cdist(f, f, p=2)
    margin_diffs = torch.abs(m_c.unsqueeze(1) - m_c.unsqueeze(0))

    valid = feature_dists >= _MIN_FEATURE_DIST
    if not valid.any():
        return features.new_zeros(())

    rates = margin_diffs / feature_dists.clamp(min=_MIN_FEATURE_DIST)
    return _soft_max(rates[valid], temperature)
