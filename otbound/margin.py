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
    "global_change_rate",
    "soft_global_change_rate",
    "soft_quantile",
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


# ============================================================================
# The global change-rate factor M(f, phi)  (Theorem 4, Appendix A)
# ============================================================================
#
# The global bound replaces the K class-wise factors with a single one,
#
#     M(f, phi) := max_{i, j, y}
#         |rho_f(phi(x_i), y_i) - rho_f(phi(x_j), y)| / ||phi(x_i) - phi(x_j)||,
#         i in I_train,  j in I_test,  y in Y.
#
# Note the asymmetry between the two sides, which is what makes this computable
# without test labels: the training node contributes the margin at its *observed*
# label y_i, while the test node contributes every hypothetical label y in Y.
# Because Y is finite, enumerating y gives a valid upper bound.

def _global_rate_blocks(features, margin_all, y, train_mask, test_mask, batch_size=None):
    """Yield blocks of global change rates, shaped ``[b, n_test, K]``.

    Blocked over training nodes so the ``[n_train, n_test, K]`` tensor is never
    materialized in full.
    """
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return

    train_features, test_features = features[train_idx], features[test_idx]
    train_margins, test_margins = margin_all[train_idx], margin_all[test_idx]
    train_labels = y.squeeze()[train_idx]

    n_train, n_test, K = len(train_idx), len(test_idx), margin_all.shape[1]
    if batch_size is None:
        # Keep the per-block tensor near 100MB in float32.
        batch_size = max(1, min(n_train, int(1e8 / max(n_test * K * 4, 1))))

    # The margin of each training node at its own observed label: [n_train, 1].
    train_margin_at_label = train_margins.gather(1, train_labels.unsqueeze(1))

    for i in range(0, n_train, batch_size):
        end = min(i + batch_size, n_train)
        block_features = train_features[i:end]
        block_margin = train_margin_at_label[i:end]

        dists = torch.norm(
            block_features.unsqueeze(1) - test_features.unsqueeze(0), dim=2
        )  # [b, n_test]
        # Near-duplicate pairs would divide by ~0; a huge denominator sends their
        # rate to ~0 so they neither win the max nor distort the quantile.
        dists = dists.masked_fill(dists < _MIN_FEATURE_DIST, 1e10)

        # [b, 1, 1] - [1, n_test, K] -> [b, n_test, K]
        margin_diffs = torch.abs(
            block_margin.unsqueeze(2) - test_margins.unsqueeze(0)
        )
        yield margin_diffs / dists.unsqueeze(2)


def global_change_rate(features, margin_all, y, train_mask, test_mask, quantile=1.0,
                       batch_size=None, per_block_samples=5000, max_samples=50000,
                       generator=None):
    """:math:`M(f, \\phi)` of Theorem 4, or a high percentile of it.

    Args:
        quantile: ``1.0`` returns the exact maximum over all ``(i, j, y)``, which is
            :math:`M(f, \\phi)` as Theorem 4 states it.  A value below 1 returns that
            percentile of the change-rate distribution instead.  **The experiments in
            Appendix D use 0.9**: the maximum is attained by a handful of outlier pairs
            and tracks generalization less well, so the reported quantity is a proxy for
            :math:`M(f, \\phi)` rather than a valid upper bound on it.
        batch_size: training nodes per block; chosen from memory when ``None``.
        per_block_samples, max_samples: the rate distribution has
            ``n_train * n_test * K`` entries, far too many to hold, so percentiles are
            taken over a random subsample -- at most ``per_block_samples`` per block and
            ``max_samples`` overall.  Ignored when ``quantile == 1.0``.
        generator: optional :class:`torch.Generator` for reproducible subsampling.

    Returns:
        float: the requested statistic, or 0.0 if either side is empty.
    """
    max_rate = 0.0
    reservoir = []

    for rates in _global_rate_blocks(features, margin_all, y, train_mask, test_mask,
                                     batch_size):
        max_rate = max(max_rate, float(rates.max()))

        if quantile < 1.0:
            flat = rates.flatten().detach()
            if len(flat) > 2 * per_block_samples:
                keep = torch.randperm(len(flat), device=flat.device,
                                      generator=generator)[:per_block_samples]
                flat = flat[keep]
            reservoir.append(flat.cpu())

        del rates
        if features.device.type == "cuda":
            torch.cuda.empty_cache()

    if quantile >= 1.0:
        return max_rate
    if not reservoir:
        return 0.0

    pooled = torch.cat(reservoir, dim=0)
    if len(pooled) > max_samples:
        keep = torch.randperm(len(pooled), generator=generator)[:max_samples]
        pooled = pooled[keep]
    return float(torch.quantile(pooled, quantile))


def soft_quantile(values, quantile, temperature=1.0):
    """Differentiable approximation of a quantile of ``values``.

    The ``q``-th quantile is the largest of the smallest ``ceil(q n)`` values, so a
    soft-max over that bottom slice approximates it.  The slice is selected under
    ``no_grad`` and then indexed out of the gradient-carrying tensor, so gradient
    reaches the selected values -- concentrated, as it should be, on those near the
    quantile boundary.

    ``quantile >= 1`` reduces to :func:`_soft_max` over everything.
    """
    if quantile >= 1.0:
        return _soft_max(values, temperature)

    k = max(1, int(torch.ceil(torch.tensor(quantile * len(values))).item()))
    with torch.no_grad():
        _, bottom_idx = torch.topk(values.detach(), k, largest=False)
    return _soft_max(values[bottom_idx], temperature)


def soft_global_change_rate(train_features, train_margin_all, train_y,
                            test_features, test_margin_all, quantile=0.9,
                            temperature=1.0, n_sample=200, generator=None):
    """Differentiable surrogate for :math:`M(f, \\phi)`, for the global regularizer.

    Mirrors :func:`global_change_rate` -- the training side contributes the margin at
    its observed label, the test side every hypothetical label -- with two
    approximations, as in :func:`soft_max_change_class`: the order statistic is smoothed
    (:func:`soft_quantile`) and the pairs come from a random subsample of at most
    ``n_sample`` points per side.

    Returns:
        torch.Tensor: a scalar, differentiable with respect to the features and margins.
    """
    n_train, n_test = train_features.shape[0], test_features.shape[0]
    if n_train == 0 or n_test == 0:
        return train_features.new_zeros(())

    device = train_features.device
    tr = torch.randperm(n_train, device=device, generator=generator)[:min(n_sample, n_train)]
    te = torch.randperm(n_test, device=device, generator=generator)[:min(n_sample, n_test)]

    f_tr, f_te = train_features[tr], test_features[te]
    # Margin of each sampled training point at its own label: [n_tr, 1].
    m_tr = train_margin_all[tr].gather(1, train_y[tr].unsqueeze(1))
    m_te = test_margin_all[te]  # [n_te, K]

    dists = torch.cdist(f_tr, f_te, p=2).clamp(min=_MIN_FEATURE_DIST)  # [n_tr, n_te]
    margin_diffs = torch.abs(m_tr.unsqueeze(2) - m_te.unsqueeze(0))    # [n_tr, n_te, K]
    rates = margin_diffs / dists.unsqueeze(2)

    return soft_quantile(rates.reshape(-1), quantile, temperature)
