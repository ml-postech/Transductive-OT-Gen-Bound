"""The Section 5.2 training regularizer: a differentiable surrogate for Theorem 1.

Setting :math:`\\gamma = 1` and keeping only the model-dependent term of Theorem 1
gives the penalty added to the task loss, :math:`\\mathcal{L} = \\mathcal{L}_{\\mathrm{CE}}
+ \\lambda R`, with

.. math::

    R := \\sum_{c=1}^{K} M_c(f, \\phi)\\,
        \\mathbb{E}_{\\pi'}\\!\\left[ \\frac{m_c^{(\\pi')}}{m}\\,
            W_1\\!\\big( \\phi_\\# \\mu_{I^{(\\pi')}_{\\mathrm{train},c}},\\,
                        \\phi_\\# \\mu_{I^{(\\pi')}_{\\mathrm{test},c}} \\big) \\right].

The bound itself is not differentiable, so two pieces are replaced by smooth
surrogates, and the expectation is estimated cheaply enough to afford every step:

=========================  =====================================================
:math:`W_1`                entropy-regularized Sinkhorn (Cuturi, 2013),
                           :func:`otbound.wasserstein.sinkhorn_distance`
:math:`\\max` in :math:`M_c`  temperature-smoothed log-sum-exp,
                           :func:`otbound.margin.soft_max_change_class`
:math:`\\mathbb{E}_{\\pi'}`   a **single class-stratified** :math:`\\pi'` sample per step,
                           :func:`otbound.splits.stratified_class_split`
=========================  =====================================================

Notes on the third row, which is where this estimator departs from
:mod:`otbound.bound`:

*One sample, not* ``T = 4``.
    The bound can afford ``T`` permutations because it is evaluated once per model; the
    regularizer is evaluated every gradient step.  A fresh :math:`\\pi'` is drawn each
    step, so over training the penalty averages over thousands of permutations even
    though each step sees one.

*Stratified, not global.*
    A global permutation makes :math:`m_c^{(\\pi')}` fluctuate; stratifying within each
    class fixes it.  Both estimate the same expectation, and stratification is the
    standard variance reduction -- which is what a single-sample estimator needs.

*The split ratio is not a free parameter.*
    The two sides are cut at :math:`m : u`, the same ratio the bound uses, because
    empirical :math:`W_1` depends on the sample sizes of the two measures being
    compared.  Splitting a class in half instead would estimate a different quantity,
    and the discrepancy would vary by class, so :math:`\\lambda` could not absorb it.

Only observed labels are read, so the regularizer is test-blind like the OT bound: the
:math:`\\pi'` split is drawn inside :math:`I^{(\\pi)}_{\\mathrm{train}}`.  Unlabelled
features still enter, through the :math:`M_c` term.

The same function serves the graph, tabular and image experiments; it takes the two
sides already sliced, so mini-batch callers can pass a batch and semi-supervised
callers can pass a buffer (see :mod:`otbound.ssl`).

Two variations on the above live here as well:

:func:`global_ot_regularizer`
    the surrogate of Theorem 4 instead of Theorem 1 (Appendix D.2) -- one transport term
    between the training and test distributions as wholes, with no class decomposition
    and so no :math:`\\pi'` sampling at all.

``terms`` and ``n_permutations`` on :func:`ot_regularizer`
    ``terms`` keeps one of the two factors and drops the other, which is the ablation of
    Appendix F.5; ``n_permutations`` raises :math:`T` above the default of 1, which is
    what Appendix F.6 sweeps.
"""

import torch

from .margin import soft_global_change_rate, soft_max_change_class
from .splits import stratified_class_split, train_ratio_of
from .wasserstein import sinkhorn_distance

__all__ = ["ot_regularizer", "ot_regularizer_from_masks", "global_ot_regularizer",
           "REGULARIZER_TERMS"]

#: Which factors of R to keep, for the ablation of Appendix F.5.
REGULARIZER_TERMS = ("full", "transport", "margin")


def ot_regularizer(train_features, train_margin_all, train_y,
                   test_features, test_margin_all,
                   num_classes, train_ratio, terms="full", n_permutations=1,
                   sinkhorn_eps=0.05, sinkhorn_iter=50, soft_temp=1.0,
                   n_sample=200, min_class_size=4, generator=None):
    """Compute :math:`R` from already-sliced labelled and unlabelled tensors.

    Args:
        train_features: ``[m, d]`` encoded features of the labelled points.
        train_margin_all: ``[m, K]`` all-class margins of the labelled points.
        train_y: ``[m]`` observed labels.
        test_features: ``[u, d]`` encoded features of the unlabelled points.  Used only
            for :math:`M_c`; no label of theirs is read.
        test_margin_all: ``[u, K]`` all-class margins of the unlabelled points.
        num_classes: :math:`K`.
        train_ratio: :math:`m / (m + u)`, the ratio at which :math:`\\pi'` cuts each
            class.  Pass the ratio of the *full* split even when the tensors above are
            a mini-batch, so that batching does not change the estimand.
        terms: which factors to keep, for the ablation of Appendix F.5 --
            ``"full"`` for :math:`R = \\sum_c M_c \\mathbb{E}_{\\pi'}[\\frac{m_c}{m} W_1]`,
            ``"transport"`` for :math:`R_t = \\sum_c \\mathbb{E}_{\\pi'}[\\frac{m_c}{m} W_1]`
            (drops the margin-sensitivity factor), or
            ``"margin"`` for :math:`R_m = \\sum_c M_c` (drops the transport factor).
            The unused factor is not computed at all, so the ablations are also cheaper.
        n_permutations: number of stratified :math:`\\pi'` samples averaged per step,
            :math:`T`.  The default of 1 is what the paper uses (Appendix H.1); since a
            fresh :math:`\\pi'` is drawn every step, training still averages over many.
            Appendix F.6 sweeps this.
        sinkhorn_eps, sinkhorn_iter: Sinkhorn entropic regularization and iteration count.
        soft_temp: temperature of the log-sum-exp standing in for the max in :math:`M_c`.
        n_sample: cap on the number of points sampled for the :math:`M_c` pair set.
        min_class_size: classes with fewer labelled points than this are skipped --
            an empirical :math:`W_1` between a handful of points is noise.  Applied to
            every ``terms`` setting alike, so the three variants sum over the same
            classes and stay comparable.
        generator: optional :class:`torch.Generator` for reproducible sampling.

    Returns:
        torch.Tensor: the scalar :math:`R`, differentiable with respect to the features
        and margins.
    """
    if terms not in REGULARIZER_TERMS:
        raise ValueError(
            "terms must be one of {}; got {!r}".format(", ".join(REGULARIZER_TERMS), terms)
        )

    n_train = train_features.shape[0]
    if n_train == 0:
        return train_features.new_zeros(())

    class_terms = []
    for c in range(num_classes):
        class_idx = (train_y == c).nonzero(as_tuple=True)[0]
        if int(class_idx.numel()) < min_class_size:
            continue

        # --- E_{pi'}[ (m_c/m) * W1(train_c, test_c) ], averaged over T samples ---
        transport_c = None
        if terms != "margin":
            samples = []
            for _ in range(max(1, n_permutations)):
                h1, h2 = stratified_class_split(class_idx, train_ratio,
                                                generator=generator)
                if h1.numel() < 1 or h2.numel() < 1:
                    continue
                w1_c = sinkhorn_distance(
                    train_features[h1], train_features[h2],
                    epsilon=sinkhorn_eps, max_iter=sinkhorn_iter,
                )
                # m_c^{(pi')} / m: the size of the pi'-train side, not of the whole class.
                samples.append((float(h1.numel()) / float(n_train)) * w1_c)
            if not samples:
                continue
            transport_c = torch.stack(samples).mean()

        # --- M_c over (class-c labelled points) union (all unlabelled points) ---
        mc_c = None
        if terms != "transport":
            support_features = torch.cat([train_features[class_idx], test_features], dim=0)
            support_margins = torch.cat(
                [train_margin_all[class_idx], test_margin_all], dim=0
            )
            mc_c = soft_max_change_class(
                support_features, support_margins, c,
                temperature=soft_temp,
                n_sample=min(n_sample, support_features.shape[0]),
                generator=generator,
            )

        if terms == "full":
            class_terms.append(mc_c * transport_c)
        elif terms == "transport":
            class_terms.append(transport_c)
        else:  # "margin"
            class_terms.append(mc_c)

    if not class_terms:
        return train_features.new_zeros(())
    return torch.stack(class_terms).sum()


def global_ot_regularizer(train_features, train_margin_all, train_y,
                          test_features, test_margin_all,
                          quantile=0.9, sinkhorn_eps=0.05, sinkhorn_iter=50,
                          soft_temp=1.0, n_sample=200, generator=None):
    """The global bound as a regularizer (Theorem 4, Appendix D.2).

    .. math::

        R_{\\mathrm{global}} := M(f, \\phi)\,
            W_1\\!\\big( \\phi_\\# \\mu_{I^{(\\pi)}_{\\mathrm{train}}},\\,
                      \\phi_\\# \\mu_{I^{(\\pi)}_{\\mathrm{test}}} \\big),
        \\qquad \\gamma = 1.

    One transport term between the training and test feature distributions as wholes,
    with no class decomposition and so no :math:`\\pi'` sampling at all.  The same two
    surrogates as :func:`ot_regularizer`: Sinkhorn for :math:`W_1`, and a smoothed order
    statistic for :math:`M(f, \\phi)`.

    Args:
        quantile: order statistic of the change-rate distribution standing in for the
            maximum.  Appendix D.2 uses **0.9**; see
            :func:`otbound.margin.global_change_rate` on why that is a proxy for
            :math:`M(f, \\phi)` rather than an upper bound on it.

    Returns:
        torch.Tensor: the scalar :math:`R_{\\mathrm{global}}`.
    """
    if train_features.shape[0] < 2 or test_features.shape[0] < 2:
        return train_features.new_zeros(())

    w1 = sinkhorn_distance(
        train_features, test_features, epsilon=sinkhorn_eps, max_iter=sinkhorn_iter,
    )
    change_rate = soft_global_change_rate(
        train_features, train_margin_all, train_y,
        test_features, test_margin_all,
        quantile=quantile, temperature=soft_temp, n_sample=n_sample,
        generator=generator,
    )
    return change_rate * w1


def ot_regularizer_from_masks(features, margin_all, y, train_mask, test_mask,
                              num_classes, variant="classwise", train_ratio=None,
                              **kwargs):
    """Mask-based wrapper for full-batch transductive training.

    Slices the two sides out of the full-graph tensors and dispatches to either
    regularizer:

    ``variant="classwise"``
        :func:`ot_regularizer` -- Theorem 1, the one used in Table 2.  Accepts ``terms``
        and ``n_permutations``.
    ``variant="global"``
        :func:`global_ot_regularizer` -- Theorem 4, Appendix D.2.  Accepts ``quantile``.

    ``train_ratio`` defaults to :math:`m / (m + u)` computed from the two masks, which is
    what the transductive graph experiments want.  It is unused by the global variant,
    which draws no :math:`\\pi'`.
    """
    labeled = (features[train_mask], margin_all[train_mask], y.squeeze()[train_mask])
    unlabeled = (features[test_mask], margin_all[test_mask])

    if variant == "global":
        return global_ot_regularizer(*labeled, *unlabeled, **kwargs)
    if variant != "classwise":
        raise ValueError(
            "variant must be 'classwise' or 'global'; got {!r}".format(variant)
        )

    if train_ratio is None:
        train_ratio = train_ratio_of(train_mask, test_mask)
    return ot_regularizer(*labeled, *unlabeled, num_classes=num_classes,
                          train_ratio=train_ratio, **kwargs)
