"""The optimal transport generalization bound of Theorem 1, and its two estimators.

Theorem 1 bounds the transductive generalization gap by

.. math::

    R_u(f \\circ \\phi; \\pi) - R_{m,\\gamma}(f \\circ \\phi; \\pi)
    \;\\le\; \\frac{1}{\\gamma} \\sum_{c=1}^{K} M_c(f, \\phi)\\,
        \\mathbb{E}_{\\pi'}\\!\\left[ \\frac{m_c^{(\\pi')}}{m}\\,
            W_1\\!\\big( \\phi_\\# \\mu_{I^{(\\pi')}_{\\mathrm{train},c}},\\,
                        \\phi_\\# \\mu_{I^{(\\pi')}_{\\mathrm{test},c}} \\big) \\right]
      + \\mathbb{E}_{\\pi'}\\!\\left[ \\sum_{c=1}^{K}
            \\Big( \\tfrac{u_c^{(\\pi')}}{u} - \\tfrac{m_c^{(\\pi')}}{m} \\Big) \\right]
      + \\varepsilon_\\delta .

What this module computes is the first (model-dependent) term, which is what the
correlation experiments rank models by.  The second term measures class-proportion
imbalance between the splits and vanishes for class-balanced splits; :math:`\\varepsilon_\\delta`
is model-agnostic.  Neither can change the ranking of architectures on a fixed dataset,
so both are reported separately -- see :func:`imbalance_term` and :func:`epsilon_delta`.

Following Appendix B, the expectation over :math:`\\pi'` is estimated by averaging over
``n_permutations`` samples (the paper uses ``T = 4``) in one of two ways:

**OT-oracle**
    All labels are assumed available.  A reference value, not usable in practice.

**OT bound**
    Test-blind.  Nodes of :math:`I^{(\\pi)}_{\\mathrm{test}}` are removed from both sides
    of the :math:`\\pi'` split before the class-wise intersection, so only observed
    labels are ever touched.

Both share :mod:`otbound.splits` for the :math:`\\pi'` draw and :mod:`otbound.margin` for
:math:`M_c`; they differ by one argument.
"""

import numpy as np
import torch

from .margin import change_rate_support, max_change_class
from .splits import classwise_index_sets, sample_pi_prime
from .wasserstein import w1_exact

__all__ = [
    "max_change_per_class",
    "expected_wasserstein_per_class",
    "ot_bound",
    "imbalance_term",
    "epsilon_delta",
    "inter_class_separation",
    "ot_bound_subsampled",
]


def _class_mask(y, c):
    return y.squeeze() == c


def max_change_per_class(features, margin_all, y, train_mask, test_mask, num_classes,
                         batch_size=None):
    """:math:`M_c(f, \\phi)` for every class, as a ``[K]`` tensor.

    Computed once per model and reused by both estimators: :math:`M_c` depends on the
    data split :math:`\\pi` but not on :math:`\\pi'`.
    """
    out = []
    for c in range(num_classes):
        support = change_rate_support(_class_mask(y, c), train_mask, test_mask)
        if int(support.sum()) < 2:
            out.append(0.0)
            continue
        out.append(
            max_change_class(features[support], margin_all[support], c, batch_size)
        )
    return torch.tensor(out)


def expected_wasserstein_per_class(features, y, train_mask, test_mask, num_classes,
                                   n_permutations=4, test_blind=True, w1_fn=w1_exact,
                                   max_resample=3, return_all=False):
    """Estimate :math:`\\mathbb{E}_{\\pi'}[(m_c^{(\\pi')}/m) W_1(\\cdot, \\cdot)]` per class.

    Args:
        features: ``[N, d]`` encoded features :math:`\\phi(x_i)`.
        y: ``[N]`` labels (only entries under ``train_mask`` are read when
            ``test_blind`` is true).
        train_mask, test_mask: the data split :math:`\\pi`.
        num_classes: :math:`K`.
        n_permutations: number of :math:`\\pi'` samples, :math:`T` in Appendix B.
        test_blind: ``True`` for the OT bound, ``False`` for OT-oracle.
        w1_fn: the :math:`W_1` estimator; :func:`otbound.wasserstein.w1_exact` by default.
        max_resample: a :math:`\\pi'` draw that leaves some class empty on either side
            carries no information for that class, so it is discarded and redrawn up to
            this many times.  With the class sizes in the paper's datasets this never
            triggers.
        return_all: also return the per-permutation values, ``[T, K]``.

    Returns:
        ``[K]`` tensor of per-class expectations (and the ``[T, K]`` matrix if
        ``return_all``).
    """
    device = features.device
    pool_mask = train_mask | test_mask
    num_train = int(train_mask.sum().item())
    known_mask = train_mask if test_blind else None

    features_np = features.detach().cpu().numpy()
    class_masks = [_class_mask(y, c) for c in range(num_classes)]

    per_permutation = []
    attempts = 0
    seed = 0
    while len(per_permutation) < n_permutations and attempts < n_permutations * (1 + max_resample):
        attempts += 1
        pi_train, pi_test = sample_pi_prime(pool_mask, num_train, device, seed=seed)
        seed += 1

        values, usable = [], True
        for c in range(num_classes):
            train_c, test_c = classwise_index_sets(
                class_masks[c], pi_train, pi_test, known_mask
            )
            n_train_c, n_test_c = int(train_c.sum()), int(test_c.sum())
            if n_train_c == 0 or n_test_c == 0:
                usable = False
                break
            w1 = w1_fn(
                features_np[train_c.cpu().numpy()], features_np[test_c.cpu().numpy()]
            )
            values.append((n_train_c / num_train) * w1)

        if usable:
            per_permutation.append(values)

    if not per_permutation:
        raise RuntimeError(
            "every sampled permutation left some class empty; the dataset may have a "
            "class too small for the requested split"
        )

    all_values = torch.tensor(per_permutation, dtype=torch.float)  # [T, K]
    expectation = all_values.mean(0)
    return (expectation, all_values) if return_all else expectation


def ot_bound(features, margin_all, y, train_mask, test_mask, num_classes,
             n_permutations=4, gamma=1.0, w1_fn=w1_exact, batch_size=None):
    """Both estimators of the Theorem 1 Wasserstein term, from one forward pass.

    Returns a dict with:

    ``oracle``
        the OT-oracle value :math:`\\frac{1}{\\gamma}\\sum_c M_c \\,\\mathbb{E}_{\\pi'}[\\cdot]`,
    ``test_blind``
        the OT bound value, the practical estimator used in the paper's tables,
    ``max_change``
        the ``[K]`` vector of :math:`M_c`,
    ``expected_w1_oracle`` / ``expected_w1_test_blind``
        the ``[K]`` vectors of per-class expectations, which the depth analysis reads as
        :math:`W_C` (see :func:`otbound.depth.intra_class_concentration`).
    """
    max_change = max_change_per_class(
        features, margin_all, y, train_mask, test_mask, num_classes, batch_size
    )

    common = dict(
        features=features, y=y, train_mask=train_mask, test_mask=test_mask,
        num_classes=num_classes, n_permutations=n_permutations, w1_fn=w1_fn,
    )
    exp_oracle = expected_wasserstein_per_class(test_blind=False, **common)
    exp_blind = expected_wasserstein_per_class(test_blind=True, **common)

    return {
        "oracle": float((max_change * exp_oracle).sum() / gamma),
        "test_blind": float((max_change * exp_blind).sum() / gamma),
        "max_change": max_change,
        "expected_w1_oracle": exp_oracle,
        "expected_w1_test_blind": exp_blind,
    }


def inter_class_separation(features, y, num_classes, w1_fn=w1_exact):
    """:math:`W_S := \\frac{1}{K(K-1)} \\sum_{c_1 \\neq c_2} W_1(\\phi_\\# \\mu_{I_{c_1}},
    \\phi_\\# \\mu_{I_{c_2}})`, the inter-class separation proxy of Section 5.3.

    Returns the mean over unordered class pairs, together with the min and max.
    """
    features_np = features.detach().cpu().numpy()
    distances = []
    for c1 in range(num_classes):
        for c2 in range(c1 + 1, num_classes):
            m1 = _class_mask(y, c1).cpu().numpy()
            m2 = _class_mask(y, c2).cpu().numpy()
            if m1.sum() == 0 or m2.sum() == 0:
                continue
            distances.append(w1_fn(features_np[m1], features_np[m2]))
    if not distances:
        return 0.0, 0.0, 0.0
    return float(np.mean(distances)), float(np.min(distances)), float(np.max(distances))


def imbalance_term(y, train_mask, test_mask, num_classes, n_permutations=4):
    """The second term of Theorem 1,
    :math:`\\mathbb{E}_{\\pi'}[\\sum_c (u_c^{(\\pi')}/u - m_c^{(\\pi')}/m)]`.

    Vanishes for class-balanced splits and does not depend on the model, so it plays no
    role in ranking architectures; reported for completeness.
    """
    device = train_mask.device
    pool_mask = train_mask | test_mask
    num_train = int(train_mask.sum().item())
    num_test = int(test_mask.sum().item())

    totals = []
    for t in range(n_permutations):
        pi_train, pi_test = sample_pi_prime(pool_mask, num_train, device, seed=t)
        total = 0.0
        for c in range(num_classes):
            cm = _class_mask(y, c)
            total += float((cm & pi_test).sum()) / num_test - float((cm & pi_train).sum()) / num_train
        totals.append(total)
    return float(np.mean(totals))


def epsilon_delta(m, u, delta=0.05):
    """The model-agnostic slack :math:`\\varepsilon_\\delta` of Theorem 1,

    .. math::

        \\varepsilon_\\delta = \\left( \\frac{m u \\omega^2}{2\\binom{m + u - 1}{2}}
            \\Big( 1 - \\tfrac{1}{2\\max\\{m, u\\}} \\Big)^{-1} \\ln \\tfrac{1}{\\delta}
            \\right)^{1/2},
        \\qquad \\omega = \\tfrac{1}{m} + \\tfrac{1}{u}.
    """
    m, u = float(m), float(u)
    omega = 1.0 / m + 1.0 / u
    binom = (m + u - 1) * (m + u - 2) / 2.0
    scale = (m * u * omega ** 2) / (2.0 * binom)
    correction = 1.0 / (1.0 - 1.0 / (2.0 * max(m, u)))
    return float(np.sqrt(scale * correction * np.log(1.0 / delta)))


def ot_bound_subsampled(features, margin_all, y, train_mask, test_mask, num_classes,
                        max_samples=0, max_batches=0, seed=0, **kwargs):
    """:func:`ot_bound` on datasets too large to evaluate in one go.

    The exact :math:`W_1` solver and the pairwise change-rate matrix are both quadratic
    in the number of points, which is fine for a graph of a few thousand nodes but not
    for CovType or the image benchmarks.  The sample set is therefore cut into disjoint
    blocks of at most ``max_samples`` points, each preserving the train/test proportion,
    the bound is evaluated on every block, and the results are averaged weighted by block
    size.

    Blocks are disjoint rather than resampled so that each point contributes once, and
    proportional so that every block poses the same estimation problem as the whole.

    Args:
        max_samples: points per block; ``0`` evaluates on everything at once.
        max_batches: cap on the number of blocks; ``0`` uses as many as it takes to
            cover the data.  A cap trades coverage for time -- the caller should say so
            when reporting.
        seed: controls the shuffle that assigns points to blocks.

    Returns:
        dict: the same keys as :func:`ot_bound`, with the per-class vectors averaged
        across blocks as well.
    """
    train_idx = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    test_idx = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    n_train, n_test = len(train_idx), len(test_idx)
    n_total = n_train + n_test

    if max_samples <= 0 or n_total <= max_samples:
        blocks = [(train_idx, test_idx)]
    else:
        rng = np.random.RandomState(seed)
        train_perm = rng.permutation(train_idx)
        test_perm = rng.permutation(test_idx)
        train_per_block = max(1, int(round(max_samples * n_train / n_total)))
        test_per_block = max(1, max_samples - train_per_block)
        n_blocks = max(
            -(-n_train // train_per_block), -(-n_test // test_per_block)
        )
        if max_batches > 0:
            n_blocks = min(n_blocks, max_batches)
        blocks = []
        for b in range(n_blocks):
            tr = train_perm[b * train_per_block:(b + 1) * train_per_block]
            te = test_perm[b * test_per_block:(b + 1) * test_per_block]
            if len(tr) == 0 or len(te) == 0:
                break
            blocks.append((tr, te))

    device = features.device
    totals = {"oracle": 0.0, "test_blind": 0.0}
    vectors = {"max_change": 0.0, "expected_w1_oracle": 0.0,
               "expected_w1_test_blind": 0.0}
    weight_total = 0

    for tr, te in blocks:
        idx = torch.as_tensor(np.concatenate([tr, te]), device=device, dtype=torch.long)
        n_block = len(idx)
        block_train = torch.zeros(n_block, dtype=torch.bool, device=device)
        block_train[:len(tr)] = True

        result = ot_bound(
            features[idx], margin_all[idx], y[idx], block_train, ~block_train,
            num_classes, **kwargs
        )
        for key in totals:
            totals[key] += result[key] * n_block
        for key in vectors:
            vectors[key] = vectors[key] + result[key] * n_block
        weight_total += n_block

    out = {k: v / weight_total for k, v in totals.items()}
    out.update({k: v / weight_total for k, v in vectors.items()})
    out["n_blocks"] = len(blocks)
    return out
