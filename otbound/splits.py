"""Train/test splits and the permutation sampler shared by the bound and the regularizer.

Two distinct notions of "split" live here, and keeping them apart matters:

``pi`` (the *data* split)
    The transductive train/test split of the dataset itself, denoted :math:`\\pi` in
    the paper.  Nodes in ``I_train`` have observed labels; nodes in ``I_test`` do not.
    Following Appendix E.1, we use 30% train / 70% test for the correlation and depth
    experiments.

``pi_prime`` (the *estimation* permutation)
    An auxiliary permutation :math:`\\pi'`, drawn independently of :math:`\\pi`, used
    only to estimate the expectation :math:`\\mathbb{E}_{\\pi'}[\\cdot]` that appears in
    Theorem 1.  It re-partitions the nodes into a fresh train/test pair of the same
    sizes :math:`(m, u)`.  It has nothing to do with which labels are observed.

Both :mod:`otbound.bound` (Theorem 1) and :mod:`otbound.regularizer` (Section 5.2)
draw their :math:`\\pi'` samples from this module, so the two can never drift apart.
They differ only in *how* a sample is drawn:

===========================  ==========================================================
:func:`sample_pi_prime`      Plain :math:`\\pi'`: a uniform permutation over all nodes,
                             split at :math:`m : u`.  Used by the bound, which can
                             afford ``T`` samples (default ``T = 4``, Appendix B).
:func:`stratified_class_split`  The class-stratified counterpart: within one class,
                             split the labelled nodes at the same :math:`m : u` ratio.
                             Used by the regularizer, which draws a single sample per
                             gradient step and needs the lower variance (Section 5.2).
===========================  ==========================================================

Both estimate the same quantity; stratification only reduces the variance of the
per-sample estimate, at the cost of fixing :math:`m_c^{(\\pi')}` instead of letting it
fluctuate.  Crucially, **both split at the** :math:`m : u` **ratio** -- that ratio is
part of the estimand, not a free knob.
"""

import numpy as np
import torch

__all__ = [
    "index_to_mask",
    "random_planetoid_splits",
    "random_split_wo_valid",
    "train_ratio_of",
    "sample_pi_prime",
    "classwise_index_sets",
    "stratified_class_split",
]


def index_to_mask(index, size):
    """Turn an index tensor/array into a boolean mask of length ``size``."""
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, seed=12134):
    """60% / 20% / 20% train / val / test split (used when ``--useval 1``)."""
    train_lb = int(round(0.6 * len(data.y)))
    val_lb = int(round(0.2 * len(data.y)))

    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(data.y.shape[0], train_lb, replace=False)
    rest_index = np.setdiff1d(np.arange(data.y.shape[0]), train_idx)
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = np.setdiff1d(rest_index, val_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    return data


def random_split_wo_valid(data, train_rate, seed=12134):
    """``train_rate`` / ``1 - train_rate`` split with no validation set.

    This is the transductive protocol of Appendix E.1: ``train_rate = 0.3``.
    """
    train_lb = int(round(train_rate * len(data.y)))

    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(data.y.shape[0], train_lb, replace=False)
    test_idx = np.setdiff1d(np.arange(data.y.shape[0]), train_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    return data


def train_ratio_of(train_mask, test_mask):
    """The ratio :math:`m / (m + u)` that :math:`\\pi'` must reproduce.

    Both estimators split at this ratio, so it is computed in one place.  Note that
    ``m + u`` is taken from the two masks rather than from ``num_nodes``: when a
    validation set is carved out, those nodes take part in neither risk.
    """
    m = float(train_mask.sum().item())
    u = float(test_mask.sum().item())
    if m + u == 0:
        raise ValueError("both train_mask and test_mask are empty")
    return m / (m + u)


def sample_pi_prime(pool_mask, num_train, device, seed=None):
    """Draw :math:`\\pi'` and return its ``(train_mask, test_mask)``.

    ``pool_mask`` marks the nodes that :math:`\\pi'` is allowed to permute -- the
    transductive population :math:`I^{(\\pi)}_{\\mathrm{train}} \\cup
    I^{(\\pi)}_{\\mathrm{test}}`.  (When a validation set has been carved out, those
    nodes belong to neither risk in Theorem 1 and so stay out of the pool.)  Of the
    pool, ``num_train`` nodes are drawn uniformly to form
    :math:`I^{(\\pi')}_{\\mathrm{train}}` and the rest form
    :math:`I^{(\\pi')}_{\\mathrm{test}}`.

    Note that :math:`\\pi'` ignores which labels are observed; restricting to observed
    labels is the job of :func:`classwise_index_sets`.

    ``seed`` makes a draw reproducible; the bound passes the permutation index so that
    the ``t``-th sample is the same across runs.

    Returns two boolean masks over the full node set, both false outside the pool.
    """
    num_nodes = int(pool_mask.numel())
    pool_idx = pool_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    if num_train > len(pool_idx):
        raise ValueError("num_train exceeds the size of the permutation pool")

    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(pool_idx, num_train, replace=False)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[torch.as_tensor(train_idx, device=device)] = True
    test_mask = pool_mask & (~train_mask)
    return train_mask, test_mask


def classwise_index_sets(class_mask, pi_train_mask, pi_test_mask, known_mask=None):
    """Class-``c`` index sets under :math:`\\pi'`, for one of the two estimators.

    With ``known_mask=None`` this is the **OT-oracle** of Appendix B: every label is
    assumed available, so

    .. math:: I^{(\\pi')}_{\\mathrm{train},c},\\quad I^{(\\pi')}_{\\mathrm{test},c}

    are taken directly.  With ``known_mask`` set to the observed-label mask
    :math:`I^{(\\pi)}_{\\mathrm{train}}` this is the test-blind **OT bound**: nodes in
    :math:`I^{(\\pi)}_{\\mathrm{test}}` are dropped before the class intersection, so no
    test label ever leaks,

    .. math::

        I^{(\\pi';\\pi)}_{\\mathrm{train},c}
            = (I^{(\\pi')}_{\\mathrm{train}} \\setminus I^{(\\pi)}_{\\mathrm{test}}) \\cap I_c ,
        \\qquad
        I^{(\\pi';\\pi)}_{\\mathrm{test},c}
            = (I^{(\\pi')}_{\\mathrm{test}} \\setminus I^{(\\pi)}_{\\mathrm{test}}) \\cap I_c .

    Returns a ``(train_c_mask, test_c_mask)`` pair of boolean masks.
    """
    train_c = class_mask & pi_train_mask
    test_c = class_mask & pi_test_mask
    if known_mask is not None:
        train_c = train_c & known_mask
        test_c = test_c & known_mask
    return train_c, test_c


def stratified_class_split(class_idx, train_ratio, generator=None):
    """The class-stratified counterpart of :func:`sample_pi_prime`, for one class.

    ``class_idx`` holds the indices of the labelled class-``c`` nodes.  They are
    shuffled and cut at ``train_ratio``, producing the :math:`\\pi'`-train and
    :math:`\\pi'`-test parts of class ``c``:

    .. math::
        k = \\operatorname{round}(|I_c| \\cdot m / (m + u)),\\quad
        h_1 = \\text{first } k,\\quad h_2 = \\text{the rest}.

    Splitting at ``train_ratio`` rather than in half is what makes this an estimator of
    the same :math:`W_1` the bound uses: the two empirical measures carry the sample
    sizes :math:`m_c` and :math:`u_c` that Theorem 1 refers to, and empirical
    :math:`W_1` is sensitive to those sizes.

    ``generator`` is an optional :class:`torch.Generator` for reproducible shuffles.

    Returns ``(h1, h2)``; either may be empty when the class has very few labelled
    nodes, and callers are expected to skip such classes.
    """
    n_c = int(class_idx.numel())
    if n_c == 0:
        empty = class_idx[:0]
        return empty, empty

    perm = torch.randperm(n_c, device=class_idx.device, generator=generator)
    k = int(round(n_c * float(train_ratio)))
    k = min(max(k, 1), n_c - 1) if n_c >= 2 else n_c
    return class_idx[perm[:k]], class_idx[perm[k:]]
