"""Applying the OT regularizer inside a semi-supervised training step (Section 6).

The regularizer itself is unchanged -- :func:`otbound.regularizer.ot_regularizer` is
called directly.  Only the two inputs need care in the low-label regime:

* the **labelled** side is widened with :class:`~otbound.ssl.buffer.LabeledFeatureBuffer`
  so the per-class :math:`W_1` has enough points to be meaningful, and
* the **unlabelled** side is the algorithm's current unlabelled batch, which every
  method in Section 6 already forwards.  Only its features are read; its pseudo-labels
  are not, so the regularizer stays label-free on that side.

``train_ratio`` should be the labelled fraction of the whole pool (0.02 in Section 6),
not the ratio within a batch: it defines the estimand, so batching must not change it.
"""

from ..margin import margins_all
from ..regularizer import ot_regularizer

__all__ = ["ssl_ot_regularizer"]


def ssl_ot_regularizer(labeled_features, labeled_logits, labeled_y,
                       unlabeled_features, unlabeled_logits,
                       num_classes, train_ratio, buffer=None, **reg_kwargs):
    """Compute :math:`R` for one semi-supervised step.

    Args:
        labeled_features, labeled_logits, labeled_y: the current labelled batch.
        unlabeled_features, unlabeled_logits: the current unlabelled batch; labels and
            pseudo-labels are never read.
        num_classes: :math:`K`.
        train_ratio: labelled fraction of the full pool.
        buffer: optional :class:`~otbound.ssl.buffer.LabeledFeatureBuffer`.  When given,
            it is read before the step and updated after.
        **reg_kwargs: forwarded to :func:`otbound.regularizer.ot_regularizer`
            (``sinkhorn_eps``, ``soft_temp``, ``n_sample``, ...).

    Returns:
        torch.Tensor: the scalar :math:`R`, to be added to the algorithm's own loss as
        :math:`\\mathcal{L}_{\\mathrm{orig}} + \\lambda R`.
    """
    labeled_margins = margins_all(labeled_logits)
    unlabeled_margins = margins_all(unlabeled_logits)

    if buffer is not None:
        feats, margins, ys = buffer.concat(labeled_features, labeled_margins, labeled_y)
    else:
        feats, margins, ys = labeled_features, labeled_margins, labeled_y

    reg = ot_regularizer(
        train_features=feats,
        train_margin_all=margins,
        train_y=ys,
        test_features=unlabeled_features,
        test_margin_all=unlabeled_margins,
        num_classes=num_classes,
        train_ratio=train_ratio,
        **reg_kwargs,
    )

    if buffer is not None:
        buffer.update(labeled_features, labeled_margins, labeled_y)

    return reg
