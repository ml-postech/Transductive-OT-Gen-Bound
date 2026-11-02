import torch

from ..margin import margins_all, soft_max_change_class
from ..wasserstein import sinkhorn_distance

__all__ = ["ssl_ot_regularizer"]


def ssl_ot_regularizer(labeled_features, labeled_y, labeled_logits,
                       unlabeled_features, unlabeled_logits, num_classes, buffer=None,
                       sinkhorn_eps=0.05, sinkhorn_iter=50, soft_temp=1.0, n_sample=200):
    """Compute :math:`R` for one semi-supervised step.

    Args:
        labeled_features, labeled_y, labeled_logits: the current labelled batch (with grad).
        unlabeled_features, unlabeled_logits: the current unlabelled batch (with grad).
        num_classes: :math:`K`.
        buffer: optional :class:`~otbound.ssl.buffer.LabeledFeatureBuffer`; read before
            the step and updated after.
        sinkhorn_eps, sinkhorn_iter: Sinkhorn entropic regularization and iteration count.
        soft_temp: temperature of the smoothed maximum in :math:`M_c`.
        n_sample: number of points sampled for the :math:`M_c` pair set.

    Returns:
        torch.Tensor: the scalar :math:`R`, added to the algorithm's own loss as
        :math:`\\mathcal{L}_{\\mathrm{orig}} + \\lambda R`.
    """
    device = labeled_features.device
    if buffer is not None:
        lab_features, lab_y = buffer.concat(labeled_features, labeled_y)
        buffer.update(labeled_features, labeled_y)
    else:
        lab_features, lab_y = labeled_features, labeled_y

    # pi' over labelled + unlabelled samples (labelled rows first); keep the labelled ones
    m = lab_features.shape[0]
    n = m + unlabeled_features.shape[0]
    pi_train = torch.zeros(n, dtype=torch.bool, device=device)
    pi_train[torch.randperm(n, device=device)[:m]] = True
    pi_train = pi_train[:m]

    labeled_margins = margins_all(labeled_logits)
    unlabeled_margins = margins_all(unlabeled_logits)

    terms = []
    for c in range(num_classes):
        train_c = (lab_y == c) & pi_train
        test_c = (lab_y == c) & ~pi_train
        if int(train_c.sum()) < 1 or int(test_c.sum()) < 1:
            continue
        w1_c = sinkhorn_distance(lab_features[train_c], lab_features[test_c],
                                 epsilon=sinkhorn_eps, max_iter=sinkhorn_iter)

        cur_c = labeled_y == c
        support_features = torch.cat([labeled_features[cur_c], unlabeled_features], dim=0)
        support_margins = torch.cat([labeled_margins[cur_c], unlabeled_margins], dim=0)
        if support_features.shape[0] < 2:
            continue
        mc_c = soft_max_change_class(support_features, support_margins, c,
                                     temperature=soft_temp, n_sample=n_sample)
        terms.append(float(train_c.sum()) / m * mc_c * w1_c)

    if not terms:
        return labeled_features.new_zeros(())
    return torch.stack(terms).sum()
