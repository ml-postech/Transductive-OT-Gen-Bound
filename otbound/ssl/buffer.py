"""Labelled-feature buffer for the semi-supervised regularizer (Appendix E.3).

In the 2%-label regime of Section 6 a mini-batch holds only a handful of labelled
samples per class -- far too few for a stable per-class :math:`W_1`, which is what
:math:`R` needs.  Appendix E.3 works around this by keeping the labelled features of
the most recent few mini-batches and concatenating them with the current batch before
computing :math:`R`.

Buffered entries are **detached**.  They enlarge the empirical measures that
:math:`W_1` and :math:`M_c` are estimated from, but gradient flows only through the
current batch, so no stale computation graph is kept alive and the optimizer step stays
a function of the current forward pass alone.
"""

import torch

__all__ = ["LabeledFeatureBuffer"]


class LabeledFeatureBuffer:
    """Ring buffer over the last ``capacity`` mini-batches of labelled features.

    Args:
        capacity: number of past mini-batches to retain.  Appendix E.3 uses 4.

    Typical use inside a training step::

        feats, margins = ...              # current batch, with gradient
        all_f, all_m, all_y = buffer.concat(feats, margins, y)
        reg = ot_regularizer(all_f, all_m, all_y, unlabeled_f, unlabeled_m, K, ratio)
        buffer.update(feats, margins, y)  # after, so the current batch is not doubled
    """

    def __init__(self, capacity=4):
        self.capacity = capacity
        self._features = []
        self._margins = []
        self._labels = []

    def __len__(self):
        return len(self._features)

    def update(self, features, margins, labels):
        """Push the current batch, evicting the oldest once past ``capacity``."""
        self._features.append(features.detach())
        self._margins.append(margins.detach())
        self._labels.append(labels.detach())
        if len(self._features) > self.capacity:
            self._features.pop(0)
            self._margins.pop(0)
            self._labels.pop(0)

    def concat(self, features, margins, labels):
        """Current batch first, then the detached buffer.

        Returns ``(features, margins, labels)`` ready to pass as the labelled side of
        :func:`otbound.regularizer.ot_regularizer`.  With an empty buffer this is the
        current batch unchanged.
        """
        if not self._features:
            return features, margins, labels
        return (
            torch.cat([features] + self._features, dim=0),
            torch.cat([margins] + self._margins, dim=0),
            torch.cat([labels] + self._labels, dim=0),
        )

    def clear(self):
        self._features.clear()
        self._margins.clear()
        self._labels.clear()
