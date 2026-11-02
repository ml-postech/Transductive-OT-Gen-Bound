import torch

__all__ = ["LabeledFeatureBuffer"]


class LabeledFeatureBuffer:
    """Ring buffer over the last ``capacity`` mini-batches of labelled features.

    Args:
        capacity: number of past mini-batches to retain.  Appendix H.3 uses 4.

    Typical use inside a training step::

        all_f, all_y = buffer.concat(feats, y)   # current batch first, with gradient
        ...                                      # R on (all_f, all_y)
        buffer.update(feats, y)                  # after, so the current batch is not doubled
    """

    def __init__(self, capacity=4):
        self.capacity = capacity
        self._features = []
        self._labels = []

    def __len__(self):
        return len(self._features)

    def update(self, features, labels):
        """Push the current batch, evicting the oldest once past ``capacity``."""
        self._features.append(features.detach())
        self._labels.append(labels.detach())
        if len(self._features) > self.capacity:
            self._features.pop(0)
            self._labels.pop(0)

    def concat(self, features, labels):
        """Current batch first, then the detached buffer.  With an empty buffer this is
        the current batch unchanged."""
        if not self._features:
            return features, labels
        return torch.cat([features] + self._features, dim=0), torch.cat([labels] + self._labels, dim=0)

    def clear(self):
        self._features.clear()
        self._labels.clear()
