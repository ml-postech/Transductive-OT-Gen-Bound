"""Semi-supervised OT regularizer for Section 6 (Table 4).

The semi-supervised algorithms themselves, and the Table 3 / Table 4 drivers, are in ``experiments/semisup/``.
"""

from .buffer import LabeledFeatureBuffer
from .regularizer import ssl_ot_regularizer

__all__ = ["LabeledFeatureBuffer", "ssl_ot_regularizer"]
