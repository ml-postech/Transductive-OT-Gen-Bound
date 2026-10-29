"""Semi-supervised hooks for Section 6 (Tables 3 and 4).

:mod:`~otbound.ssl.buffer` and :mod:`~otbound.ssl.regularizer` are complete: they hold
the parts that are specific to *this* paper, namely how :math:`R` is estimated when only
2% of the pool is labelled.

:mod:`~otbound.ssl.algorithms` is a skeleton.  The four semi-supervised methods
themselves are not in this repository yet; see that module for the interface they need
to satisfy.
"""

from .algorithms import (SSL_ALGORITHMS, SSLAlgorithm, SSLStepOutput,
                         build_ssl_algorithm)
from .buffer import LabeledFeatureBuffer
from .regularizer import ssl_ot_regularizer

__all__ = [
    "LabeledFeatureBuffer",
    "ssl_ot_regularizer",
    "SSLAlgorithm",
    "SSLStepOutput",
    "SSL_ALGORITHMS",
    "build_ssl_algorithm",
]
