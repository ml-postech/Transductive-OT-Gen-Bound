"""Reference implementation for *Transductive Generalization for GNNs via Optimal Transport*.

Module map, by the part of the paper each one implements:

============================  =================================================
:mod:`otbound.splits`         the data split :math:`\\pi` and the estimation
                              permutation :math:`\\pi'` (Appendix C)
:mod:`otbound.wasserstein`    empirical :math:`W_1`, exact and Sinkhorn
:mod:`otbound.margin`         margins and :math:`M_c(f, \\phi)` (Theorem 1)
:mod:`otbound.bound`          the OT bound itself (Theorem 1, Appendix C)
:mod:`otbound.regularizer`    the differentiable surrogate :math:`R` (Section 5.2)
:mod:`otbound.depth`          :math:`W_C` and :math:`W_S` (Section 5.3)
:mod:`otbound.baselines`      PAC-Bayes and Rademacher comparisons (Table 1)
:mod:`otbound.models`         GNN encoders and classifiers
:mod:`otbound.tabular`        MLP / CNN encoders for the non-graph experiments
:mod:`otbound.data`           dataset loading
:mod:`otbound.ssl`            semi-supervised hooks for Section 6
============================  =================================================

:mod:`otbound.bound` and :mod:`otbound.regularizer` deliberately share
:mod:`otbound.splits` and :mod:`otbound.margin`: the regularizer is the bound with two
non-smooth pieces replaced, and routing both through the same estimator code keeps them
from drifting apart.
"""

__version__ = "1.0.0"
