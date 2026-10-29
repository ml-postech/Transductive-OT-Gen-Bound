"""Semi-supervised algorithm interface for Section 6 -- **skeleton, not yet implemented**.

Tables 3 and 4 evaluate the OT bound and the OT regularizer on top of four
semi-supervised methods.  That code is not part of this repository yet (see the
checklist at the top of the README); what is here is the interface it needs to satisfy,
so that :mod:`otbound.bound` and :mod:`otbound.ssl.regularizer` can be dropped in
without touching them.

Implementing one method means subclassing :class:`SSLAlgorithm` and filling in
:meth:`compute_loss`.  Two requirements come from the bound, not from the algorithm:

1. the model must expose ``feature(x)`` returning :math:`\\phi(x)` separately from the
   logits, because every quantity in Theorem 1 lives in feature space, and
2. :meth:`compute_loss` must return the unlabelled batch's features and logits along
   with the loss, since :math:`M_c` is estimated over
   :math:`I_{\\mathrm{train},c} \\cup I_{\\mathrm{test}}`.

Reference implementations to port from, all available in the Unified SSL Benchmark
(Wang et al., 2022, https://github.com/microsoft/Semi-supervised-learning):

======================  ================================================
LP-DeepSSL              Iscen et al., 2019
Mean Teacher            Tarvainen and Valpola, 2017
FixMatch                Sohn et al., 2020
SimMatch                Zheng et al., 2022
======================  ================================================

Appendix H.3 fixes the protocol: 2% of the pool labelled, class-balanced; each method
keeps the base architecture and default optimizer, learning rate and batch size from its
own paper; reported accuracy is the median over the last few checkpoints, averaged over
5 runs.
"""

__all__ = ["SSLAlgorithm", "SSLStepOutput", "LPDeepSSL", "MeanTeacher", "FixMatch",
           "SimMatch", "build_ssl_algorithm", "SSL_ALGORITHMS"]


class SSLStepOutput(object):
    """What one training step must hand back for the regularizer to be applied.

    Attributes:
        loss: the algorithm's own loss :math:`\\mathcal{L}_{\\mathrm{orig}}`.
        labeled_features, labeled_logits, labeled_y: the labelled batch.
        unlabeled_features, unlabeled_logits: the unlabelled batch; no labels, real or
            pseudo, are needed.
    """

    def __init__(self, loss, labeled_features, labeled_logits, labeled_y,
                 unlabeled_features, unlabeled_logits):
        self.loss = loss
        self.labeled_features = labeled_features
        self.labeled_logits = labeled_logits
        self.labeled_y = labeled_y
        self.unlabeled_features = unlabeled_features
        self.unlabeled_logits = unlabeled_logits


class SSLAlgorithm(object):
    """Base class for the four Section 6 methods.

    Subclasses own the model, the optimizer and whatever extra state they need (a
    teacher network, a memory bank, a label propagation graph).  The only contract with
    this repository is :meth:`compute_loss`.
    """

    name = None

    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device

    def compute_loss(self, labeled_batch, unlabeled_batch):
        """Run one step of the algorithm and return an :class:`SSLStepOutput`.

        Args:
            labeled_batch: ``(x, y)`` for the labelled samples.
            unlabeled_batch: the unlabelled samples, in whatever form the algorithm
                expects (FixMatch and SimMatch want weak and strong views).
        """
        raise NotImplementedError(
            "{}.compute_loss is not implemented yet; see the README checklist".format(
                type(self).__name__)
        )

    def on_epoch_end(self):
        """Hook for per-epoch bookkeeping (label propagation, teacher EMA decay, ...)."""
        return None


class LPDeepSSL(SSLAlgorithm):
    """Label propagation for deep semi-supervised learning (Iscen et al., 2019)."""

    name = "lp_deepssl"


class MeanTeacher(SSLAlgorithm):
    """Mean Teacher (Tarvainen and Valpola, 2017)."""

    name = "mean_teacher"


class FixMatch(SSLAlgorithm):
    """FixMatch (Sohn et al., 2020)."""

    name = "fixmatch"


class SimMatch(SSLAlgorithm):
    """SimMatch (Zheng et al., 2022)."""

    name = "simmatch"


SSL_ALGORITHMS = {
    cls.name: cls for cls in (LPDeepSSL, MeanTeacher, FixMatch, SimMatch)
}


def build_ssl_algorithm(name, model, args, device):
    """Instantiate one of :data:`SSL_ALGORITHMS` by name."""
    if name not in SSL_ALGORITHMS:
        raise ValueError(
            "unknown algorithm {!r}; expected one of {}".format(
                name, ", ".join(sorted(SSL_ALGORITHMS)))
        )
    return SSL_ALGORITHMS[name](model, args, device)
