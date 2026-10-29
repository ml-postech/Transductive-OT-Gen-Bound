"""Tables 3 and 4 (semi-supervised columns) -- **skeleton, not yet runnable**.

Section 6 evaluates the bound and the regularizer on top of four semi-supervised
algorithms on CIFAR-10 and SVHN at a 2% label budget.  The algorithms themselves are not
in this repository yet (see the checklist at the top of the README); this file is the
driver they plug into, so that finishing them is a matter of implementing
:meth:`otbound.ssl.algorithms.SSLAlgorithm.compute_loss` rather than rewriting the
measurement.

Everything specific to *this paper* is already implemented and wired up below:

* :class:`~otbound.ssl.buffer.LabeledFeatureBuffer` -- the Appendix H.3 buffer that
  widens the labelled side of :math:`R`, since a 2%-label mini-batch holds too few
  points per class for a stable per-class :math:`W_1`;
* :func:`~otbound.ssl.regularizer.ssl_ot_regularizer` -- assembling :math:`R` from a
  labelled batch plus buffer and an unlabelled batch;
* :func:`~otbound.bound.ot_bound_subsampled` -- evaluating the bound afterwards.

What is missing is marked ``TODO`` and raises rather than silently producing numbers.

Appendix H.3 protocol: 2% of the pool labelled, class-balanced (1,200 labels for
CIFAR-10, 2,000 for SVHN); each algorithm keeps the architecture, optimizer, learning
rate and batch size of its own paper; reported accuracy is the median over the last few
checkpoints, averaged over 5 runs, following the Unified SSL Benchmark convention.
"""

import argparse

from _common import (SEEDS, add_regularizer_args, get_device)

from otbound.data import TabularDataset
from otbound.ssl import LabeledFeatureBuffer, build_ssl_algorithm, ssl_ot_regularizer


def build_dataloaders(dataset, labeled_fraction, seed, args):
    """TODO: class-balanced labelled / unlabelled split, with the weak and strong views
    FixMatch and SimMatch need.

    Must return ``(labeled_loader, unlabeled_loader, eval_loader)``.  The split has to be
    class-balanced -- Appendix H.3 takes the same number of labels per class -- and the
    unlabelled pool is the remainder of the *same* dataset, since the setting is
    transductive.
    """
    raise NotImplementedError(
        "semi-supervised data pipeline not implemented; see the README checklist"
    )


def train(args, dataset, device, seed):
    """One semi-supervised run, with the OT regularizer optionally attached."""
    labeled_loader, unlabeled_loader, eval_loader = build_dataloaders(
        dataset, args.labeled_fraction, seed, args
    )

    model = None  # TODO: the algorithm's base architecture (WideResNet in the USB default)
    algorithm = build_ssl_algorithm(args.algorithm, model, args, device)
    optimizer = None  # TODO: the algorithm's own optimizer and schedule

    buffer = LabeledFeatureBuffer(capacity=args.buffer_batches)
    use_reg = args.regularizer != "None" and args.reg_weight != 0

    for step, (labeled_batch, unlabeled_batch) in enumerate(
            zip(labeled_loader, unlabeled_loader)):
        output = algorithm.compute_loss(labeled_batch, unlabeled_batch)
        loss = output.loss

        if use_reg:
            loss = loss + args.reg_weight * ssl_ot_regularizer(
                labeled_features=output.labeled_features,
                labeled_logits=output.labeled_logits,
                labeled_y=output.labeled_y,
                unlabeled_features=output.unlabeled_features,
                unlabeled_logits=output.unlabeled_logits,
                num_classes=dataset.num_classes,
                # The estimand is fixed by the pool, not by the batch: 2% labelled.
                train_ratio=args.labeled_fraction,
                buffer=buffer,
                sinkhorn_eps=args.sinkhorn_eps,
                sinkhorn_iter=args.sinkhorn_iter,
                soft_temp=args.soft_temp,
                n_sample=args.reg_n_sample,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    raise NotImplementedError(
        "evaluation loop not implemented: report the median accuracy over the last "
        "few checkpoints, per Appendix H.3"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "svhn"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["lp_deepssl", "mean_teacher", "fixmatch", "simmatch"])
    parser.add_argument("--labeled_fraction", type=float, default=0.02,
                        help="fraction of the pool that is labelled (Appendix H.3: 0.02)")
    parser.add_argument("--buffer_batches", type=int, default=4,
                        help="labelled mini-batches retained for the per-class W1 "
                             "estimate (Appendix H.3: 4)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--result_csv", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    add_regularizer_args(parser)
    args = parser.parse_args()

    device = get_device(args)
    dataset = TabularDataset(args.dataset, args.data_root)
    for run in range(args.runs):
        train(args, dataset, device, SEEDS[run])


if __name__ == "__main__":
    main()
