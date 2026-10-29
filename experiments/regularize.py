"""Table 2: the OT bound used as a training regularizer.

Trains with :math:`\\mathcal{L} = \\mathcal{L}_{\\mathrm{CE}} + \\lambda R`, where
:math:`R` is the differentiable surrogate of Section 5.2
(:func:`otbound.regularizer.ot_regularizer`).  Passing ``--regularizer None`` trains the
identical architecture without the penalty, which is the baseline row of Table 2.

``--regularizer global`` swaps in the Theorem 4 surrogate instead (Appendix D.2), and
``--reg_terms`` drops one of the two factors for the ablation of Appendix F.5.

Appendix H.1 fixes the protocol: encoder depth 8, classifier depth 2, all encoders given
skip connections (the ``res`` prefix) and batch normalization so that the comparison
isolates the regularizer's geometric effect from plain underfitting, and
:math:`\\lambda` chosen from ``{1e-1, ..., 1e-5}`` on a validation set over 5 seeds.
"""

import argparse

from _common import (SEEDS, add_common_args, add_regularizer_args, append_csv_row,
                     get_device, regularizer_kwargs, summarize)

from otbound.data import load_graph_dataset
from otbound.margin import margins_all
from otbound.models import build_model
from otbound.regularizer import ot_regularizer_from_masks
from otbound.splits import train_ratio_of
from otbound.training import train_graph_model


def make_regularizer_fn(args, num_classes):
    """Build the ``fn(logits, features, data)`` hook consumed by the training loop.

    Dispatches on ``--regularizer``: the class-wise surrogate of Theorem 1, the global
    one of Theorem 4, or ``None`` for the baseline, in which case training is plain
    cross entropy.
    """
    if args.regularizer == "None" or args.reg_weight == 0:
        return None

    kwargs = regularizer_kwargs(args)

    def regularizer_fn(logits, features, data):
        reg = ot_regularizer_from_masks(
            features=features,
            margin_all=margins_all(logits),
            y=data.y,
            train_mask=data.train_mask,
            test_mask=data.test_mask,
            num_classes=num_classes,
            # The pi' split ratio is m/(m+u) of the actual transductive problem, which
            # excludes any validation nodes -- see otbound.splits.train_ratio_of.
            train_ratio=train_ratio_of(data.train_mask, data.test_mask),
            **kwargs,
        )
        return args.reg_weight * reg

    return regularizer_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    add_regularizer_args(parser)
    parser.set_defaults(
        # Appendix H.1 settings for the regularization experiments.
        net="resgcn", e_depth=8, c_depth=2, layernorm="BatchNorm",
        use_val=1, train_rate=0.0, epochs=1000, early_stopping=200, runs=5,
    )
    args = parser.parse_args()

    device = get_device(args)
    dataset = load_graph_dataset(args.dataset, args.data_root, args.hetero_root)
    print(args)

    test_accs = []
    for run in range(args.runs):
        args.seed = SEEDS[run]
        dataset.resplit(args.seed, use_val=bool(args.use_val), train_rate=args.train_rate)
        data = dataset.data.to(device)

        model = build_model(args, dataset.num_features, dataset.num_classes)
        _, history = train_graph_model(
            model, data, args, device,
            regularizer_fn=make_regularizer_fn(args, dataset.num_classes),
            verbose=args.verbose,
        )
        test_accs.append(history.best_test_acc)
        print("run {}/{}  test acc {:.4f}  (epochs {})".format(
            run + 1, args.runs, history.best_test_acc, history.epochs_run))
        dataset.next_split()

    mean, std = summarize(test_accs)
    label = args.regularizer
    if args.regularizer == "ot" and args.reg_terms != "full":
        label += " ({}-only)".format(args.reg_terms)
    print("\n{} + {}: test accuracy {:.2f} +- {:.2f}".format(
        args.net, label, 100 * mean, 100 * std))

    append_csv_row(args.result_csv, {
        "net": args.net, "dataset": args.dataset, "regularizer": args.regularizer,
        "reg_terms": args.reg_terms, "reg_n_perm": args.reg_n_perm,
        "global_quantile": args.global_quantile,
        "reg_weight": args.reg_weight, "lr": args.lr,
        "e_depth": args.e_depth, "c_depth": args.c_depth,
        "test_acc_mean": round(100 * mean, 4), "test_acc_std": round(100 * std, 4),
    })


if __name__ == "__main__":
    main()
