"""Table 9, and the supervised columns of Table 4: the OT regularizer beyond graphs.

Same surrogate :math:`R` as the graph experiments -- :func:`otbound.regularizer.ot_regularizer`
is called directly -- applied to MLP and CNN encoders on image-shaped data.

Appendix D.2 fixes the protocol: encoder depth 8, classifier depth 2, hidden dimension
64, skip connections and batch normalization on every encoder (use ``--net resmlp`` or
``--net rescnn`` with ``--layernorm BatchNorm``), :math:`\\lambda` chosen from
``{1e-1, ..., 1e-4}`` on a validation set, 5 seeds.

``--regularizer None`` trains the identical architecture without the penalty, which is
the baseline row.

On mini-batched runs the labelled side of :math:`R` is the current batch.  The split
ratio passed to :math:`R` is nonetheless :math:`m / (m + u)` of the whole pool, computed
from the data split, because it defines what :math:`R` estimates and must not move with
the batch size.
"""

import argparse

from tabular_correlation import add_tabular_args

from _common import (SEEDS, add_regularizer_args, append_csv_row, get_device, summarize)

from otbound.data import TabularDataset
from otbound.margin import margins_all
from otbound.regularizer import ot_regularizer
from otbound.tabular import build_tabular_model
from otbound.training import train_tabular_model


def make_regularizer_fn(args, num_classes, train_ratio):
    """Build the hook the tabular training loop calls each step.

    Returns ``None`` for the baseline.
    """
    if args.regularizer == "None" or args.reg_weight == 0:
        return None

    def regularizer_fn(train_features, train_logits, train_y,
                       test_features, test_logits):
        reg = ot_regularizer(
            train_features=train_features,
            train_margin_all=margins_all(train_logits),
            train_y=train_y,
            test_features=test_features,
            test_margin_all=margins_all(test_logits),
            num_classes=num_classes,
            train_ratio=train_ratio,
            sinkhorn_eps=args.sinkhorn_eps,
            sinkhorn_iter=args.sinkhorn_iter,
            soft_temp=args.soft_temp,
            n_sample=args.reg_n_sample,
        )
        return args.reg_weight * reg

    return regularizer_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_tabular_args(parser)
    add_regularizer_args(parser)
    parser.set_defaults(net="rescnn", e_depth=8, c_depth=2, layernorm="BatchNorm",
                        use_val=1, train_rate=0.0, runs=5)
    args = parser.parse_args()

    device = get_device(args)
    dataset = TabularDataset(args.dataset, args.data_root)
    print(args)

    test_accs = []
    for run in range(args.runs):
        seed = SEEDS[run]
        train_mask, val_mask, test_mask = dataset.get_split(
            seed, use_val=bool(args.use_val), train_rate=args.train_rate
        )
        X, y = dataset.X.to(device), dataset.y.to(device)
        train_mask, test_mask = train_mask.to(device), test_mask.to(device)
        val_mask = val_mask.to(device) if args.use_val else None

        # m / (m + u) of the full pool; validation points belong to neither risk.
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
        train_ratio = n_train / float(n_train + n_test)

        model = build_tabular_model(args, dataset.num_features, dataset.num_classes,
                                    dataset.image_shape)
        _, history = train_tabular_model(
            model, X, y, train_mask, test_mask, args, device, val_mask=val_mask,
            regularizer_fn=make_regularizer_fn(args, dataset.num_classes, train_ratio),
            verbose=args.verbose,
        )
        test_accs.append(history.best_test_acc)
        print("run {}/{}  test acc {:.4f}  (epochs {})".format(
            run + 1, args.runs, history.best_test_acc, history.epochs_run))

    mean, std = summarize(test_accs)
    print("\n{} + {} on {}: test accuracy {:.2f} +- {:.2f}".format(
        args.net, args.regularizer, args.dataset, 100 * mean, 100 * std))

    append_csv_row(args.result_csv, {
        "net": args.net, "dataset": args.dataset, "regularizer": args.regularizer,
        "reg_weight": args.reg_weight, "lr": args.lr,
        "e_depth": args.e_depth, "c_depth": args.c_depth,
        "test_acc_mean": round(100 * mean, 4), "test_acc_std": round(100 * std, 4),
    })


if __name__ == "__main__":
    main()
