"""Tables 7 and 8, and the supervised columns of Table 3: correlation beyond graphs.

Nothing about the bound is graph-specific, so the same measurement is run on tabular and
image data with MLP and CNN encoders.  Appendix D.1 fixes the protocol: 30% of the pool
labelled, Adam at lr 0.01, hidden dimension 64, 500 epochs, and 18 configurations per
(dataset, architecture) from encoder depth {1, 2, 4, 8, 16, 32} crossed with classifier
depth {1, 2, 4}.  The rank correlation is taken across those 18 rows.

Datasets in the paper: ``breast_cancer``, ``letter``, ``covtype`` (Table 7);
``digits``, ``mnist``, ``fashion_mnist`` (Table 8); ``cifar10``, ``svhn`` (Table 3,
supervised).

``--bound_max_samples`` caps how many points the bound is evaluated on at once; the
exact :math:`W_1` solver is quadratic, so the larger datasets need it.  See
:func:`otbound.bound.ot_bound_subsampled`.
"""

import argparse

import torch

from _common import (SEEDS, append_csv_row, get_device, summarize)

from otbound.baselines import pac_bayes_bound
from otbound.bound import ot_bound_subsampled
from otbound.data import TABULAR_DATASETS, TabularDataset
from otbound.margin import margins_all, margins_true
from otbound.tabular import TABULAR_NETS, batched_forward, build_tabular_model
from otbound.training import train_tabular_model


def add_tabular_args(parser):
    parser.add_argument("--dataset", type=str, required=True, choices=list(TABULAR_DATASETS))
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--net", type=str, default="mlp", choices=list(TABULAR_NETS))
    parser.add_argument("--e_depth", type=int, default=8)
    parser.add_argument("--c_depth", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--act", type=str, default="ReLU")
    parser.add_argument("--layernorm", type=str, default="None")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 trains full-batch; set it for the image benchmarks")
    parser.add_argument("--train_rate", type=float, default=0.3)
    parser.add_argument("--use_val", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--result_csv", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_once(args, dataset, device, seed):
    train_mask, val_mask, test_mask = dataset.get_split(
        seed, use_val=bool(args.use_val), train_rate=args.train_rate
    )
    X, y = dataset.X.to(device), dataset.y.to(device)
    train_mask, test_mask = train_mask.to(device), test_mask.to(device)
    val_mask = val_mask.to(device) if args.use_val else None

    model = build_tabular_model(args, dataset.num_features, dataset.num_classes,
                                dataset.image_shape)
    model, _ = train_tabular_model(model, X, y, train_mask, test_mask, args, device,
                                   val_mask=val_mask, verbose=args.verbose)

    model.eval()
    with torch.no_grad():
        features = batched_forward(model.feature, X, args.batch_size)
        logits = batched_forward(model, X, args.batch_size)

    margin = margins_true(logits, y) if logits.dim() > 1 else logits * (2 * y - 1)
    train_err = float((margin <= 0)[train_mask].float().mean())
    test_err = float((margin <= 0)[test_mask].float().mean())

    margin_all = margins_all(logits if logits.dim() > 1
                             else torch.stack([-logits, logits], dim=-1))
    bounds = ot_bound_subsampled(
        features, margin_all, y, train_mask, test_mask, dataset.num_classes,
        max_samples=args.bound_max_samples, max_batches=args.bound_max_batches,
        seed=seed, n_permutations=args.n_permutations, gamma=args.gamma,
    )
    if bounds["n_blocks"] > 1:
        print("  bound evaluated on {} disjoint blocks of <= {} points".format(
            bounds["n_blocks"], args.bound_max_samples))

    return {
        "gen_error": abs(test_err - train_err),
        "train_error": train_err, "test_error": test_err,
        "ot_oracle": bounds["oracle"], "ot_bound": bounds["test_blind"],
        "pac_bound": pac_bayes_bound(model, int(train_mask.sum())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_tabular_args(parser)
    parser.add_argument("--n_permutations", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--bound_max_samples", type=int, default=4000,
                        help="points per block when evaluating the bound; 0 for all")
    parser.add_argument("--bound_max_batches", type=int, default=0,
                        help="cap on the number of blocks; 0 covers the whole dataset")
    args = parser.parse_args()

    device = get_device(args)
    dataset = TabularDataset(args.dataset, args.data_root)
    print(args)
    print("{}: {} samples, {} features, {} classes, image_shape={}".format(
        dataset.name, dataset.num_samples, dataset.num_features,
        dataset.num_classes, dataset.image_shape))

    per_run = []
    for run in range(args.runs):
        per_run.append(run_once(args, dataset, device, SEEDS[run]))
        print("run {}/{}  ".format(run + 1, args.runs)
              + "  ".join("{}={:.6g}".format(k, v) for k, v in per_run[-1].items()))

    row = {"net": args.net, "dataset": args.dataset,
           "e_depth": args.e_depth, "c_depth": args.c_depth}
    for key in per_run[0]:
        mean, std = summarize([r[key] for r in per_run])
        row[key] = round(mean, 8)
        row[key + "_std"] = round(std, 8)

    print("\n" + "  ".join("{}={}".format(k, v) for k, v in row.items()))
    append_csv_row(args.result_csv, row)


if __name__ == "__main__":
    main()
