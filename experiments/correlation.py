"""Table 1: rank correlation between each bound and the empirical generalization error.

One invocation trains a single (architecture, depth, dataset) configuration, averages
the bound and the generalization gap over ``--runs`` random splits, and appends a row to
``--result_csv``.  The rank correlation itself is computed across the 18 configurations
that Appendix E.1 prescribes -- encoder depth in {1, 2, 4, 8, 16, 32} crossed with
classifier depth in {1, 2, 4} -- so sweep those and correlate the resulting rows.

Reported per row:

``gen_error``
    :math:`|R_u - R_m|`, the empirical generalization gap on 0-1 loss.
``ot_oracle`` / ``ot_bound``
    Theorem 1 with and without access to test labels (Appendix B).
``pac_bound``
    The PAC-Bayesian baseline.
``rc_bound``
    The Rademacher baseline of Esser et al. (2021).  Only defined for SGC and GCN with
    no classifier head, so it requires ``--net sgc|gcn --c_depth 0 --baseline_rc 1``.
"""

import argparse

from _common import (SEEDS, add_common_args, append_csv_row, get_device, summarize)

from otbound.baselines import pac_bayes_bound, rademacher_bound
from otbound.bound import ot_bound
from otbound.data import load_graph_dataset
from otbound.margin import margins_all, margins_true
from otbound.models import build_model
from otbound.training import train_graph_model


def run_once(args, dataset, device, seed):
    """Train one model on split ``seed`` and evaluate every bound on it."""
    dataset.resplit(seed, use_val=bool(args.use_val), train_rate=args.train_rate)
    data = dataset.data.to(device)

    head_free = bool(args.baseline_rc)
    model = build_model(args, dataset.num_features, dataset.num_classes,
                        head_free=head_free)
    model, _ = train_graph_model(model, data, args, device, verbose=args.verbose)

    model.eval()
    features = model.feature(data.x, data.edge_index).detach()
    logits = model(data.x, data.edge_index).detach()

    margin = margins_true(logits, data.y)
    train_err = float((margin <= 0)[data.train_mask].float().mean())
    test_err = float((margin <= 0)[data.test_mask].float().mean())

    result = {"gen_error": abs(test_err - train_err),
              "train_error": train_err, "test_error": test_err}

    if args.baseline_rc:
        # The Rademacher bound is derived for a head-free GNN, so the OT bound is not
        # comparable on the same run; sweep it separately.
        result["rc_bound"] = rademacher_bound(
            model, data, args.net, args.e_depth, dataset.num_features,
            int(data.num_nodes), int(data.train_mask.sum()), int(data.test_mask.sum()),
        )
        return result

    bounds = ot_bound(
        features, margins_all(logits), data.y, data.train_mask, data.test_mask,
        dataset.num_classes, n_permutations=args.n_permutations, gamma=args.gamma,
    )
    result["ot_oracle"] = bounds["oracle"]
    result["ot_bound"] = bounds["test_blind"]

    if args.baseline_pac:
        result["pac_bound"] = pac_bayes_bound(model, int(data.train_mask.sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument("--n_permutations", type=int, default=4,
                        help="number of pi' samples T used to estimate the expectation "
                             "in Theorem 1 (Appendix B uses 4)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="margin threshold; the bound carries a 1/gamma factor")
    parser.add_argument("--baseline_pac", type=int, default=1)
    parser.add_argument("--baseline_rc", type=int, default=0,
                        help="compute the Rademacher bound instead of the OT bound; "
                             "requires --net sgc|gcn and --c_depth 0")
    args = parser.parse_args()

    if args.baseline_rc and (args.net not in ("sgc", "gcn") or args.c_depth != 0):
        parser.error("--baseline_rc requires --net sgc|gcn and --c_depth 0")

    device = get_device(args)
    dataset = load_graph_dataset(args.dataset, args.data_root, args.hetero_root)
    print(args)

    per_run = []
    for run in range(args.runs):
        args.seed = SEEDS[run]
        per_run.append(run_once(args, dataset, device, args.seed))
        print("run {}/{}  ".format(run + 1, args.runs)
              + "  ".join("{}={:.6g}".format(k, v) for k, v in per_run[-1].items()))
        dataset.next_split()

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
