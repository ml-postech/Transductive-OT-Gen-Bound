"""Figure 1 and Appendix C.3: how feature geometry and generalization vary with depth.

Trains one encoder at one depth and records the three quantities Figure 1 plots:

``w_c``
    intra-class concentration :math:`W_C` -- the Theorem 1 Wasserstein term, averaged
    over classes.  Propositions 2 and 3 predict it contracts with depth.
``w_s``
    inter-class separation :math:`W_S` -- mean :math:`W_1` between distinct classes.  It
    contracts too, which is the harmful half of the trade-off.
``gen_error``
    the empirical generalization gap, which the two pull in opposite directions and is
    therefore non-monotone in depth.

``--cong_bound 1`` additionally records the uniform-stability bound of Cong et al.
(2021), the depth-monotone comparison discussed in Section 5.3.

Sweep ``--e_depth`` over {1, 4, 8, 12, 16, 20, 24, 28, 32} for one dataset and encoder
to reproduce one panel; Appendix E.2 fixes a two-layer classifier, hidden dimension 64,
a 30/70 split with no validation set, and 5 random splits.
"""

import argparse

from _common import (SEEDS, add_common_args, append_csv_row, get_device, summarize)

from otbound.baselines import cong_stability_bound
from otbound.bound import inter_class_separation
from otbound.data import load_graph_dataset
from otbound.depth import intra_class_concentration
from otbound.margin import margins_true
from otbound.models import build_model
from otbound.training import train_graph_model


def run_once(args, dataset, device, seed):
    dataset.resplit(seed, use_val=bool(args.use_val), train_rate=args.train_rate)
    data = dataset.data.to(device)

    model = build_model(args, dataset.num_features, dataset.num_classes)
    model, _ = train_graph_model(model, data, args, device, verbose=args.verbose)

    model.eval()
    features = model.feature(data.x, data.edge_index).detach()
    logits = model(data.x, data.edge_index).detach()

    margin = margins_true(logits, data.y)
    train_err = float((margin <= 0)[data.train_mask].float().mean())
    test_err = float((margin <= 0)[data.test_mask].float().mean())

    w_c_mean, w_c_min, w_c_max = intra_class_concentration(
        features, data.y, data.train_mask, data.test_mask, dataset.num_classes,
        n_permutations=args.n_permutations,
    )
    w_s_mean, w_s_min, w_s_max = inter_class_separation(
        features, data.y, dataset.num_classes
    )

    result = {
        "gen_error": abs(test_err - train_err),
        "w_c": w_c_mean, "w_c_min": w_c_min, "w_c_max": w_c_max,
        "w_s": w_s_mean, "w_s_min": w_s_min, "w_s_max": w_s_max,
    }
    if args.cong_bound:
        result["cong_bound"] = cong_stability_bound(
            model, data, args.net, args.e_depth, args.lr, args.epochs,
            int(data.train_mask.sum()),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument("--n_permutations", type=int, default=4)
    parser.add_argument("--cong_bound", type=int, default=0,
                        help="also record the Cong et al. (2021) stability bound "
                             "(SGC and GCN only)")
    parser.set_defaults(net="sgc", c_depth=2, train_rate=0.3, use_val=0, epochs=500,
                        runs=5)
    args = parser.parse_args()

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

    row = {"net": args.net, "dataset": args.dataset, "e_depth": args.e_depth}
    for key in per_run[0]:
        mean, std = summarize([r[key] for r in per_run])
        row[key] = round(mean, 8)
        row[key + "_std"] = round(std, 8)

    print("\n" + "  ".join("{}={}".format(k, v) for k, v in row.items()))
    append_csv_row(args.result_csv, row)


if __name__ == "__main__":
    main()
