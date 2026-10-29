"""Table 8 (Appendix E.1): rank correlation on transductive link prediction.

Link prediction is the second transductive task with dependent features, and the bound
transfers to it by treating an *edge* as the sample -- see :mod:`otbound.link`.

Protocol from Appendix E.1: 30% of the observed edges are used for training and the
remaining 70% for testing, with an equal number of negative edges sampled uniformly from
non-edges; Adam at lr 0.01 with no weight decay for 500 epochs.  Sweep encoder depth over
{1, 2, 4, 8, 16, 32} and predictor depth over {1, 2, 4} for the 18 configurations the
rank correlation is taken over.

The reported gap is on 0-1 loss over edges; test AUC is logged alongside for reference.
"""

import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, train_test_split_edges

from _common import (SEEDS, add_common_args, append_csv_row, get_device, summarize)

from otbound.bound import ot_bound
from otbound.data import load_graph_dataset
from otbound.link import LinkPredictionModel, build_edge_samples, link_labels


def _sample_negatives(pos_edge_index, num_nodes):
    return negative_sampling(
        edge_index=pos_edge_index, num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    )


@torch.no_grad()
def _auc(model, z, pos_edge_index, neg_edge_index):
    pos = model.decode(z, pos_edge_index).sigmoid()
    neg = model.decode(z, neg_edge_index).sigmoid()
    scores = torch.cat([pos, neg]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))
    ]).numpy()
    return float(roc_auc_score(labels, scores))


def run_once(args, dataset, device, seed):
    torch.manual_seed(seed)
    data = train_test_split_edges(
        dataset.data.clone(), val_ratio=0.0, test_ratio=1.0 - args.train_rate
    ).to(device)

    model = LinkPredictionModel(args, dataset.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Message passing uses training edges only, so no test edge leaks into the
        # representation the predictor sees.
        z = model.encode(data.x, data.train_pos_edge_index)
        neg_edge_index = _sample_negatives(data.train_pos_edge_index, data.num_nodes)
        edges = torch.cat([data.train_pos_edge_index, neg_edge_index], dim=-1)
        labels = link_labels(data.train_pos_edge_index.size(1),
                             neg_edge_index.size(1), device)

        loss = F.binary_cross_entropy_with_logits(model.decode(z, edges), labels)
        loss.backward()
        optimizer.step()

        if args.verbose and epoch % 50 == 0:
            print("  epoch {:4d}  loss {:.4f}".format(epoch, float(loss)))

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)

        train_neg = _sample_negatives(data.train_pos_edge_index, data.num_nodes)
        train_edges = torch.cat([data.train_pos_edge_index, train_neg], dim=-1)
        train_labels = link_labels(data.train_pos_edge_index.size(1),
                                   train_neg.size(1), device)
        test_edges = torch.cat(
            [data.test_pos_edge_index, data.test_neg_edge_index], dim=-1)
        test_labels = link_labels(data.test_pos_edge_index.size(1),
                                  data.test_neg_edge_index.size(1), device)

        features, margin_all, y, train_mask, test_mask = build_edge_samples(
            model, z, train_edges, train_labels, test_edges, test_labels
        )

        # 0-1 loss per edge: a sample is wrong when its signed score is non-positive.
        signed = torch.cat([
            model.decode(z, train_edges) * (2 * train_labels - 1),
            model.decode(z, test_edges) * (2 * test_labels - 1),
        ])
        train_err = float((signed[train_mask] <= 0).float().mean())
        test_err = float((signed[test_mask] <= 0).float().mean())

        bounds = ot_bound(
            features, margin_all, y, train_mask, test_mask, num_classes=2,
            n_permutations=args.n_permutations, gamma=args.gamma,
        )
        test_auc = _auc(model, z, data.test_pos_edge_index, data.test_neg_edge_index)

    return {
        "gen_error": abs(test_err - train_err),
        "train_error": train_err, "test_error": test_err, "test_auc": test_auc,
        "ot_oracle": bounds["oracle"], "ot_bound": bounds["test_blind"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument("--n_permutations", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.set_defaults(net="gcn", e_depth=2, c_depth=1, epochs=500, train_rate=0.3,
                        use_val=0, runs=5)
    args = parser.parse_args()

    device = get_device(args)
    dataset = load_graph_dataset(args.dataset, args.data_root, args.hetero_root)
    print(args)

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
