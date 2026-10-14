"""Shared CLI arguments, seeds and CSV output for the experiment scripts.

Importing this also puts the repository root on ``sys.path``, so the scripts run as
``python experiments/<name>.py`` from anywhere without installing the package.
"""

import csv
import os
import os.path as osp
import sys

import torch

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: Fixed split seeds, so run ``i`` uses the same split across every configuration and
#: the rank correlations of Table 1 compare like with like.
SEEDS = [
    1941488137, 4198936517, 983997847, 4023022221, 4019585660,
    2108550661, 1648766618, 629014539, 3212139042, 2424918363,
]

__all__ = ["REPO_ROOT", "SEEDS", "add_common_args", "add_regularizer_args",
           "get_device", "append_csv_row", "summarize"]


def add_common_args(parser):
    """Arguments shared by every experiment: data, architecture, optimization."""
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=str, required=True,
                      help="dataset name; see otbound.data.ALL_DATASETS")
    data.add_argument("--data_root", type=str, default=None,
                      help="where PyG downloads Planetoid/Amazon (default <repo>/data)")
    data.add_argument("--hetero_root", type=str, default=None,
                      help="directory of the heterophilic .npz files "
                           "(default <repo>/new_data)")
    data.add_argument("--train_rate", type=float, default=0.3,
                      help="labelled fraction of the transductive split; "
                           "0 keeps the dataset's own split")
    data.add_argument("--use_val", type=int, default=0,
                      help="1 to carve out a validation set (60/20/20) and report the "
                           "best-validation checkpoint")
    data.add_argument("--runs", type=int, default=5,
                      help="number of random splits to average over")

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--net", type=str, default="gcn",
                      choices=["gcn", "resgcn", "gcn2", "gat", "resgat", "sage",
                               "ressage", "sgc", "mlp", "resmlp"],
                      help="encoder; the 'res' prefix adds skip connections")
    arch.add_argument("--e_depth", type=int, default=8, help="encoder depth L")
    arch.add_argument("--c_depth", type=int, default=2, help="classifier depth")
    arch.add_argument("--hidden_dim", type=int, default=64)
    arch.add_argument("--heads", type=int, default=1, help="attention heads, GAT only")
    arch.add_argument("--act", type=str, default="ReLU",
                      choices=["ReLU", "GeLU", "Leaky", "Tanh", "None"])
    arch.add_argument("--layernorm", type=str, default="None",
                      choices=["None", "LayerNorm", "BatchNorm", "GraphNorm"])

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=500)
    optim.add_argument("--lr", type=float, default=0.01)
    optim.add_argument("--weight_decay", type=float, default=0.0)
    optim.add_argument("--early_stopping", type=int, default=0,
                       help="stop after this many epochs without validation improvement; "
                            "0 disables")

    out = parser.add_argument_group("output")
    out.add_argument("--result_csv", type=str, default=None,
                     help="append one summary row to this CSV")
    out.add_argument("--no_cuda", action="store_true")
    out.add_argument("--verbose", action="store_true")
    return parser


def add_regularizer_args(parser):
    """Arguments controlling the Section 5.2 regularizer."""
    group = parser.add_argument_group("regularizer")
    group.add_argument("--regularizer", type=str, default="ot", choices=["None", "ot"],
                       help="'None' trains the unregularized baseline")
    group.add_argument("--reg_weight", type=float, default=1e-3,
                       help="lambda in L = L_task + lambda * R")
    group.add_argument("--sinkhorn_eps", type=float, default=0.05,
                       help="entropic regularization of the Sinkhorn W1 surrogate")
    group.add_argument("--sinkhorn_iter", type=int, default=50)
    group.add_argument("--soft_temp", type=float, default=1.0,
                       help="temperature of the log-sum-exp standing in for the max in M_c")
    group.add_argument("--reg_n_sample", type=int, default=200,
                       help="points sampled per class when estimating M_c")
    return parser


def get_device(args):
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def append_csv_row(path, row):
    """Append ``row`` to ``path``, writing a header if the file is new."""
    if path is None:
        return
    directory = osp.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = osp.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize(values):
    """``(mean, std)`` of a list of floats, as plain Python floats."""
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float64)
    std = float(tensor.std()) if tensor.numel() > 1 else 0.0
    return float(tensor.mean()), std
