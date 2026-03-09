"""
Main script for measuring GNN generalization bound.

This script trains GNN models (GCN, SGC, etc.) on various graph datasets
and computes transductive generalization bounds based on optimal transport.
"""

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.logging import log

import numpy as np

import os
import os.path as osp
import argparse
from sklearn.metrics import roc_auc_score

import time
from datetime import datetime

from heterodata_loader import Hetero
from model import gnn, sgc

from utils import (
    random_planetoid_splits,
    random_planetoid_splits_wo_valid,
    cal_margin,
    W1_distance_simple,
    cal_all_margins,
    cal_max_change_with_known_labels,
    cal_max_change_class,
)


def RunExp(args, dataset, data, Net, device, save_name=None):
    """
    Run a single experiment: train a GNN model and compute generalization bounds.

    Args:
        args: Command line arguments containing hyperparameters
        dataset: PyG dataset object
        data: PyG data object containing graph structure and features
        Net: GNN model class (gnn or sgc)
        device: torch device (cuda or cpu)
        save_name: Optional path to save the best model

    Returns:
        test_metric: Test accuracy/AUC after training
        best_val_metric: Best validation metric achieved
        time_run: List of training times per epoch
        ge_info: Generalization bound information
    """

    def train(model, optimizer, data):
        """Single training step with cross-entropy loss."""
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        if data.y.max() == 1:
            loss = F.binary_cross_entropy_with_logits(
                out[data.train_mask], data.y[data.train_mask].float()
            )
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        del out

    def test(model, data):
        """Evaluate model on train/val/test splits."""
        model.eval()
        out = model(data.x, data.edge_index)

        accs, ce_losses, roc_aucs = [], [], []
        mask_list = [data.train_mask, data.val_mask, data.test_mask] if args.useval else [data.train_mask, data.test_mask]
        for mask in mask_list:
            if mask is None:
                continue
            if data.y.max() == 1:
                pred = (out.sigmoid() > 0.5).float()
                loss = F.binary_cross_entropy_with_logits(out[mask], data.y[mask].float())
                roc_auc = roc_auc_score(
                    data.y[mask].float().cpu().detach().numpy(),
                    out[mask].cpu().detach().numpy(),
                ).item()
                roc_aucs.append(roc_auc)
            else:
                pred = out.argmax(dim=-1)
                loss = F.cross_entropy(out[mask], data.y[mask])

            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
            ce_losses.append(loss.detach().cpu())

        return accs, ce_losses, roc_aucs

    # Initialize model
    model = Net(args, dataset, device)

    # Setup data splits based on dataset type
    if args.dataset in ("Cora", "CiteSeer", "PubMed", "Computers", "Photo"):
        data = random_planetoid_splits(data, args.seed)
        if not args.useval:
            if args.train_rate != 0:
                data = random_planetoid_splits_wo_valid(data, args.train_rate, args.seed)
            else:
                data.train_mask = data.train_mask | data.val_mask
        model, data = model.to(device), data.to(device)
    elif args.dataset in (
        "squirrel-filtered", "chameleon-filtered","roman-empire", "amazon-ratings"
    ):
        if not args.useval:
            if args.train_rate != 0:
                data = random_planetoid_splits_wo_valid(data, args.train_rate, args.seed)
            else:
                data.train_mask = data.train_mask | data.val_mask
        model = model.to(device)
        data.to_device()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_metric = test_metric = 0
    patience_count = 0

    if save_name is None:
        save_name = (
            "model/{}_{}_{:.2f}_".format(args.net, args.dataset, args.e_depth)
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".pt"
        )
    torch.save(model, save_name)

    time_run = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t_st = time.time()
        train(model, optimizer, data)
        time_epoch = time.time() - t_st
        time_run.append(time_epoch)

        # Evaluation
        accs, losses, roc_aucs = test(model, data)
        if len(roc_aucs) == 2:
            train_metric, tmp_test_metric = roc_aucs
        elif len(roc_aucs) == 3:
            train_metric, val_metric, tmp_test_metric = roc_aucs
        else:
            if args.useval:
                train_ce_loss, val_ce_loss, test_ce_loss = losses
                train_metric, val_metric, tmp_test_metric = accs
            else:
                train_ce_loss, test_ce_loss = losses
                train_metric, tmp_test_metric = accs

        # Update best model based on validation or test metric
        if not args.useval:
            test_metric = tmp_test_metric
            log(Epoch=epoch, Loss=train_ce_loss, Train=train_metric, Test=tmp_test_metric)
        else:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                test_metric = tmp_test_metric
                patience_count = 0
                torch.save(model, save_name)
            else:
                patience_count += 1
            log(Epoch=epoch, Loss=train_ce_loss, Train=train_metric, Val=val_metric, Test=test_metric)

        # Early stopping
        if args.early_stopping > 0 and patience_count >= args.early_stopping:
            print("Early stopping at epoch:", epoch)
            break
        if np.isnan(train_ce_loss.item()):
            print("NaN loss at epoch:", epoch)
            break

    # Load best model if using validation
    if args.useval:
        model = torch.load(save_name, weights_only=False)

    # Compute generalization bounds
    model.eval()
    features = model.feature(data.x, data.edge_index)
    out = model(data.x, data.edge_index)

    # Calculate margins for all samples
    margin_all = cal_all_margins(out)
    margin = cal_margin(out, data.y)
    margin_filtered = margin[data.train_mask]

    # Compute train/test error
    test_loss = (margin <= 0)[data.test_mask].float().mean()
    train_loss = (margin <= 0)[data.train_mask].float().mean()
    gen_loss = abs(test_loss - train_loss)

    # Compute class-wise maximum change (Lipschitz-like constant)
    global_max_change, global_quantile_90, global_quantile_50 = cal_max_change_with_known_labels(
        features, margin_all, data.train_mask, data.y.squeeze()
    )

    max_change_class = []
    for c in range(data.y.max() + 1):
        class_idx = (data.y == c)
        filtered_idx = (class_idx & data.train_mask) | data.test_mask
        max_change_class_c = cal_max_change_class(features[filtered_idx], margin_all[filtered_idx], c)
        max_change_class.append(max_change_class_c)
    max_change_class = torch.tensor(max_change_class)

    # Compute Wasserstein distance-based bounds
    k = args.mc_k  # Number of permutations for expectation approximation
    class_num = data.y.max() + 1
    permutation_list_oracle = []
    permutation_list_approx = []
    empty_c = torch.zeros(class_num)

    for idx in range(k):
        data_num = data.y.shape[0]
        test_num = int((data.test_mask).float().sum().item())
        train_num = data_num - test_num

        rnd_state = np.random.RandomState(idx)
        train_idx = rnd_state.choice(data_num, train_num, replace=False)
        test_idx = np.setdiff1d(np.arange(data_num), train_idx)
        train_mask = torch.zeros(data_num, dtype=torch.bool).to(device)
        test_mask = torch.zeros(data_num, dtype=torch.bool).to(device)
        train_mask[train_idx] = 1
        test_mask[test_idx] = 1

        classwise_list_oracle = []
        classwise_list_approx = []
        retry = False

        for c in range(class_num):
            class_idx = (data.y == c)
            train_c_idx = class_idx & train_mask
            test_c_idx = class_idx & test_mask
            train_c_idx_approx = train_c_idx & data.train_mask
            test_c_idx_approx = test_c_idx & data.train_mask

            if (train_c_idx.float().sum() == 0 or test_c_idx.float().sum() == 0 or
                train_c_idx_approx.float().sum() == 0 or test_c_idx_approx.float().sum() == 0):
                empty_c[c] = empty_c[c] + 1
                if empty_c[c] == 3:
                    raise ImportError(f"Class {c} is empty in train/test split")
                else:
                    retry = True
                    break

            w1_oracle = W1_distance_simple(
                features[train_c_idx].cpu().detach().numpy(),
                features[test_c_idx].cpu().detach().numpy()
            )
            w1_approx = W1_distance_simple(
                features[train_c_idx_approx].cpu().detach().numpy(),
                features[test_c_idx_approx].cpu().detach().numpy()
            )
            classwise_list_oracle.append((train_c_idx.float().sum() / train_num) * w1_oracle)
            classwise_list_approx.append((train_c_idx_approx.float().sum() / train_num) * w1_approx)

        if retry:
            continue
        else:
            permutation_list_oracle.append(classwise_list_oracle)
            permutation_list_approx.append(classwise_list_approx)

    # Compute final bounds
    w1_global = W1_distance_simple(
        features[data.train_mask].cpu().detach().numpy(),
        features[data.test_mask].cpu().detach().numpy()
    )
    global_bound = global_max_change * w1_global
    global_approx90_bound = global_quantile_90 * w1_global
    global_approx50_bound = global_quantile_50 * w1_global

    oracle_bound = (max_change_class.to(device) *
                   torch.tensor(permutation_list_oracle).to(device).mean(0)).sum()
    approx_bound = (max_change_class.to(device) *
                   torch.tensor(permutation_list_approx).to(device).mean(0)).sum()

    ge_info = [test_loss, train_loss, gen_loss, global_bound, global_approx90_bound,
               global_approx50_bound, oracle_bound, approx_bound]
    
    inter_class_WD = []
    for c1 in range(data.y.max()+1):
        for c2 in range(data.y.max()+1):
            if c1>c2:
                continue
            c1_idx = data.y==c1
            c2_idx = data.y==c2
            inter_class_WD.append(W1_distance_simple(features[c1_idx].cpu().detach().numpy(), features[c2_idx].cpu().detach().numpy()))
    sep_min = np.min(inter_class_WD)

    class_expectation_list = torch.tensor(permutation_list_oracle).to(device).mean(0)
    con_mean = class_expectation_list.mean().item()

    depth_ana_info = [w1_global, con_mean, sep_min]

    os.remove(save_name)

    return test_metric, best_val_metric, time_run, ge_info, depth_ana_info


def main():
    """Main function to run GNN training and bound computation experiments."""

    args.useval = bool(args.useval)
    os.makedirs("./model", exist_ok=True)

    print(args)
    print("-" * 50)

    # Select model architecture
    Net = sgc if args.net == "sgc" else gnn

    # Load dataset
    if args.dataset in ("Cora", "CiteSeer", "PubMed"):
        path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in ("Computers", "Photo"):
        path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Amazon")
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in (
        "squirrel-filtered", "chameleon-filtered","roman-empire", "amazon-ratings"
    ):
        dataset = Hetero(args, device)
        data = dataset.data
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    # Run experiment
    test_metric, best_val_metric, time_run, ge_info, depth_ana_info = RunExp(
        args, dataset, data, Net, device
    )

    # Print results
    print("=" * 50)
    print(f"{args.net} on {args.dataset}:")
    print(f"Test metric: {test_metric * 100:.2f}")
    print(f"Best val metric: {best_val_metric * 100:.2f}")
    print(f"Training time: {sum(time_run):.2f}s ({len(time_run)} epochs)")

    # Unpack ge_info
    test_loss, train_loss, gen_loss, global_bound, global_approx90_bound, \
        global_approx50_bound, oracle_bound, approx_bound = ge_info
    print("-" * 50)
    print("Generalization Info:")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Gen loss: {gen_loss:.4f}")
    print(f"  Global bound: {global_bound:.4f}")
    print(f"  Global approx90 bound: {global_approx90_bound:.4f}")
    print(f"  Global approx50 bound: {global_approx50_bound:.4f}")
    print(f"  Oracle bound: {oracle_bound:.4f}")
    print(f"  Approx bound: {approx_bound:.4f}")

    # Unpack depth_ana_info
    w1_global, con_mean, sep_min = depth_ana_info
    print("-" * 50)
    print("Depth Analysis Info:")
    print(f"  W1 global: {w1_global:.4f}")
    print(f"  Concentration mean: {con_mean:.4f}")
    print(f"  Separation min: {sep_min:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Generalization Bound Experiments")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--early_stopping", type=int, default=0, help="Early stopping patience (0 to disable)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")

    # Data configuration
    parser.add_argument("--useval", type=int, default=0, help="Use validation set (1) or not (0)")
    parser.add_argument("--train_rate", type=float, default=0.0, help="Custom train set ratio (0 for default)")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "Cora", "CiteSeer", "PubMed", "Computers", "Photo",
            "squirrel-filtered", "chameleon-filtered","roman-empire", "amazon-ratings",
        ],
        default="Cora",
        help="Dataset name"
    )
    parser.add_argument("--seed", type=int, default=2108550661, help="Random seed")

    # Model architecture
    parser.add_argument(
        "--net",
        type=str,
        choices=["gcn", "resgcn", "gcn2", "gat", "resgat", "sage", "ressage", "sgc"],
        default="gcn",
        help="GNN architecture"
    )
    parser.add_argument("--layernorm", type=str, choices=["None", "LayerNorm", "BatchNorm", "GraphNorm"],
                       default="None", help="Normalization layer type")
    parser.add_argument("--act", type=str, choices=["ReLU", "GeLU", "Leaky", "Tanh", "None"],
                       default="ReLU", help="Activation function")
    parser.add_argument("--heads", type=int, default=1, help="Number of attention heads (for GAT)")
    parser.add_argument("--e_depth", type=int, default=2, help="Encoder depth (number of GNN layers)")
    parser.add_argument("--c_depth", type=int, default=1, help="Classifier depth (number of MLP layers)")

    # Bound computation
    parser.add_argument("--mc_k", type=int, default=4, help="Number of permutations for bound estimation")

    # Device
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    main()
