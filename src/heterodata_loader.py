"""
Data loader for heterophilic graph datasets.

Loads datasets from the 'new_data' directory including:
- squirrel, squirrel-filtered
- chameleon, chameleon-filtered
- roman-empire, amazon-ratings
- minesweeper, tolokers, questions

Reference: https://github.com/CUAI/Non-Homophily-Large-Scale
"""

import torch
from torch_geometric.utils import to_undirected
import numpy as np
import os
from os import path

# Path to heterophilic dataset files
DATAPATH = path.abspath(path.join(path.dirname(__file__), "..")) + "/new_data/"


class Data:
    """
    Simple data container for graph data.

    Attributes:
        edge_index: Graph connectivity [2, num_edges]
        x: Node features [num_nodes, num_features]
        y: Node labels [num_nodes]
        train_mask, val_mask, test_mask: Boolean masks for data splits
    """

    def __init__(self):
        self.edge_index = None
        self.edge_attr = None
        self.x = None
        self.y = None
        self.num_features = None
        self.num_classes = None
        self.num_nodes = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.device = None

    def to_device(self):
        """Move all tensors to the specified device."""
        self.edge_index = self.edge_index.to(self.device)
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.val_mask = self.val_mask.to(self.device)
        self.test_mask = self.test_mask.to(self.device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(self.device)


class Hetero:
    """
    Heterophilic graph dataset loader.

    Loads pre-processed heterophilic datasets with multiple train/val/test splits.

    Args:
        args: Arguments containing dataset name
        device: Torch device

    Attributes:
        data: Data object containing graph and features
        num_features: Number of input features
        num_classes: Number of classes
        num_data_splits: Number of available data splits
    """

    def __init__(self, args, device):
        print("Preparing data...")

        # Load dataset from npz file
        data = np.load(os.path.join(DATAPATH, f'{args.dataset.replace("-", "_")}.npz'))

        self.data = Data()
        self.data.edge_index = to_undirected(
            torch.tensor(data["edges"], dtype=torch.int64).T
        )
        self.data.x = torch.from_numpy(data["node_features"])
        self.data.y = torch.from_numpy(
            np.array(data["node_labels"], dtype=int).flatten()
        )

        # Load pre-defined splits
        self.train_masks = torch.tensor(data["train_masks"])
        self.val_masks = torch.tensor(data["val_masks"])
        self.test_masks = torch.tensor(data["test_masks"])

        self.cur_data_split = 0
        self.data.train_mask = self.train_masks[self.cur_data_split]
        self.data.val_mask = self.val_masks[self.cur_data_split]
        self.data.test_mask = self.test_masks[self.cur_data_split]
        self.num_data_splits = len(self.train_masks)

        # Set metadata
        self.num_features = self.data.x.shape[-1]
        self.num_classes = self.data.y.max() + 1
        self.data.num_features = self.num_features
        self.data.num_classes = self.num_classes
        self.data.num_nodes = self.data.x.shape[0]

        self.name = args.dataset
        self.device = device
        self.data.device = device

    def next_data_split(self):
        """Cycle to the next train/val/test split."""
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits
        self.data.train_mask = self.train_masks[self.cur_data_split]
        self.data.val_mask = self.val_masks[self.cur_data_split]
        self.data.test_mask = self.test_masks[self.cur_data_split]
