"""Dataset loading for the graph and the non-graph (tabular / image) experiments.

Two loaders, one per family of experiments:

=============================  ==========================================================
:func:`load_graph_dataset`     Planetoid, Amazon and the heterophilic benchmarks, wrapped
                               in a :class:`GraphDataset` that owns the data split
                               :math:`\\pi` (see :mod:`otbound.splits`).
:class:`TabularDataset`        sklearn / OpenML / torchvision datasets with no graph,
                               used in the transductive setting of Section 6.
=============================  ==========================================================

Split protocol (Appendix E.1), shared by both loaders:

* ``use_val=True``  -- 60% / 20% / 20% train / val / test.  The heterophilic datasets
  keep their own published splits instead.
* ``use_val=False`` and ``train_rate > 0`` -- ``train_rate`` / ``1 - train_rate``
  train / test, no validation set.
* ``use_val=False`` and ``train_rate == 0`` -- the validation nodes are folded into the
  training set.
"""

import os.path as osp

import numpy as np
import torch

from .splits import random_planetoid_splits, random_split_wo_valid

__all__ = [
    "PLANETOID_DATASETS",
    "AMAZON_DATASETS",
    "HETERO_DATASETS",
    "ALL_DATASETS",
    "TABULAR_DATASETS",
    "GraphDataset",
    "load_graph_dataset",
    "TabularDataset",
]

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
_DEFAULT_DATA_ROOT = osp.join(_REPO_ROOT, "data")
_DEFAULT_HETERO_ROOT = osp.join(_REPO_ROOT, "new_data")

PLANETOID_DATASETS = ("Cora", "CiteSeer", "PubMed")
AMAZON_DATASETS = ("Computers", "Photo")
#: Heterophilic benchmarks of Platonov et al. (2023), read from ``<hetero_root>/<name>.npz``
#: with ``-`` replaced by ``_`` in the file name.
HETERO_DATASETS = (
    "squirrel", "squirrel-filtered", "chameleon", "chameleon-filtered",
    "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions",
)
ALL_DATASETS = PLANETOID_DATASETS + AMAZON_DATASETS + HETERO_DATASETS

TABULAR_DATASETS = (
    "iris", "wine", "breast_cancer", "digits", "covtype", "letter",
    "mnist", "fashion_mnist", "cifar10", "svhn",
)


# --------------------------------------------------------------------------- graphs


class GraphDataset:
    """A single graph together with its current data split :math:`\\pi`.

    ``data`` is a PyG :class:`~torch_geometric.data.Data` whose ``train_mask`` /
    ``val_mask`` / ``test_mask`` are rewritten by :meth:`resplit`.  For the
    heterophilic datasets, :meth:`next_split` advances to the next of the published
    splits, which :meth:`resplit` starts from.
    """

    def __init__(self, name, data, native_masks=None):
        self.name = name
        self.data = data
        self.num_features = int(data.x.size(-1))
        self.num_classes = int(data.y.max()) + 1
        # (train, val, test) stacks of shape [num_splits, num_nodes], or None.
        self._native_masks = native_masks
        self._cur_split = 0

    @property
    def num_native_splits(self):
        return 0 if self._native_masks is None else int(self._native_masks[0].size(0))

    def resplit(self, seed, use_val=False, train_rate=0.3):
        """Draw the data split :math:`\\pi` for one run; see the module docstring."""
        data = self.data
        if self._native_masks is None:
            random_planetoid_splits(data, seed)
        else:
            train, val, test = self._native_masks
            data.train_mask = train[self._cur_split].clone()
            data.val_mask = val[self._cur_split].clone()
            data.test_mask = test[self._cur_split].clone()

        if not use_val:
            if train_rate != 0:
                random_split_wo_valid(data, train_rate, seed)
            else:
                data.train_mask = data.train_mask | data.val_mask
            data.val_mask = torch.zeros_like(data.train_mask)
        return data

    def next_split(self):
        """Advance to the next published split (heterophilic datasets only)."""
        if self._native_masks is not None:
            self._cur_split = (self._cur_split + 1) % self.num_native_splits


def _load_hetero(name, hetero_root):
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    path = osp.join(hetero_root, "{}.npz".format(name.replace("-", "_")))
    if not osp.isfile(path):
        raise FileNotFoundError("heterophilic dataset file not found: {}".format(path))
    raw = np.load(path)

    x = torch.as_tensor(raw["node_features"], dtype=torch.float32)
    y = torch.as_tensor(np.asarray(raw["node_labels"], dtype=np.int64).flatten())
    edge_index = to_undirected(torch.as_tensor(raw["edges"], dtype=torch.long).T,
                               num_nodes=x.size(0))
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.size(0))

    native = tuple(torch.as_tensor(raw[key], dtype=torch.bool)
                   for key in ("train_masks", "val_masks", "test_masks"))
    return data, native


def load_graph_dataset(name, data_root=None, hetero_root=None):
    """Load a node-classification graph by name.

    Planetoid and Amazon graphs are downloaded by PyG into ``data_root`` (default
    ``<repo>/data``) with row-normalized features; the heterophilic graphs are read
    from ``hetero_root`` (default ``<repo>/new_data``).
    """
    data_root = data_root or _DEFAULT_DATA_ROOT
    hetero_root = hetero_root or _DEFAULT_HETERO_ROOT

    if name in PLANETOID_DATASETS or name in AMAZON_DATASETS:
        import torch_geometric.transforms as T
        from torch_geometric.datasets import Amazon, Planetoid

        if name in PLANETOID_DATASETS:
            pyg = Planetoid(osp.join(data_root, "Planetoid"), name,
                            transform=T.NormalizeFeatures())
        else:
            pyg = Amazon(osp.join(data_root, "Amazon"), name,
                         transform=T.NormalizeFeatures())
        return GraphDataset(name, pyg[0])

    if name in HETERO_DATASETS:
        data, native = _load_hetero(name, hetero_root)
        return GraphDataset(name, data, native_masks=native)

    raise ValueError("unknown graph dataset {!r}; choose from {}".format(
        name, ", ".join(ALL_DATASETS)))


# ----------------------------------------------------------------- tabular / images


class TabularDataset:
    """A dataset with no graph, used transductively.

    ``X`` is always a flat ``(N, D)`` float tensor, standardized per feature (per pixel
    for images).  For image datasets ``image_shape`` is ``(C, H, W)`` so the CNN
    encoders of :mod:`otbound.tabular` can reshape it; otherwise it is ``None``.
    """

    def __init__(self, name, data_root=None):
        if name not in TABULAR_DATASETS:
            raise ValueError("unknown tabular dataset {!r}; choose from {}".format(
                name, ", ".join(TABULAR_DATASETS)))
        self.name = name
        self.data_root = data_root or _DEFAULT_DATA_ROOT
        self.image_shape = None
        self._load()

    def _load(self):
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        X, y = self._fetch()
        if self.name == "letter":
            y = LabelEncoder().fit_transform(y)
        X = StandardScaler().fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)
        self.num_features = int(X.shape[1])
        self.num_classes = int(len(np.unique(y)))
        self.num_samples = int(len(y))

    def _fetch(self):
        from sklearn import datasets as skd

        name = self.name
        if name in ("iris", "wine", "breast_cancer", "digits"):
            raw = getattr(skd, "load_" + name)()
            if name == "digits":
                self.image_shape = (1, 8, 8)
            return raw.data, raw.target
        if name == "covtype":
            raw = skd.fetch_covtype()
            return raw.data, raw.target - raw.target.min()
        if name == "letter":
            raw = skd.fetch_openml("letter", version=1, as_frame=False)
            return raw.data, raw.target
        if name in ("mnist", "fashion_mnist"):
            openml_name = "mnist_784" if name == "mnist" else "Fashion-MNIST"
            raw = skd.fetch_openml(openml_name, version=1, as_frame=False)
            self.image_shape = (1, 28, 28)
            return raw.data, raw.target.astype(int)
        if name == "cifar10":
            from torchvision.datasets import CIFAR10

            train = CIFAR10(root=self.data_root, train=True, download=True)
            test = CIFAR10(root=self.data_root, train=False, download=True)
            X = np.concatenate([train.data, test.data], axis=0)  # (N, H, W, C)
            y = np.concatenate([np.array(train.targets), np.array(test.targets)])
            X = X.transpose(0, 3, 1, 2).reshape(len(y), -1).astype(np.float32) / 255.0
            self.image_shape = (3, 32, 32)
            return X, y
        if name == "svhn":
            from torchvision.datasets import SVHN

            root = osp.join(self.data_root, "svhn")
            train = SVHN(root=root, split="train", download=True)
            test = SVHN(root=root, split="test", download=True)
            X = np.concatenate([train.data, test.data], axis=0)  # already (N, C, H, W)
            y = np.concatenate([train.labels, test.labels])
            X = X.reshape(len(y), -1).astype(np.float32) / 255.0
            self.image_shape = (3, 32, 32)
            return X, y
        raise AssertionError(name)

    def get_split(self, seed, use_val=True, train_rate=0.0):
        """``(train_mask, val_mask, test_mask)`` for one run; see the module docstring.

        ``val_mask`` is all-false when ``use_val`` is off.
        """
        n = self.num_samples
        indices = np.arange(n)
        np.random.RandomState(seed).shuffle(indices)

        train_size = int(round(0.6 * n))
        val_size = int(round(0.2 * n))

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        if use_val:
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
        elif train_rate != 0:
            cut = int(round(train_rate * n))
            train_mask[indices[:cut]] = True
            test_mask[indices[cut:]] = True
        else:
            train_mask[indices[:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
        return train_mask, val_mask, test_mask
