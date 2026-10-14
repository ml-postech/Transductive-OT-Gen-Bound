# Transductive Generalization for GNNs via Optimal Transport

Official implementation of **Transductive Generalization for GNNs via Optimal Transport** (NeurIPS 2026) [[arXiv](https://arxiv.org/abs/2603.09257)].

MoonJeong Park\*, Seungbeom Lee\*, Kyungmin Kim, Jaeseung Heo, Seunghyuk Cho, Shouheng Li, Sangdon Park, Dongwoo Kim

\* Equal contribution

> **Note.** This is the code for the revised, accepted version of the paper. The code for the earlier preprint is kept under the [`v1-arxiv`](../../tree/v1-arxiv) tag.

## Overview

We bound the transductive generalization gap of a node classifier with optimal transport. The bound links generalization to two properties of the learned feature space: how tightly each class is concentrated, and how far apart the classes are. The code covers four uses of the bound:

- **Evaluating** the bound and measuring how well it rank-correlates with the empirical generalization gap (Table 1), on graphs, link prediction, and non-graph data.
- **Analyzing depth.** Intra-class concentration $W_C$ and inter-class separation $W_S$ are tracked as message passing deepens (Figure 1). This accounts for the non-monotone relationship between depth and generalization.
- **Regularizing** training with a differentiable surrogate of the bound (Table 2).
- **Semi-supervised learning** with the regularizer added on top of standard SSL algorithms (Section 6; coming soon, see [Status](#status)).

## Installation

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.8, PyTorch 1.13.1, PyTorch Geometric 2.4.0, POT 0.9.5 and scikit-learn 1.3.0. Install `torch` and `torch_geometric` (with `torch_scatter`) to match your CUDA version first; see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Repository structure

```
├── otbound/                  # library: everything the paper defines
│   ├── splits.py             # data split π and estimation permutation π' (App. B)
│   ├── wasserstein.py        # empirical W1: exact and Sinkhorn
│   ├── margin.py             # margins and M_c(f, φ) (Thm. 1)
│   ├── bound.py              # the OT bound (Thm. 1, App. B)
│   ├── regularizer.py        # differentiable surrogate R (Sec. 5.2)
│   ├── depth.py              # W_C and W_S (Sec. 5.3)
│   ├── baselines.py          # PAC-Bayes, Rademacher, uniform-stability baselines
│   ├── link.py               # the bound for link prediction (App. C.1)
│   ├── models.py             # GNN encoders and classifiers
│   ├── tabular.py            # MLP / CNN encoders for non-graph data
│   ├── training.py           # training loops
│   ├── data.py               # dataset loading and splits
│   └── ssl/                  # semi-supervised hooks (Sec. 6)
├── experiments/              # one script per table / figure
│   ├── correlation.py        # Table 1
│   ├── depth.py              # Figure 1, App. C.3
│   ├── regularize.py         # Table 2
│   ├── link.py               # Table 6 (App. C.1)
│   ├── tabular_correlation.py  # Tables 7, 8; Table 3 (supervised)
│   ├── tabular_regularize.py   # Table 9; Table 4 (supervised)
│   └── semisup.py            # Tables 3, 4 (semi-supervised) — skeleton
├── data/                     # Planetoid / Amazon (PyG downloads here if missing)
└── new_data/                 # heterophilic graphs (.npz)
```

## Datasets

- **Planetoid** (`Cora`, `CiteSeer`, `PubMed`) and **Amazon** (`Computers`, `Photo`) are read from `data/`. PyG downloads them there if they are missing.
- **Heterophilic graphs** (`squirrel`, `squirrel-filtered`, `chameleon`, `chameleon-filtered`, `roman-empire`, `amazon-ratings`, `minesweeper`, `tolokers`, `questions`) are read from the `.npz` files in `new_data/`. These are the files released by [Platonov et al. (2023)](https://github.com/yandex-research/heterophilous-graphs).
- **Non-graph data** (`iris`, `wine`, `breast_cancer`, `digits`, `covtype`, `letter`, `mnist`, `fashion_mnist`, `cifar10`, `svhn`) are fetched by scikit-learn, OpenML or torchvision on first use.

Use `--data_root` and `--hetero_root` to point to other locations.

## Usage

Run every script from the repository root. Each call trains one configuration over `--runs` random splits, prints a summary, and appends a row to `--result_csv` if one is given. For the rank correlations, sweep the configurations listed in each script's docstring, then correlate the rows of the CSV.

**Bound vs. generalization gap (Table 1)**

```bash
python experiments/correlation.py --dataset Cora --net gcn --e_depth 4 --c_depth 2 \
    --train_rate 0.3 --runs 5 --result_csv results/correlation.csv

# Rademacher baseline (head-free SGC/GCN only)
python experiments/correlation.py --dataset Cora --net sgc --c_depth 0 --baseline_rc 1
```

**Depth analysis (Figure 1)**

```bash
for L in 1 4 8 12 16 20 24 28 32; do
  python experiments/depth.py --dataset Cora --net sgc --e_depth $L \
      --result_csv results/depth.csv
done
```

**OT regularizer (Table 2)**

```bash
python experiments/regularize.py --dataset roman-empire --net resgcn --reg_weight 1e-3
python experiments/regularize.py --dataset roman-empire --net resgcn --regularizer None  # baseline
```

**Link prediction (Table 6)**

```bash
python experiments/link.py --dataset Cora --net gcn --e_depth 2 --c_depth 1
```

**Beyond graphs (Tables 7–9; supervised columns of Tables 3–4)**

```bash
python experiments/tabular_correlation.py --dataset covtype --net mlp --e_depth 4 --c_depth 2
python experiments/tabular_regularize.py --dataset cifar10 --net rescnn --reg_weight 1e-3
```

### Common arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | dataset name (see above) | required |
| `--net` | encoder: `gcn`, `gat`, `sage`, `sgc`, `gcn2`, `mlp`; a `res` prefix adds skip connections | `gcn` |
| `--e_depth` / `--c_depth` | encoder / classifier depth | 8 / 2 |
| `--hidden_dim` | hidden dimension | 64 |
| `--train_rate` | labelled fraction of the transductive split (0 keeps the dataset's own split) | 0.3 |
| `--use_val` | 1 for a 60/20/20 split with best-validation model selection | 0 |
| `--runs` | number of random splits | 5 |
| `--n_permutations` | number of π' samples used to estimate the bound (App. B) | 4 |
| `--reg_weight` | λ in L = L_CE + λR | 1e-3 |

Each script sets its own defaults to match the protocol in the paper. Run `python experiments/<script>.py --help` for the full list.

## Status

- [x] OT bound, oracle and test-blind estimators (Theorem 1)
- [x] Baselines: PAC-Bayes, Rademacher, uniform stability
- [x] Depth analysis ($W_C$, $W_S$)
- [x] OT regularizer: graphs, tabular and image data
- [x] Link prediction
- [ ] Semi-supervised experiments (Section 6). `otbound/ssl/buffer.py` and `otbound/ssl/regularizer.py` are complete. The four SSL algorithms in `otbound/ssl/algorithms.py` and the data pipeline in `experiments/semisup.py` are still skeletons and raise `NotImplementedError`; the full implementation will be added soon.

## Citation

```bibtex
@inproceedings{park2026transductive,
  title     = {Transductive Generalization for {GNN}s via Optimal Transport},
  author    = {Park, MoonJeong and Lee, Seungbeom and Kim, Kyungmin and Heo, Jaeseung and
               Cho, Seunghyuk and Li, Shouheng and Park, Sangdon and Kim, Dongwoo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
