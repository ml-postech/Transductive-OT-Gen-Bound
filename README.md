# Transductive Generalization via Optimal Transport and Its Application to Graph Node Classification

Official implementation of [Transductive Generalization via Optimal Transport and Its Application to Graph Node Classification](https://arxiv.org/abs/2603.09257).

## Overview

This repository provides code for computing and evaluating transductive generalization bounds for GNN node classification. The bounds are based on:
- Wasserstein distance between train and test node feature distributions
- Class-wise margin analysis
- Lipschitz-like constants of the learned classifier

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.12
- PyTorch Geometric >= 2.0
- POT (Python Optimal Transport) >= 0.8

### Install dependencies
```bash
pip install torch torch-geometric
pip install POT scikit-learn
```

## Project Structure

```
├── src/
│   ├── main.py              # Main training and evaluation script
│   ├── model.py             # GNN model architectures (GCN, GAT, SGC, etc.)
│   ├── utils.py             # Utility functions for bounds computation
│   └── heterodata_loader.py # Heterophilic dataset loader
├── data/                    # Planetoid and Amazon datasets (auto-downloaded)
├── new_data/                # Heterophilic graph datasets (.npz files)
└── README.md
```

## Usage

### Basic Training
```bash
cd src

# Train GCN on Cora
python main.py --dataset Cora --net gcn --e_depth 2 --epochs 200

# Train SGC on CiteSeer
python main.py --dataset CiteSeer --net sgc --e_depth 2 --epochs 200

# Train on heterophilic datasets
python main.py --dataset roman-empire --net gcn --e_depth 2
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name | Cora |
| `--net` | GNN architecture (gcn, gat, sgc, sage, gcn2) | gcn |
| `--e_depth` | Number of GNN layers | 2 |
| `--c_depth` | Number of classifier MLP layers | 1 |
| `--hidden_dim` | Hidden dimension | 64 |
| `--lr` | Learning rate | 0.01 |
| `--epochs` | Maximum training epochs | 1000 |
| `--early_stopping` | Early stopping patience (0 to disable) | 0 |
| `--useval` | Use validation set (1) or not (0) | 0 |
| `--mc_k` | Number of permutations for bound estimation | 4 |
