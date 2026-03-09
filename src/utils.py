"""
Utility functions for GNN generalization bound computation.

This module provides:
- Data splitting utilities
- Margin computation functions
- Wasserstein distance computation
- Maximum change (Lipschitz-like) computation for generalization bounds
"""

import torch
import numpy as np
from sklearn.metrics import pairwise_distances

try:
    import ot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False


# =============================================================================
# Data Splitting Utilities
# =============================================================================

def index_to_mask(index, size):
    """
    Convert index tensor to boolean mask.

    Args:
        index: Tensor of indices
        size: Size of the output mask

    Returns:
        Boolean mask tensor of shape [size]
    """
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, seed=12134):
    """
    Create random train/val/test splits (60/20/20) for Planetoid-style datasets.

    Args:
        data: PyG data object
        seed: Random seed for reproducibility

    Returns:
        Data object with train_mask, val_mask, test_mask attributes
    """
    train_lb = int(round(0.6 * len(data.y)))
    val_lb = int(round(0.2 * len(data.y)))

    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(data.y.shape[0], train_lb, replace=False)
    rest_index = np.setdiff1d(np.arange(data.y.shape[0]), train_idx)
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = np.setdiff1d(rest_index, val_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data


def random_planetoid_splits_wo_valid(data, train_rate, seed=12134):
    """
    Create random train/test splits without validation set.

    Args:
        data: PyG data object
        train_rate: Proportion of data to use for training
        seed: Random seed for reproducibility

    Returns:
        Modified data object with masks
    """
    train_lb = int(round(train_rate * len(data.y)))

    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(data.y.shape[0], train_lb, replace=False)
    test_idx = np.setdiff1d(np.arange(data.y.shape[0]), train_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    return data


# =============================================================================
# Margin Computation
# =============================================================================

def cal_margin(score, y):
    """
    Compute classification margin for each sample.

    Margin = score of true class - max score of other classes
    Positive margin means correct classification.

    Args:
        score: Logit tensor [N, K] where K is number of classes
        y: True label tensor [N]

    Returns:
        Margin tensor [N]
    """
    N = score.shape[0]
    device = score.device

    # Get true class scores
    true_scores = score.gather(1, y.unsqueeze(1)).squeeze(1)

    # Create a copy and mask out true class
    score_copy = score.clone()
    score_copy[torch.arange(N, device=device), y] = float('-inf')

    # Get maximum of other class scores
    max_other_scores = torch.max(score_copy, dim=1)[0]

    return true_scores - max_other_scores


def cal_all_margins(score):
    """
    Compute margins for all possible class assignments.

    For each sample i and class c, computes the margin assuming c is the true label.

    Args:
        score: Logit tensor [N, K]

    Returns:
        Margin tensor [N, K] where [i, c] is margin of sample i if class c were true
    """
    N, K = score.shape
    device = score.device

    # Expand scores for all class combinations
    score_expanded = score.unsqueeze(2).expand(N, K, K)

    # Create diagonal mask
    mask = torch.eye(K, device=device).bool().unsqueeze(0).expand(N, K, K)

    # Mask out diagonal (true class) with -inf
    score_masked = score_expanded.clone()
    score_masked[mask] = float('-inf')

    # Max of other classes for each potential true class
    max_other_scores = torch.max(score_masked, dim=1)[0]

    return score - max_other_scores


# =============================================================================
# Wasserstein Distance
# =============================================================================

def W1_distance_simple(feats_0, feats_1):
    """
    Compute exact Wasserstein-1 distance between two feature distributions.

    Uses the POT (Python Optimal Transport) library with Earth Mover's Distance.

    Args:
        feats_0: Feature array [n0, d] (numpy)
        feats_1: Feature array [n1, d] (numpy)

    Returns:
        W1 distance (float)
    """
    if not OT_AVAILABLE:
        raise ImportError("POT library required. Install with: pip install POT")

    k0, k1 = len(feats_0), len(feats_1)
    M = pairwise_distances(feats_0, feats_1)
    a = np.ones(k0) / k0
    b = np.ones(k1) / k1
    return ot.emd2(a, b, M, numItermax=1000000)


# =============================================================================
# Maximum Change Computation (Lipschitz-like Constants)
# =============================================================================

def cal_max_change_with_known_labels(features, margin_all, known_mask, true_labels,
                                      batch_size=None):
    """
    Compute maximum change rate of margin with respect to feature distance.

    This is a key quantity in the generalization bound:
    max_{i in train, j in test, y} |margin(i, y_i) - margin(j, y)| / ||f(i) - f(j)||

    Args:
        features: Node embeddings [N, d]
        margin_all: Margins for all classes [N, K]
        known_mask: Boolean mask for labeled (train) nodes [N]
        true_labels: True labels [N]
        batch_size: Batch size for memory-efficient computation

    Returns:
        Tuple of (max_change, quantile_90, quantile_50)
    """
    N, d = features.shape
    K = margin_all.shape[1]
    device = features.device

    # Split into train (known) and test (unknown) sets
    train_mask = known_mask
    test_mask = ~known_mask

    train_indices = torch.where(train_mask)[0]
    test_indices = torch.where(test_mask)[0]

    if len(train_indices) == 0 or len(test_indices) == 0:
        return (0.0, 0.0, 0.0)

    train_features = features[train_indices]
    test_features = features[test_indices]
    train_margins = margin_all[train_indices]
    test_margins = margin_all[test_indices]
    train_labels = true_labels[train_indices]

    # Auto batch size
    if batch_size is None:
        n_train, n_test = len(train_indices), len(test_indices)
        estimated_memory = n_train * n_test * K * 4
        batch_size = max(1, min(100, int(1e8 / (n_test * K)))) if estimated_memory > 1e9 else n_train

    max_change_global = 0.0
    all_changes = []

    # Process train nodes in batches
    for i in range(0, len(train_indices), batch_size):
        end_i = min(i + batch_size, len(train_indices))

        batch_train_features = train_features[i:end_i]
        batch_train_margins = train_margins[i:end_i]
        batch_train_labels = train_labels[i:end_i]

        # Feature distances
        feature_diffs = batch_train_features.unsqueeze(1) - test_features.unsqueeze(0)
        feature_norms = torch.norm(feature_diffs, dim=2)
        feature_norms = feature_norms.masked_fill(feature_norms < 1e-6, 1e10)

        # Train margins for true labels
        batch_train_margins_yi = batch_train_margins.gather(1, batch_train_labels.unsqueeze(1))

        # Margin differences
        margin_diffs = torch.abs(batch_train_margins_yi.unsqueeze(2) - test_margins.unsqueeze(0))

        # Change rates
        change_rates = margin_diffs / feature_norms.unsqueeze(2)

        # Update max
        batch_max = torch.max(change_rates).item()
        max_change_global = max(max_change_global, batch_max)

        # Collect for quantiles
        flattened = change_rates.flatten()
        if len(flattened) > 10000:
            indices = torch.randperm(len(flattened), device=device)[:5000]
            flattened = flattened[indices]
        all_changes.append(flattened.detach().cpu())

        # Memory cleanup
        del feature_diffs, feature_norms, margin_diffs, change_rates
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Compute quantiles
    if all_changes:
        all_changes_tensor = torch.cat(all_changes, dim=0)
        if len(all_changes_tensor) > 50000:
            indices = torch.randperm(len(all_changes_tensor))[:50000]
            all_changes_tensor = all_changes_tensor[indices]

        quantile_90 = torch.quantile(all_changes_tensor, 0.9).item()
        quantile_50 = torch.quantile(all_changes_tensor, 0.5).item()

        return (max_change_global, quantile_90, quantile_50)

    return (max_change_global, 0.0, 0.0)


def cal_max_change_class(features, margin_all, c, batch_size=None):
    """
    Compute maximum change rate for a specific class.

    For class c, computes:
    max_{i,j} |margin(i, c) - margin(j, c)| / ||f(i) - f(j)||

    Args:
        features: Node embeddings [N, d]
        margin_all: Margins for all classes [N, K]
        c: Target class index
        batch_size: Batch size for memory-efficient computation

    Returns:
        Maximum change rate for class c (float)
    """
    N, d = features.shape
    device = features.device

    if batch_size is None:
        if device.type == 'cuda':
            available = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated()
            free = available - allocated
            batch_size = max(10, min(N, int(free * 0.3 / (N * d * 4))))
        else:
            batch_size = min(50, N)

    max_change = 0.0
    margins_c = margin_all[:, c]

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_features = features[i:end_i]
        batch_margins = margins_c[i:end_i]

        # Feature distances
        feature_diffs = batch_features.unsqueeze(1) - features.unsqueeze(0)
        feature_norms = torch.norm(feature_diffs, dim=2)

        # Margin differences
        margin_diffs = torch.abs(batch_margins.unsqueeze(1) - margins_c.unsqueeze(0))

        # Mask self-comparisons
        feature_norms = feature_norms.masked_fill(feature_norms < 1e-6, float('inf'))

        # Change rates
        change_rates = margin_diffs / feature_norms
        change_rates = change_rates.masked_fill(feature_norms == float('inf'), 0)

        max_change = max(max_change, torch.max(change_rates).item())

        del feature_diffs, feature_norms, margin_diffs, change_rates
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return max_change
