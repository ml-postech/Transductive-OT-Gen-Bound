"""Transductive generalization bounds we compare against in Table 1.

Two baselines are computable for the architectures in the paper:

**PAC-Bayes** (Begin et al., 2014)
    A transductive PAC-Bayesian bound.  Instantiated, as is standard, with a Gaussian
    posterior centred at the learned weights and a zero-mean Gaussian prior, so the
    bound reduces to :math:`\\mathrm{KL}/m`.  Applies to any architecture.

**Transductive Rademacher complexity** (El-Yaniv and Pechyony, 2009; Esser et al., 2021)
    The general transductive Rademacher bound is not computable, so we use the
    GNN-specific upper bound of Esser et al. (2021), which is derived only for SGC and
    GCN and requires the encoder to output logits directly (no MLP classifier head).

:func:`cong_stability_bound` is the uniform-stability bound of Cong et al. (2021), used
in Section 5.3 as the depth-monotone foil to our non-monotone one.

Bounds based on VC dimension, permutational Rademacher complexity, or stability of the
general kind are intractable to evaluate and are not implemented.
"""

import math

import numpy as np
import torch
from torch_geometric.nn import SGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

__all__ = [
    "gaussian_kl",
    "pac_bayes_bound",
    "rademacher_bound",
    "cong_stability_bound",
    "a_hat_inf_norm",
    "a_hat_x",
]


def gaussian_kl(mu, sigma_q, sigma_p):
    """:math:`\\mathrm{KL}(\\mathcal{N}(\\mu, \\sigma_q^2 I) \\,\\|\\, \\mathcal{N}(0, \\sigma_p^2 I))`."""
    d = mu.numel()
    term1 = (sigma_q ** 2) / (sigma_p ** 2)
    term2 = (mu ** 2).sum() / (sigma_p ** 2)
    term3 = -d
    term4 = d * torch.log(torch.tensor(sigma_p ** 2 / sigma_q ** 2))
    return 0.5 * (term1 * d + term2 + term3 + term4)


def pac_bayes_bound(model, num_train, sigma_q=1e-3, sigma_p=1.0):
    """PAC-Bayesian baseline: :math:`\\mathrm{KL}(Q \\| P) / m`.

    The posterior :math:`Q` is an isotropic Gaussian centred at the learned parameters
    and the prior :math:`P` is the same Gaussian centred at the origin, so the KL term
    is driven by the parameter norm.
    """
    mu = torch.cat([p.data.view(-1) for p in model.parameters()])
    return float(gaussian_kl(mu, sigma_q, sigma_p) / num_train)


def a_hat_inf_norm(num_nodes, edge_index, edge_weight=None):
    """:math:`\\|\\hat{A}\\|_\\infty` for the renormalized adjacency, without ever
    materializing :math:`\\hat{A}`.

    The infinity norm is the largest absolute row sum, which can be read straight off
    the normalized edge weights.
    """
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device)

    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    row_sums = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
    return float(row_sums.max())


def a_hat_x(x, edge_index, edge_weight=None):
    """One propagation step :math:`\\hat{A} X`."""
    conv = SGConv(in_channels=x.shape[0], out_channels=1, K=1, cached=True)
    edge_index_tmp, edge_weight_tmp = gcn_norm(
        edge_index, edge_weight, x.size(conv.node_dim), False,
        conv.add_self_loops, conv.flow, dtype=x.dtype,
    )
    return conv.propagate(edge_index_tmp, x=x, edge_weight=edge_weight_tmp)


def rademacher_bound(model, data, net, e_depth, num_features, num_nodes, num_train,
                     num_test):
    """Transductive Rademacher complexity bound of Esser et al. (2021).

    Only defined for ``net in {"sgc", "gcn"}``, and only when the encoder emits logits
    directly -- the derivation does not cover an MLP head, so runs that use this
    baseline are configured with ``--c_depth 0``.

    The bound has the shape

    .. math::

        \\underbrace{c_1 \\frac{N^2}{m u}}_{\\text{split}}
        \\underbrace{\\sum_{\\ell=0}^{L-1} (c_2 \\|\\hat{A}\\|_\\infty)^\\ell}_{\\text{depth}}
        + \\underbrace{c_3 c_2^{L} \\|\\hat{A} X\\|_{2 \\to \\infty} \\sqrt{\\log N}}_{\\text{data}},

    with :math:`c_1 = 2\\beta`, :math:`c_2 = 2\\omega` and
    :math:`c_3 = \\omega \\sqrt{2 / d}` built from the bias :math:`\\ell_1` norm
    :math:`\\beta` and the weight :math:`\\ell_\\infty` norm :math:`\\omega`.
    """
    if net == "sgc":
        beta = torch.linalg.vector_norm(model.conv1.lin.bias, ord=1)
        omega = max(
            torch.linalg.matrix_norm(model.conv1.lin.weight, ord=float("inf")), 1.0
        )
        a_hat_x_2toinf = max(
            torch.linalg.vector_norm(
                model.feature(data.x, data.edge_index, K=1), ord=2, dim=0
            )
        )
    elif net == "gcn":
        beta = max(
            torch.linalg.vector_norm(model.encoder.stack[l].bias, ord=1)
            for l in range(e_depth)
        )
        omega = max(
            torch.linalg.matrix_norm(model.encoder.stack[l].lin.weight, ord=float("inf"))
            for l in range(e_depth)
        )
        a_hat_x_2toinf = max(
            torch.linalg.vector_norm(a_hat_x(data.x, data.edge_index), ord=2, dim=0)
        )
    else:
        raise NotImplementedError(
            "the Rademacher bound of Esser et al. (2021) is derived only for SGC and GCN"
        )

    c1, c2 = 2 * beta, 2 * omega
    c3 = omega * np.sqrt(2 / num_features)
    inf_norm = a_hat_inf_norm(num_nodes, data.edge_index)

    split_term = c1 * (num_nodes ** 2) / (num_train * num_test)
    depth_term = torch.tensor([(c2 * inf_norm) ** l for l in range(e_depth)]).sum()
    data_term = c3 * (c2 ** e_depth) * a_hat_x_2toinf * np.sqrt(np.log(num_nodes))
    return float(split_term * depth_term + data_term)


def _safe_geom_sum(lr_x, n_steps):
    """:math:`\\sum_{t=0}^{T-1} (1 + \\eta X)^t` in closed form, evaluated in log space.

    The per-step factor is well above 1, so the naive power overflows at the depths used
    in Section 5.3.  Returns ``inf`` when the true value exceeds the float range, which
    is the honest answer for this bound at large depth.
    """
    if lr_x <= 0:
        return float(n_steps)
    log_max = n_steps * math.log1p(lr_x)
    if log_max > 700.0:  # exp(700) is about 1e304
        return float("inf")
    return (math.exp(log_max) - 1.0) / lr_x


def cong_stability_bound(model, data, net, e_depth, lr, epochs, num_train):
    """Uniform-stability bound of Cong et al. (2021), for SGC and GCN.

    Included in Section 5.3 as a contrast: it grows monotonically and exponentially with
    depth, whereas the empirical generalization error does not.
    """
    b_x = float(torch.max(torch.norm(data.x, p="fro", dim=1)))
    degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
    max_degree = float(degree.max())

    if net == "sgc":
        c_1 = (max_degree ** (e_depth / 2)) * b_x
        X = c_1 ** 2 + 1
        return (2 * lr / num_train) * (c_1 ** 2) * _safe_geom_sum(lr * X, epochs)

    if net == "gcn":
        b_w = max(
            float(torch.linalg.norm(model.encoder.stack[l].lin.weight, ord=2))
            for l in range(e_depth)
        )
        c_1 = max(1.0, (max_degree ** 0.5) * b_w)
        c_2 = (max_degree ** 0.5) * (1 + b_x)
        X = (c_1 ** e_depth) * c_2 * (
            (e_depth + 2) * (c_1 ** e_depth) * (c_1 ** e_depth) * c_2 + 2
        )
        return (
            (2 * lr / num_train) * (c_1 ** (2 * e_depth)) * (c_2 ** 2)
            * _safe_geom_sum(lr * X, epochs)
        )

    raise NotImplementedError("the Cong et al. (2021) bound is implemented for SGC and GCN")
