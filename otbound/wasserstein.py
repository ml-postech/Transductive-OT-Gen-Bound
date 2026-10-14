"""Empirical 1-Wasserstein distances between feature point clouds.

Every :math:`W_1` in the paper is between two *empirical* measures
:math:`\\phi_\\# \\mu_I = \\frac{1}{|I|}\\sum_{i \\in I} \\delta_{\\phi(x_i)}`, i.e. uniform
weights on a finite set of encoded points.  Two regimes are needed:

* **Evaluation** (Theorem 1, Figure 1) -- no gradient required, so the exact
  linear-programming solution is used: :func:`w1_exact`.
* **Training** (Section 5.2) -- the regularizer must be differentiable, so the
  entropy-regularized Sinkhorn approximation is used instead:
  :func:`sinkhorn_distance`.

:func:`w1_sliced` is a cheap fallback for point clouds too large for the exact solver.
"""

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

try:
    import ot

    OT_AVAILABLE = True
except ImportError:  # pragma: no cover - POT is a hard requirement in practice
    OT_AVAILABLE = False

__all__ = ["w1_exact", "w1_sliced", "sinkhorn_distance", "OT_AVAILABLE"]


def _as_numpy(feats):
    if isinstance(feats, torch.Tensor):
        return feats.detach().cpu().numpy()
    return np.asarray(feats)


def w1_exact(feats_0, feats_1):
    """Exact empirical :math:`W_1` with a Euclidean ground cost.

    Solves the discrete optimal transport problem between the uniform measures on
    ``feats_0`` and ``feats_1``.  Not differentiable -- use :func:`sinkhorn_distance`
    inside a training loop.

    Args:
        feats_0: ``[k0, d]`` array or tensor.
        feats_1: ``[k1, d]`` array or tensor.

    Returns:
        float: the :math:`W_1` distance.
    """
    if not OT_AVAILABLE:
        raise ImportError("POT is required for exact W1. Install with: pip install POT")

    f0, f1 = _as_numpy(feats_0), _as_numpy(feats_1)
    k0, k1 = len(f0), len(f1)
    M = pairwise_distances(f0, f1)
    a = np.ones(k0) / k0
    b = np.ones(k1) / k1
    return float(ot.emd2(a, b, M, numItermax=1000000))


def w1_sliced(feats_0, feats_1, n_projections=100):
    """Sliced :math:`W_1`: the average of one-dimensional :math:`W_1` over random
    projections.  Much cheaper than :func:`w1_exact` and fully GPU-resident, at the
    cost of being a lower bound rather than the exact value.
    """
    f0 = feats_0 if isinstance(feats_0, torch.Tensor) else torch.as_tensor(feats_0)
    f1 = feats_1 if isinstance(feats_1, torch.Tensor) else torch.as_tensor(feats_1)
    k0, k1, d = f0.shape[0], f1.shape[0], f0.shape[1]

    directions = torch.randn(n_projections, d, device=f0.device)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    proj_0 = torch.sort(f0 @ directions.T, dim=0)[0]  # [k0, n_projections]
    proj_1 = torch.sort(f1 @ directions.T, dim=0)[0]  # [k1, n_projections]

    if k0 != k1:
        # Compare the two empirical quantile functions on a common grid.
        common_q = torch.linspace(0, 1, min(k0, k1), device=f0.device)
        proj_0 = torch.quantile(proj_0, common_q, dim=0)
        proj_1 = torch.quantile(proj_1, common_q, dim=0)

    return float(torch.abs(proj_0 - proj_1).mean())


def sinkhorn_distance(x, y, epsilon=0.05, max_iter=50, ground_p=2):
    """Differentiable :math:`W_1` approximation via Sinkhorn iterations (Cuturi, 2013).

    By the envelope theorem the gradient of the optimal transport objective with
    respect to the cost matrix :math:`C` is the optimal plan :math:`\\pi`, which lets us
    split the computation in two:

    1. run the Sinkhorn iterations under ``no_grad`` to obtain :math:`\\pi` (this is
       where the numerical work and any instability lives), then
    2. return :math:`\\langle \\pi, C \\rangle` with the graph attached to :math:`C`.

    The result carries the correct gradient to ``x`` and ``y`` without differentiating
    through the iterations themselves.

    Args:
        x: ``[n, d]`` tensor.
        y: ``[m, d]`` tensor.
        epsilon: entropic regularization; smaller is closer to exact OT but less stable.
        max_iter: number of Sinkhorn iterations.
        ground_p: exponent of the **ground metric** :math:`\\|x - y\\|_p`, *not* the order
            of the Wasserstein distance.  The order is 1 either way, because the cost
            enters :math:`\\langle \\pi, C \\rangle` linearly.  Leave at the Euclidean
            default: Theorem 1 measures feature displacement with
            :math:`\\|\\phi(x_i) - \\phi(x_j)\\|_2` -- the same norm that appears in the
            denominator of :math:`M_c` -- and :func:`w1_exact` uses it too.  Switching to
            ``ground_p=1`` makes this estimate a *different* quantity, inflated by a
            factor that depends on the feature dimension and distribution and so cannot
            be absorbed into the regularization weight :math:`\\lambda`.

    Returns:
        torch.Tensor: a scalar, differentiable with respect to ``x`` and ``y``.
    """
    n, m = x.shape[0], y.shape[0]
    device = x.device

    C = torch.cdist(x, y, p=float(ground_p))  # [n, m], differentiable in x and y

    with torch.no_grad():
        Cd = C.detach()
        log_a = torch.full((n,), -float(np.log(n)), device=device)
        log_b = torch.full((m,), -float(np.log(m)), device=device)
        log_u = torch.zeros(n, device=device)
        log_v = torch.zeros(m, device=device)
        for _ in range(max_iter):
            log_u = log_a - torch.logsumexp(-Cd / epsilon + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(-Cd / epsilon + log_u.unsqueeze(1), dim=0)
        log_pi = log_u.unsqueeze(1) + log_v.unsqueeze(0) - Cd / epsilon
        pi = torch.exp(log_pi.clamp(-60, 60))

    return (pi * C).sum()
