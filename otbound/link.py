"""Link prediction model and edge-level sample construction (Appendix E.1).

Link prediction is the second transductive task in the paper, and it has the same
feature-dependency problem as node classification: the encoder sees the whole observed
graph, so edge representations are not independent.

Theorem 1 applies verbatim once the *sample* is taken to be an edge rather than a node:

* :math:`\\phi` maps an edge to the concatenation :math:`[z_{\\mathrm{src}}, z_{\\mathrm{dst}}]`
  of its endpoint embeddings,
* :math:`f` is the link predictor, a binary classifier, and
* the label is 1 for an observed edge and 0 for a sampled non-edge.

:func:`build_edge_samples` performs that translation, after which
:func:`otbound.bound.ot_bound` is called unchanged.
"""

import torch
import torch.nn as nn

from .margin import margins_all
from .models import GNNEncoder, make_activation

__all__ = ["LinkPredictor", "LinkPredictionModel", "link_labels", "build_edge_samples"]


class LinkPredictor(nn.Module):
    """MLP over concatenated endpoint embeddings, emitting one logit per edge."""

    def __init__(self, hidden_dim, c_depth=1, act="ReLU"):
        super().__init__()
        self.stack = nn.ModuleList()
        for depth_idx in range(c_depth):
            in_dim = hidden_dim * 2 if depth_idx == 0 else hidden_dim
            out_dim = 1 if depth_idx == c_depth - 1 else hidden_dim
            self.stack.append(nn.Linear(in_dim, out_dim))
        self.act = make_activation(act)

    def forward(self, z, edge_index):
        x = self.edge_features(z, edge_index)
        for idx, layer in enumerate(self.stack):
            x = layer(x)
            if idx != len(self.stack) - 1:
                x = self.act(x)
        return x.squeeze(-1)

    @staticmethod
    def edge_features(z, edge_index):
        """:math:`\\phi` at the edge level: ``[z_src, z_dst]``, shape ``[E, 2 * hidden]``."""
        return torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)


class LinkPredictionModel(nn.Module):
    """GNN encoder plus link predictor."""

    def __init__(self, args, num_features):
        super().__init__()
        self.encoder = GNNEncoder(
            args.net, args.e_depth, num_features, args.hidden_dim,
            args.layernorm, args.act, heads=getattr(args, "heads", 1),
        )
        self.predictor = LinkPredictor(args.hidden_dim, args.c_depth, args.act)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        return self.predictor(z, edge_index)

    def forward(self, x, edge_index, pred_edge_index):
        return self.decode(self.encode(x, edge_index), pred_edge_index)


def link_labels(num_pos, num_neg, device):
    """``[1] * num_pos + [0] * num_neg``."""
    labels = torch.zeros(num_pos + num_neg, device=device)
    labels[:num_pos] = 1
    return labels


def build_edge_samples(model, z, train_edges, train_labels, test_edges, test_labels):
    """Recast edges as the samples Theorem 1 operates on.

    Returns ``(features, margin_all, y, train_mask, test_mask)``, ready to hand to
    :func:`otbound.bound.ot_bound` exactly as node-level quantities would be.

    The predictor emits a single logit ``s``; the two-class score vector is taken as
    ``[-s, s]`` so that :func:`otbound.margin.margins_all` yields the usual
    :math:`\\rho_f(\\phi(e), c)` for ``c`` in ``{0, 1}``.
    """
    train_features = LinkPredictor.edge_features(z, train_edges)
    test_features = LinkPredictor.edge_features(z, test_edges)
    train_scores = model.decode(z, train_edges)
    test_scores = model.decode(z, test_edges)

    features = torch.cat([train_features, test_features], dim=0)
    scores = torch.cat([train_scores, test_scores], dim=0)
    y = torch.cat([train_labels, test_labels], dim=0).long()

    margin_all = margins_all(torch.stack([-scores, scores], dim=-1))

    n_train, n_total = train_features.size(0), features.size(0)
    train_mask = torch.zeros(n_total, dtype=torch.bool, device=features.device)
    train_mask[:n_train] = True
    return features, margin_all, y, train_mask, ~train_mask
