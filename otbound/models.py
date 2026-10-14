"""Encoder / classifier architectures for the node classification experiments.

Every model here splits into an **encoder** :math:`\\phi` and a **classifier** :math:`f`,
mirroring the compositional hypothesis class :math:`\\mathcal{F} \\circ \\Phi` of Section 4.1.
That split is not cosmetic: the bound is evaluated on :math:`\\phi`'s output, so each
model exposes :meth:`feature` alongside :meth:`forward`.

``net`` names combine three choices:

``gcn`` / ``gat`` / ``sage`` / ``sgc`` / ``gcn2``
    the message passing operator.
``res`` prefix (``resgcn``, ``resgat``, ``ressage``)
    add a residual connection around each layer.  Appendix E.1 uses these for the
    regularization experiments, where an 8-layer encoder without skips underfits and
    would confound the regularizer's effect with plain optimization failure.
``mlp`` suffix
    drop ``edge_index`` entirely; a graph-free control.

``GCNII`` (``gcn2``) carries its own initial-residual connection, so it takes no ``res``
prefix.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, GraphNorm, SAGEConv, SGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

__all__ = ["NORMALIZATION", "ACTIVATION", "GNNEncoder", "MLPClassifier", "GNN", "SGC", "make_activation",
           "build_model"]

NORMALIZATION = {
    "None": nn.Identity,
    "LayerNorm": nn.LayerNorm,
    "BatchNorm": nn.BatchNorm1d,
    "GraphNorm": GraphNorm,
}

ACTIVATION = {
    "None": nn.Identity,
    "ReLU": nn.ReLU,
    "GeLU": nn.GELU,
    "Leaky": nn.LeakyReLU,
    "Tanh": nn.Tanh,
}


def make_activation(act):
    return ACTIVATION[act](0.8) if act == "Leaky" else ACTIVATION[act]()


class GNNEncoder(nn.Module):
    """Stack of ``e_depth`` message passing layers producing node features :math:`\\phi(x)`.

    Args:
        net: architecture name, see the module docstring.
        e_depth: number of layers, :math:`L`.
        num_features: input dimension.
        hidden_dim: width of every hidden layer.
        layernorm: key into :data:`NORMALIZATION`; applied before each layer but the first.
        act: key into :data:`ACTIVATION`.
        heads: attention heads, for ``gat`` only.
        out_dim: when set, the last layer emits ``out_dim`` channels instead of
            ``hidden_dim``, turning the encoder into a complete classifier.  Needed for
            the Rademacher baseline, which is derived for a head-free GNN.
    """

    def __init__(self, net, e_depth, num_features, hidden_dim, layernorm, act,
                 heads=None, out_dim=None):
        super().__init__()
        self.e_depth = e_depth
        self.out_dim = out_dim
        self.norm = nn.ModuleList()
        self.stack = nn.ModuleList()
        self.inlin = nn.Linear(num_features, hidden_dim) if net == "gcn2" else None

        for depth_idx in range(e_depth):
            in_dim = num_features if depth_idx == 0 else hidden_dim
            if out_dim is None or depth_idx != e_depth - 1:
                layer_out = hidden_dim
            else:
                layer_out = 1 if out_dim == 2 else out_dim

            if depth_idx != 0:
                self.norm.append(NORMALIZATION[layernorm](in_dim))

            if net.endswith("gcn"):
                self.stack.append(GCNConv(in_dim, layer_out))
            elif net == "gcn2":
                self.stack.append(
                    GCN2Conv(hidden_dim, alpha=0.1, theta=0.5, layer=depth_idx + 1,
                             shared_weights=True, normalize=True)
                )
            elif net.endswith("gat"):
                self.stack.append(GATConv(in_dim, layer_out // heads, heads))
            elif net.endswith("sage"):
                self.stack.append(SAGEConv(in_dim, layer_out))
            elif net.endswith("mlp"):
                self.stack.append(nn.Linear(in_dim, layer_out))
            else:
                raise NotImplementedError("unknown architecture: {}".format(net))

        self.act = make_activation(act)
        self.res = net.startswith("res")
        self.is_mlp = net.endswith("mlp")

    def forward(self, x, edge_index, edge_weight=None):
        if self.inlin is not None:
            # GCNII: every layer mixes in the initial representation x0, which is
            # produced by a plain linear map with no activation.
            x0 = self.inlin(x)
            x = x0
            for depth_idx in range(self.e_depth):
                x = self.norm[depth_idx - 1](x) if depth_idx != 0 else x
                x = self.stack[depth_idx](x, x0, edge_index)
                x = self.act(x)
            return x

        for depth_idx in range(self.e_depth):
            fx = self.norm[depth_idx - 1](x) if depth_idx != 0 else x
            fx = self.stack[depth_idx](fx) if self.is_mlp else self.stack[depth_idx](fx, edge_index)
            if depth_idx == self.e_depth - 1 and self.out_dim is not None:
                x = fx  # final layer emits logits, so no activation or residual
            else:
                x = self.act(fx) if not self.res or depth_idx == 0 else x + self.act(fx)
        return x


class MLPClassifier(nn.Module):
    """The score-based classifier :math:`f = [f_1, \\dots, f_K]` on top of :math:`\\phi`.

    ``c_depth`` linear layers with ``act`` in between; the last one is left linear so
    its outputs are logits.  Binary tasks emit a single logit.
    """

    def __init__(self, c_depth, num_classes, hidden_dim, act):
        super().__init__()
        self.c_depth = c_depth
        self.stack = nn.ModuleList()
        for depth_idx in range(c_depth):
            out_dim = hidden_dim
            if depth_idx == c_depth - 1:
                out_dim = 1 if num_classes == 2 else num_classes
            self.stack.append(nn.Linear(hidden_dim, out_dim))
        self.act = make_activation(act)

    def forward(self, x):
        for depth_idx in range(self.c_depth):
            x = self.stack[depth_idx](x)
            x = x if depth_idx == self.c_depth - 1 else self.act(x)
        return x.squeeze(1)


class GNN(nn.Module):
    """Encoder plus classifier, the hypothesis :math:`f \\circ \\phi`.

    Set ``head_free=True`` to fold the classifier into the encoder's last layer, which
    is what the Rademacher baseline requires.
    """

    def __init__(self, args, num_features, num_classes, head_free=False):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.head_free = head_free

        out_dim = num_classes if head_free else None
        heads = getattr(args, "heads", 1)
        self.encoder = GNNEncoder(
            args.net, args.e_depth, num_features, args.hidden_dim,
            args.layernorm, args.act, heads=heads, out_dim=out_dim,
        )
        self.classifier = (
            None if head_free
            else MLPClassifier(args.c_depth, num_classes, args.hidden_dim, args.act)
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.encoder(x, edge_index, edge_weight)
        return x if self.head_free else self.classifier(x)

    def feature(self, x, edge_index, edge_weight=None):
        """The encoder output :math:`\\phi(x)` -- the representation the bound sees."""
        return self.encoder(x, edge_index, edge_weight)


class SGC(nn.Module):
    """Simplified GCN: ``e_depth`` propagation steps with no non-linearity in between.

    Because propagation carries no parameters, :meth:`feature` is just
    :math:`\\hat{A}^\\ell X` and can be evaluated at any depth ``K``, which the
    Proposition 2 check in Section 5.3 uses directly.
    """

    def __init__(self, args, num_features, num_classes, head_free=False):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.head_free = head_free
        self.e_depth = args.e_depth

        self.conv1 = SGConv(
            in_channels=num_features,
            out_channels=1 if num_classes == 2 else num_classes,
            K=args.e_depth,
            cached=True,
        )
        # The classifier reads raw propagated features, so its width is num_features.
        self.classifier = (
            None if head_free
            else MLPClassifier(args.c_depth, num_classes, num_features, args.act)
        )

    def forward(self, x, edge_index, edge_weight=None):
        if self.head_free:
            return self.conv1(x, edge_index).squeeze(1)
        return self.classifier(self.feature(x, edge_index, edge_weight))

    def feature(self, x, edge_index, edge_weight=None, K=None):
        """:math:`\\hat{A}^K X`, defaulting to ``K = e_depth``."""
        edge_index_tmp, edge_weight_tmp = gcn_norm(
            edge_index, edge_weight, x.size(self.conv1.node_dim), False,
            self.conv1.add_self_loops, self.conv1.flow, dtype=x.dtype,
        )
        K = self.conv1.K if K is None else K
        for _ in range(K):
            x = self.conv1.propagate(edge_index_tmp, x=x, edge_weight=edge_weight_tmp)
        return x


def build_model(args, num_features, num_classes, head_free=False):
    """Instantiate :class:`SGC` or :class:`GNN` according to ``args.net``."""
    cls = SGC if args.net == "sgc" else GNN
    return cls(args, num_features, num_classes, head_free=head_free)
