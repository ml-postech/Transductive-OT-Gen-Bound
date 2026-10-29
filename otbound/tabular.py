"""Encoders and classifiers for the non-graph experiments (Section 6, Appendix G).

The bound is stated for a general encoder :math:`\\phi : \\mathcal{X} \\to \\mathcal{Z}`;
nothing about it is graph-specific.  These are the tabular and image counterparts of
:mod:`otbound.models`, with the same encoder / classifier split and the same
:meth:`feature` accessor, so :mod:`otbound.bound` and :mod:`otbound.regularizer` apply
unchanged.

Both architectures take a ``residual`` flag.  As in the graph case, the regularization
experiments enable it together with normalization (Appendix G.2), so that an 8-layer
encoder trains stably and the regularizer's effect is not confounded with underfitting.
"""

import torch
import torch.nn as nn

__all__ = ["TabularMLP", "TabularCNN", "build_tabular_model", "batched_forward",
           "ACTIVATION", "NORM_MLP", "NORM_CNN", "TABULAR_NETS"]

ACTIVATION = {
    "None": nn.Identity,
    "ReLU": nn.ReLU,
    "GeLU": nn.GELU,
    "Leaky": nn.LeakyReLU,
    "Tanh": nn.Tanh,
}

NORM_MLP = {
    "None": lambda c: nn.Identity(),
    "LayerNorm": lambda c: nn.LayerNorm(c),
    "BatchNorm": lambda c: nn.BatchNorm1d(c),
}

NORM_CNN = {
    "None": lambda c: nn.Identity(),
    "BatchNorm": lambda c: nn.BatchNorm2d(c),
    "GroupNorm": lambda c: nn.GroupNorm(num_groups=min(8, c), num_channels=c),
}

TABULAR_NETS = ("mlp", "resmlp", "cnn", "rescnn")


def _make_act(act):
    return ACTIVATION[act](0.8) if act == "Leaky" else ACTIVATION[act]()


@torch.no_grad()
def batched_forward(fn, X, batch_size):
    """Apply ``fn`` to ``X`` in chunks and concatenate; ``batch_size == 0`` means one pass.

    Needed because the full-dataset forward pass that the bound requires does not fit in
    memory for CovType or the image benchmarks.
    """
    if batch_size == 0 or len(X) <= batch_size:
        return fn(X)
    return torch.cat([fn(X[i:i + batch_size]) for i in range(0, len(X), batch_size)], dim=0)


class _ClassifierHead(nn.Module):
    """``c_depth`` linear layers ending in logits; binary tasks emit a single logit."""

    def __init__(self, hidden_dim, num_classes, c_depth, act):
        super().__init__()
        self.c_depth = c_depth
        self.layers = nn.ModuleList()
        for depth_idx in range(c_depth):
            out_dim = hidden_dim
            if depth_idx == c_depth - 1:
                out_dim = 1 if num_classes == 2 else num_classes
            self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.act = _make_act(act)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != self.c_depth - 1:
                x = self.act(x)
        return x.squeeze(-1) if x.dim() > 1 and x.shape[-1] == 1 else x


class TabularMLP(nn.Module):
    """Fully connected encoder over flat feature vectors.

    With ``residual=True`` each block after the first becomes pre-norm residual,
    :math:`h \\leftarrow h + \\sigma(W_i \\,\\mathrm{norm}_i(h))`.  The first block cannot
    carry a skip because the input and hidden widths differ.
    """

    def __init__(self, num_features, num_classes, hidden_dim=64, e_depth=2, c_depth=1,
                 act="ReLU", residual=False, layernorm="None"):
        super().__init__()
        self.e_depth = e_depth
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.residual = residual

        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        for depth_idx in range(e_depth):
            in_dim = num_features if depth_idx == 0 else hidden_dim
            self.encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if depth_idx != 0:
                self.encoder_norms.append(NORM_MLP[layernorm](in_dim))

        self.classifier = _ClassifierHead(hidden_dim, num_classes, c_depth, act)
        self.act = _make_act(act)

    def encode(self, x):
        for depth_idx in range(self.e_depth):
            if depth_idx == 0:
                x = self.act(self.encoder_layers[0](x))
            elif self.residual:
                fx = self.encoder_layers[depth_idx](self.encoder_norms[depth_idx - 1](x))
                x = x + self.act(fx)
            else:
                x = self.act(self.encoder_layers[depth_idx](x))
        return x

    def forward(self, x):
        return self.classifier(self.encode(x))

    def feature(self, x):
        return self.encode(x)


class TabularCNN(nn.Module):
    """Convolutional encoder for image-shaped inputs.

    Inputs arrive flat as ``[N, C*H*W]`` and are reshaped internally.  Every block is a
    3x3 convolution with stride 1 and padding 1, so the spatial size is preserved and a
    final global average pool yields a ``hidden_dim`` feature vector -- which is the
    :math:`\\phi(x)` the bound is computed on.
    """

    def __init__(self, image_shape, num_classes, hidden_dim=64, e_depth=2, c_depth=1,
                 act="ReLU", residual=False, layernorm="None"):
        super().__init__()
        self.image_shape = image_shape  # (C, H, W)
        self.e_depth = e_depth
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.residual = residual

        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        for depth_idx in range(e_depth):
            cin = image_shape[0] if depth_idx == 0 else hidden_dim
            self.encoder_layers.append(
                nn.Conv2d(cin, hidden_dim, kernel_size=3, stride=1, padding=1)
            )
            if depth_idx != 0:
                self.encoder_norms.append(NORM_CNN[layernorm](cin))

        self.classifier = _ClassifierHead(hidden_dim, num_classes, c_depth, act)
        self.act = _make_act(act)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def _reshape(self, x):
        return x.view(-1, *self.image_shape) if x.dim() == 2 else x

    def encode(self, x):
        x = self._reshape(x)
        for depth_idx in range(self.e_depth):
            if depth_idx == 0:
                x = self.act(self.encoder_layers[0](x))
            elif self.residual:
                fx = self.encoder_layers[depth_idx](self.encoder_norms[depth_idx - 1](x))
                x = x + self.act(fx)
            else:
                x = self.act(self.encoder_layers[depth_idx](x))
        return self.gap(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.encode(x))

    def feature(self, x):
        return self.encode(x)


def build_tabular_model(args, num_features, num_classes, image_shape=None):
    """Instantiate one of :data:`TABULAR_NETS` from parsed arguments.

    Convolutional variants require ``image_shape``; the dataset loader supplies it for
    image-shaped benchmarks and leaves it ``None`` otherwise.
    """
    residual = args.net.startswith("res")
    kind = args.net[3:] if residual else args.net
    common = dict(
        num_classes=num_classes, hidden_dim=args.hidden_dim, e_depth=args.e_depth,
        c_depth=args.c_depth, act=args.act, residual=residual,
        layernorm=getattr(args, "layernorm", "None"),
    )

    if kind == "mlp":
        return TabularMLP(num_features=num_features, **common)
    if kind == "cnn":
        if image_shape is None:
            raise ValueError("a convolutional encoder needs an image-shaped dataset")
        return TabularCNN(image_shape=image_shape, **common)
    raise NotImplementedError(
        "unknown architecture {!r}; expected one of {}".format(args.net, ", ".join(TABULAR_NETS))
    )
