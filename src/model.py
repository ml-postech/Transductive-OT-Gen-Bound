"""
GNN model architectures for node classification.

This module provides GNN encoder-classifier architectures including:
- GCN (Graph Convolutional Network)
- GCN2 (GCN with initial residual connections)
- GAT (Graph Attention Network)
- GraphSAGE
- SGC (Simple Graph Convolution)
- Residual variants (ResGCN, ResGAT, ResSAGE)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, SGConv, SAGEConv, GraphNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# Normalization layer mapping
NORMALIZATION = {
    "None": nn.Identity,
    "LayerNorm": nn.LayerNorm,
    "BatchNorm": nn.BatchNorm1d,
    "GraphNorm": GraphNorm,
}

# Activation function mapping
ACTIVATION = {
    "None": nn.Identity,
    "ReLU": nn.ReLU,
    "GeLU": nn.GELU,
    "Leaky": nn.LeakyReLU,
    "Tanh": nn.Tanh,
}


class gnn_encoder(nn.Module):
    """
    GNN encoder that stacks multiple graph convolution layers.

    Supports multiple GNN architectures: GCN, GCN2, GAT, GraphSAGE,
    and their residual variants.

    Args:
        net: Network type ('gcn', 'resgcn', 'gcn2', 'gat', 'resgat', 'sage', 'ressage')
        e_depth: Number of encoder layers
        num_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        layernorm: Normalization type ('None', 'LayerNorm', 'BatchNorm', 'GraphNorm')
        act: Activation function ('ReLU', 'GeLU', 'Leaky', 'Tanh', 'None')
        heads: Number of attention heads (for GAT only)
        no_classifier: If set, output dimension equals this value (for direct classification)
    """

    def __init__(self, net, e_depth, num_features, hidden_dim, layernorm, act, heads=None, no_classifier=None):
        super(gnn_encoder, self).__init__()

        self.e_depth = e_depth
        self.norm = nn.ModuleList()
        self.stack = nn.ModuleList()
        self.inlin = None
        self.no_classifier = no_classifier

        # GCN2 requires initial linear transformation
        if net == "gcn2":
            self.inlin = nn.Linear(num_features, hidden_dim)

        # Build encoder layers
        for depth_idx in range(e_depth):
            in_dim = num_features if depth_idx == 0 else hidden_dim

            if self.no_classifier is None:
                out_dim = hidden_dim
            else:
                out_dim = hidden_dim if depth_idx != e_depth - 1 else (1 if self.no_classifier == 2 else self.no_classifier)

            # Add normalization (skip first layer)
            if depth_idx != 0:
                self.norm.append(NORMALIZATION[layernorm](in_dim))

            # Add convolution layer based on network type
            if net.endswith("gcn"):
                self.stack.append(GCNConv(in_dim, out_dim))
            elif net == "gcn2":
                self.stack.append(GCN2Conv(hidden_dim, alpha=0.1, theta=0.5, layer=depth_idx + 1,
                                          shared_weights=True, normalize=True))
            elif net.endswith("gat"):
                self.stack.append(GATConv(in_dim, out_dim // heads, heads))
            elif net.endswith("sage"):
                self.stack.append(SAGEConv(in_dim, out_dim))
            else:
                raise NotImplementedError(f"Network type {net} not supported")

        self.act = ACTIVATION[act]() if act != "Leaky" else ACTIVATION[act](0.8)
        self.res = net.startswith("res")

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the GNN encoder.

        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            Node embeddings [num_nodes, hidden_dim] or [num_nodes, num_classes]
        """
        if self.inlin is not None:
            # GCN2 forward pass with initial residual
            x0 = self.inlin(x)
            x = x0
            for depth_idx in range(self.e_depth):
                x = self.norm[depth_idx - 1](x) if depth_idx != 0 else x
                x = self.stack[depth_idx](x, x0, edge_index)
                x = self.act(x)
        else:
            # Standard GNN forward pass
            for depth_idx in range(self.e_depth):
                Fx = self.norm[depth_idx - 1](x) if depth_idx != 0 else x
                Fx = self.stack[depth_idx](Fx, edge_index)
                if (depth_idx == self.e_depth - 1) and (self.no_classifier is not None):
                    x = Fx
                else:
                    x = self.act(Fx) if not self.res or depth_idx == 0 else x + self.act(Fx)
        return x


class mlp_classifier(nn.Module):
    """
    MLP classifier head for node classification.

    Args:
        c_depth: Number of classifier layers
        num_classes: Number of output classes
        hidden_dim: Input/hidden dimension
        act: Activation function
    """

    def __init__(self, c_depth, num_classes, hidden_dim, act):
        super(mlp_classifier, self).__init__()

        self.c_depth = c_depth
        self.stack = nn.ModuleList()

        for depth_idx in range(self.c_depth):
            in_dim = hidden_dim
            out_dim = hidden_dim if depth_idx != self.c_depth - 1 else (1 if num_classes == 2 else num_classes)
            self.stack.append(nn.Linear(in_dim, out_dim))

        self.act = ACTIVATION[act]() if act != "Leaky" else ACTIVATION[act](0.8)

    def forward(self, x):
        """Forward pass through MLP classifier."""
        for depth_idx in range(self.c_depth):
            x = self.stack[depth_idx](x)
            x = x if depth_idx == self.c_depth - 1 else self.act(x)
        return x.squeeze(1)


class gnn(nn.Module):
    """
    Complete GNN model with encoder and classifier.

    Combines a GNN encoder with an MLP classifier for node classification.

    Args:
        args: Arguments containing model configuration
        dataset: PyG dataset object
        device: Torch device
    """

    def __init__(self, args, dataset, device=torch.device("cpu")):
        super(gnn, self).__init__()

        self.args = args
        self.e_depth = args.e_depth
        self.c_depth = args.c_depth
        self.num_classes = dataset.num_classes

        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.hidden_dim = args.hidden_dim

        # Build encoder
        if args.net.endswith("gcn") or args.net.endswith("sage") or (args.net == "gcn2"):
            self.encoder = gnn_encoder(args.net, args.e_depth, self.num_features,
                                       self.hidden_dim, args.layernorm, args.act)
        elif args.net.endswith("gat"):
            self.encoder = gnn_encoder(args.net, args.e_depth, self.num_features,
                                       self.hidden_dim, args.layernorm, args.act, args.heads)

        # Build classifier
        self.classifier = mlp_classifier(args.c_depth, self.num_classes, self.hidden_dim, args.act)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass: encoder -> classifier.

        Returns:
            Logits [num_nodes, num_classes] or [num_nodes] for binary
        """
        x = self.encoder(x, edge_index, edge_weight)
        x = self.classifier(x)
        return x

    def feature(self, x, edge_index, edge_weight=None):
        """
        Extract node embeddings from encoder (before classifier).

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        x = self.encoder(x, edge_index, edge_weight)
        return x


class sgc(nn.Module):
    """
    Simple Graph Convolution (SGC) model.

    SGC simplifies GCN by removing nonlinearities between layers,
    allowing precomputation of the propagated features.

    Args:
        args: Arguments containing model configuration
        dataset: PyG dataset object
        device: Torch device
    """

    def __init__(self, args, dataset, device=torch.device("cpu")):
        super(sgc, self).__init__()

        self.args = args
        self.e_depth = args.e_depth
        self.num_classes = dataset.num_classes

        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device

        # SGC convolution with K hops
        self.conv1 = SGConv(
            in_channels=self.num_features,
            out_channels=1 if self.num_classes == 2 else self.num_classes,
            K=self.e_depth,
            cached=True,
        )

        # MLP classifier
        self.classifier = mlp_classifier(args.c_depth, self.num_classes, self.num_features, args.act)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with feature propagation then classification.

        Returns:
            Logits [num_nodes, num_classes] or [num_nodes] for binary
        """
        # Propagate features K times
        edge_index_tmp, edge_weight_tmp = gcn_norm(
            edge_index, edge_weight, x.size(self.conv1.node_dim), False,
            self.conv1.add_self_loops, self.conv1.flow, dtype=x.dtype
        )
        for k in range(self.conv1.K):
            x = self.conv1.propagate(edge_index_tmp, x=x, edge_weight=edge_weight_tmp)

        # Classify
        x = self.classifier(x)
        return x.squeeze(1)

    def feature(self, x, edge_index, edge_weight=None, K=None):
        """
        Extract propagated features (before classifier).

        Args:
            K: Number of propagation steps (default: self.conv1.K)

        Returns:
            Propagated features [num_nodes, num_features]
        """
        edge_index_tmp, edge_weight_tmp = gcn_norm(
            edge_index, edge_weight, x.size(self.conv1.node_dim), False,
            self.conv1.add_self_loops, self.conv1.flow, dtype=x.dtype
        )
        if K is None:
            K = self.conv1.K
        for k in range(K):
            x = self.conv1.propagate(edge_index_tmp, x=x, edge_weight=edge_weight_tmp)
        return x
