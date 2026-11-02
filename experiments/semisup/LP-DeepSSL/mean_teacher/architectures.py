# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.


import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils import weight_norm

from .utils import export, parameter_count


@export
def cifar_cnn_flex(pretrained=False, **kwargs):
    assert not pretrained
    model = FlexCNN(**kwargs)
    return model



class FlexCNN(nn.Module):
    """
    Flexible CNN with configurable encoder depth (n_blocks), width
    (base_channels), and classifier depth (c_depth).
    Each conv block: 3 x (Conv-BN-LeakyReLU) + MaxPool + Dropout.
    Channel schedule: base, 2*base, 4*base, ...

    Classifier: (c_depth - 1) hidden Linear+LeakyReLU layers of width feat_dim,
    followed by a final Linear(feat_dim, num_classes).  c_depth=1 recovers the
    original single-layer classifier; phi, the feature the bound is computed on,
    is the input to the final Linear.
    """

    def __init__(self, num_classes=10, isL2=False, double_output=False,
                 n_blocks=3, base_channels=128, c_depth=1):
        super(FlexCNN, self).__init__()
        self.isL2 = isL2
        self.double_output = double_output
        self.n_blocks = n_blocks
        self.c_depth = c_depth

        self.blocks = nn.ModuleList()
        in_ch = 3
        for i in range(n_blocks):
            out_ch = base_channels * (2 ** i)
            block = nn.Sequential(
                weight_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, stride=2),
                nn.Dropout(0.5),
            )
            self.blocks.append(block)
            in_ch = out_ch

        self.feat_dim = in_ch  # channels after last block
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.hidden_fcs = nn.ModuleList()
        for _ in range(max(c_depth - 1, 0)):
            self.hidden_fcs.append(nn.Sequential(
                weight_norm(nn.Linear(self.feat_dim, self.feat_dim)),
                nn.LeakyReLU(0.1),
            ))

        self.fc1 = weight_norm(nn.Linear(self.feat_dim, num_classes))
        if self.double_output:
            self.fc2 = weight_norm(nn.Linear(self.feat_dim, num_classes))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.isL2:
            x = F.normalize(x)
        for hidden in self.hidden_fcs:
            x = hidden(x)
        if self.double_output:
            return self.fc1(x), self.fc2(x), x
        else:
            return self.fc1(x), x


