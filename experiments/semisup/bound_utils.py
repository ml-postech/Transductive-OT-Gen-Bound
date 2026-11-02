"""Helpers for correlation.py: data, checkpoints, gap and bounds.

gap = |labeled error - unlabeled error| on the transductive pool; OT-oracle and OT bound
are the two estimators of the Theorem 1 term (Appendix C), computed by otbound.bound.
"""
import os
import sys
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..'))  # repository root, for otbound
from otbound.bound import ot_bound
from otbound.margin import margins_all

NUM_LABELED = {'cifar10': 1200, 'svhn': 2000}
# Input normalization of each code base (LP-DeepSSL: mean_teacher/datasets.py, FixMatch: dataset/cifar.py)
NORM = {'lp': {'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
               'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))},
        'fixmatch': {'cifar10': ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                     'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}}


def label_file(dataset):
    n = NUM_LABELED[dataset]
    return os.path.join(HERE, 'LP-DeepSSL', 'data-local', 'labels', dataset, f'{n}_balanced_labels', '10.txt')


def load_cnn(checkpoint, n_blocks, base_channels, c_depth, use_ema, num_classes=10):
    """LP-DeepSSL / Mean Teacher CNN; use_ema loads the Mean Teacher EMA weights (fc2 head skipped)."""
    sys.path.insert(0, os.path.join(HERE, 'LP-DeepSSL'))
    from mean_teacher import architectures
    model = architectures.cifar_cnn_flex(num_classes=num_classes, isL2=True, double_output=False,
                                         n_blocks=n_blocks, base_channels=base_channels, c_depth=c_depth)
    model = torch.nn.DataParallel(model).cuda()
    ckpt = torch.load(checkpoint, map_location='cuda')
    model.load_state_dict(ckpt['ema_state_dict' if use_ema else 'state_dict'], strict=not use_ema)
    return model.eval()


def load_wrn(checkpoint, depth, width, c_depth, use_ema=True, num_classes=10):
    """FixMatch / SimMatch WideResNet (EMA weights by default). Returns forward(x) -> (logits, phi)."""
    sys.path.insert(0, os.path.join(HERE, 'FixMatch'))
    from models.wideresnet import build_wideresnet
    model = build_wideresnet(depth=depth, widen_factor=width, dropout=0,
                             num_classes=num_classes, c_depth=c_depth)
    ckpt = torch.load(checkpoint, map_location='cpu')
    sd = ckpt['ema_state_dict'] if use_ema and ckpt.get('ema_state_dict') is not None else ckpt['state_dict']
    model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in sd.items()})
    model = model.cuda().eval()
    return lambda x: model(x, return_features=True)


def pac_bound(checkpoint, num_labeled, use_ema):
    """KL(Q || P) / m with P = N(0, I), Q = N(theta, I): ||theta||^2 / (2m) over trainable weights."""
    ckpt = torch.load(checkpoint, map_location='cpu')
    state = ckpt['ema_state_dict'] if use_ema and ckpt.get('ema_state_dict') is not None else ckpt['state_dict']
    bn_buffers = ('running_mean', 'running_var', 'num_batches_tracked')
    theta_sq = sum((v.float() ** 2).sum().item() for k, v in state.items()
                   if v.dtype.is_floating_point and not any(b in k for b in bn_buffers))
    return theta_sq / (2 * num_labeled)


def load_label_file(label_file_path):
    """Parse LP-DeepSSL label file → set of labeled filenames."""
    labeled_filenames = set()
    with open(label_file_path) as f:
        for line in f.read().splitlines():
            filename, _ = line.split(" ")
            labeled_filenames.add(filename)
    return labeled_filenames


def build_pool(data_dir, train_subdir, mean, std, batch_size, workers, label_file):
    """Un-augmented loader over the whole transductive pool + labeled mask."""
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, train_subdir), transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    labeled_names = load_label_file(label_file)
    mask = np.array([os.path.basename(p) in labeled_names for p, _ in dataset.samples])
    return loader, mask


@torch.no_grad()
def extract(forward, loader, device):
    """forward(x) -> (logits, phi). Returns penultimate features, logits, true labels (numpy)."""
    penult, logits, labels = [], [], []
    for x, y in loader:
        lo, phi = forward(x.to(device))
        penult.append(phi.cpu()); logits.append(lo.cpu()); labels.append(y)
    return torch.cat(penult).numpy(), torch.cat(logits).numpy(), torch.cat(labels).numpy()


def ot_bounds(penult, logits, y_true, labeled_mask, num_classes=10, n_permutations=4, device='cuda'):
    """gen_gap, OT-oracle and OT bound of one trained model."""
    preds = logits.argmax(axis=1)
    labeled_error = (preds[labeled_mask] != y_true[labeled_mask]).mean()
    unlabeled_error = (preds[~labeled_mask] != y_true[~labeled_mask]).mean()

    train_mask = torch.from_numpy(labeled_mask).to(device)
    res = ot_bound(torch.from_numpy(penult).to(device), margins_all(torch.from_numpy(logits).to(device)),
                   torch.from_numpy(y_true).long().to(device), train_mask, ~train_mask,
                   num_classes, n_permutations=n_permutations)
    return {'labeled_error': float(labeled_error), 'unlabeled_error': float(unlabeled_error),
            'gen_gap': abs(float(labeled_error) - float(unlabeled_error)),
            'ot_oracle': res['oracle'], 'ot_bound': res['test_blind']}
