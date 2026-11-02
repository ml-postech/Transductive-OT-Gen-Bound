import logging
import math
import os

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)

NUM_LABELED = {'cifar10': 1200, 'svhn': 2000}
LABEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'LP-DeepSSL', 'data-local', 'labels')


def set_dataset_defaults(args):
    """Fill --num-labeled, --data-dir and --label-file from --dataset when not given."""
    args.num_labeled = args.num_labeled or NUM_LABELED[args.dataset]
    args.data_dir = args.data_dir or os.path.join('data', args.dataset, 'by-image')
    args.label_file = args.label_file or os.path.join(
        LABEL_DIR, args.dataset, f'{args.num_labeled}_balanced_labels', '10.txt')


class TransformFixMatch(object):
    def __init__(self, mean, std, use_flip=True):
        weak_ops = []
        strong_ops = []
        if use_flip:
            weak_ops.append(transforms.RandomHorizontalFlip())
            strong_ops.append(transforms.RandomHorizontalFlip())
        weak_ops.append(transforms.RandomCrop(
            size=32, padding=int(32*0.125), padding_mode='reflect'))
        strong_ops.append(transforms.RandomCrop(
            size=32, padding=int(32*0.125), padding_mode='reflect'))
        strong_ops.append(RandAugmentMC(n=2, m=10))
        self.weak = transforms.Compose(weak_ops)
        self.strong = transforms.Compose(strong_ops)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


# Transductive setting: the whole pool (all/) is the unlabeled set; labels come from the label file

class ImageFolderSSL(datasets.ImageFolder):
    """ImageFolder with index-based subset and optional transform override."""
    def __init__(self, root, indexs=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples


def load_label_file(label_file_path):
    """Parse LP-DeepSSL style label file → set of labeled filenames."""
    labeled_filenames = set()
    with open(label_file_path) as f:
        for line in f.read().splitlines():
            if line.strip():
                filename = line.split(' ')[0]
                labeled_filenames.add(filename)
    return labeled_filenames


def get_transductive(args, root):
    """
    Transductive data loader over the all/ pool; args needs data_dir and label_file.
    """
    if args.dataset == 'svhn':
        mean, std = svhn_mean, svhn_std
        use_flip = False   # digits are not horizontally symmetric
    else:
        mean, std = cifar10_mean, cifar10_std
        use_flip = True

    labeled_ops = []
    if use_flip:
        labeled_ops.append(transforms.RandomHorizontalFlip())
    labeled_ops.extend([
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_labeled = transforms.Compose(labeled_ops)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    data_dir = os.path.join(args.data_dir, 'all')

    full_dataset = datasets.ImageFolder(data_dir)
    labeled_filenames = load_label_file(args.label_file)

    labeled_idx = []
    unlabeled_idx = list(range(len(full_dataset)))  # all data as unlabeled (FixMatch style)
    for idx, (path, _) in enumerate(full_dataset.samples):
        if os.path.basename(path) in labeled_filenames:
            labeled_idx.append(idx)

    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)

    logger.info(f"Transductive pool: {len(full_dataset)} images")
    logger.info(f"Labeled: {len(labeled_idx)}, Unlabeled: {len(unlabeled_idx)}")

    if args.expand_labels or len(labeled_idx) < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / len(labeled_idx))
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    train_labeled_dataset = ImageFolderSSL(
        data_dir, indexs=labeled_idx, transform=transform_labeled)

    train_unlabeled_dataset = ImageFolderSSL(
        data_dir, indexs=unlabeled_idx,
        transform=TransformFixMatch(mean=mean, std=std, use_flip=use_flip))

    # In transductive setting, "test" = unlabeled samples of the pool with eval transform
    labeled_set = set(labeled_idx.tolist())
    test_dataset = ImageFolderSSL(
        data_dir, indexs=[i for i in range(len(full_dataset)) if i not in labeled_set],
        transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


DATASET_GETTERS = {'cifar10': get_transductive, 'svhn': get_transductive}
