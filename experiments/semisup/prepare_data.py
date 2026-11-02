import argparse
import os

import torchvision


def dump(ds, out_dir, prefix, class_name):
    for i, (img, y) in enumerate(ds):
        name = class_name(y)
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        path = os.path.join(out_dir, name, f'{prefix}{i}_{name}.png')
        if not os.path.exists(path):
            img.save(path)


def link_all(by_image, splits):
    for split in splits:
        for c in os.listdir(os.path.join(by_image, split)):
            os.makedirs(os.path.join(by_image, 'all', c), exist_ok=True)
            for f in os.listdir(os.path.join(by_image, split, c)):
                dst = os.path.join(by_image, 'all', c, f)
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(os.path.join(by_image, split, c, f)), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data')
    root = ap.parse_args().root
    raw = os.path.join(root, 'raw')

    by_image = os.path.join(root, 'cifar10', 'by-image')
    train = torchvision.datasets.CIFAR10(raw, train=True, download=True)
    test = torchvision.datasets.CIFAR10(raw, train=False, download=True)
    dump(train, os.path.join(by_image, 'train+val'), '', lambda y: train.classes[y])
    dump(test, os.path.join(by_image, 'test'), 'test_', lambda y: train.classes[y])
    link_all(by_image, ['train+val', 'test'])

    by_image = os.path.join(root, 'svhn', 'by-image')
    for split, prefix in [('train', ''), ('test', 'test_')]:
        ds = torchvision.datasets.SVHN(raw, split=split, download=True)
        dump(ds, os.path.join(by_image, split), prefix, str)
    link_all(by_image, ['train', 'test'])


if __name__ == '__main__':
    main()
