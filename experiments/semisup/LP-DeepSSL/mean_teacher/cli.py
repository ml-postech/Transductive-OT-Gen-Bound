# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import re
import argparse
import logging

from . import architectures, datasets


LOG = logging.getLogger('main')

__all__ = ['parse_commandline_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
    parser.add_argument('--train-subdir', type=str, default='all',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='all',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--label-split', default=10, type=int, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--labeled-batch-size', default=50, type=int,
                        metavar='N', help="labeled examples per minibatch (default: 50; unused with --exclude-unlabeled)")
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=140, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=True, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=5, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--checkpoint-epochs', default=0, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to keep only best.ckpt (default: 0)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='gpu id')
    parser.add_argument('--dfs-k', type=int, default=50,
                        help='diffusion k')
    parser.add_argument('--fully-supervised', default=False, type=str2bool, metavar='BOOL',
                        help='is fully-supervised')
    parser.add_argument('--isL2', default=True, type=str2bool, metavar='BOOL',
                        help='is l2 normalized features')
    parser.add_argument('--num-labeled', type=int, default=None,
                        help='number of labeled instances (default: 1200 for cifar10, 2000 for svhn)')
    parser.add_argument('--test-mode', type=str, default='',
                        help='number of labeled instances')
    parser.add_argument('--isMT', default=False, type=str2bool, metavar='BOOL',
                        help='is combined with mean teacher')
    parser.add_argument('--datadir', default='', type=str, metavar='PATH',
                        help='override data directory (default: use path from datasets.py)')
    parser.add_argument('--update-interval', default=1, type=int,
                        help='update pseudo-labels every N epochs (default: 1, larger = faster)')
    parser.add_argument('--max-iter', default=20, type=int,
                        help='max iterations for CG solver in label propagation (default: 20)')
    parser.add_argument('--arch', default='cifar_cnn_flex', type=str,
                        help='model architecture')
    parser.add_argument('--n-blocks', default=3, type=int,
                        help='number of conv blocks in FlexCNN (default: 3)')
    parser.add_argument('--base-channels', default=128, type=int,
                        help='base channel width in FlexCNN (default: 128)')
    parser.add_argument('--c-depth', default=1, type=int,
                        help='classifier depth in FlexCNN: c_depth-1 hidden '
                             'Linear+LeakyReLU layers before the final head '
                             '(default: 1 = single linear head, old behavior)')

    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='random seed; if set, fixes torch/numpy/random + cudnn deterministic for repeatable runs')

    # OT regularizer R
    parser.add_argument('--reg_weight', type=float, default=0.0,
                        help='lambda of the OT regularizer R (0 = off)')
    parser.add_argument('--sinkhorn_eps', type=float, default=0.05,
                        help='entropic OT epsilon for Sinkhorn W1')
    parser.add_argument('--sinkhorn_iter', type=int, default=50,
                        help='Sinkhorn iterations')
    parser.add_argument('--soft_temp', type=float, default=1.0,
                        help='temperature of the smoothed max in M_c')
    parser.add_argument('--reg_n_sample', type=int, default=200,
                        help='number of pairs sampled for change-rate')
    parser.add_argument('--buffer_size', type=int, default=4,
                        help='number of past labeled batches kept in the feature buffer')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


