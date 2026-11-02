# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import torchvision.transforms as transforms

from . import data
from .utils import export

import os

@export
def cifar10(isTwice=True):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))


    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]

    data_dir = '../../../data/cifar10/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }


@export
def svhn(isTwice=True):
    channel_stats = dict(mean=[0.4377, 0.4438, 0.4728],
                         std=[0.1980, 0.2010, 0.1970])

    # no horizontal flip for digits
    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = '../../../data/svhn/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }
