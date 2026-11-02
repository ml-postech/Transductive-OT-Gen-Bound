# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import re
import argparse
import os
import random
import shutil
import time
import math


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

from helpers import *

args = None
best_prec1 = 0
global_step = 0



def main():
    global global_step
    global best_prec1

    # Name of the model to be trained
    if args.isMT:
    	model_name = '%s_%d_mt_ss_split_%d_isL2_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    else:
    	model_name = '%s_%d_ss_split_%d_isL2_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    if args.arch == 'cifar_cnn_flex':
    	model_name += '_nb%d_bc%d_cd%d_k%d_bs%d' % (
    		args.n_blocks, args.base_channels, args.c_depth,
    		args.dfs_k, args.batch_size)
    	if abs(args.lr - 0.05) > 1e-6:
    		model_name += '_lr%03d' % int(round(args.lr * 1000))
    	if args.seed is not None:
    		model_name += '_seed%d' % args.seed
    if args.reg_weight > 0:
        model_name += '_reg_lam%g' % args.reg_weight

    if args.dataset in ('svhn',):
        checkpoint_path = os.path.join('models', args.dataset, model_name)
    else:
        checkpoint_path = 'models/%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'a')

    # Create the dataset and loaders
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    if args.datadir:
        dataset_config['datadir'] = args.datadir
    train_loader, eval_loader, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)

    # Create the model
    model = create_model(num_classes,args)

    # If Mean Teacher is turned on, create the ema model
    if args.isMT:
        ema_model = create_model(num_classes,args,ema=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    if args.seed is None:
        cudnn.benchmark = True

    # Name of the model trained in Stage 1
    if args.isMT:
    	stage1_name = '%s_%d_mean_teacher_split_%d_isL2_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    else:
    	stage1_name = '%s_%d_split_%d_isL2_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    if args.arch == 'cifar_cnn_flex':
    	stage1_name += '_nb%d_bc%d_cd%d_k%d_bs%d' % (
    		args.n_blocks, args.base_channels, args.c_depth,
    		args.dfs_k, args.batch_size)
    	if abs(args.lr - 0.05) > 1e-6:
    		stage1_name += '_lr%03d' % int(round(args.lr * 1000))
    	if args.seed is not None:
    		stage1_name += '_seed%d' % args.seed
    if args.reg_weight > 0:
        stage1_name += '_reg_lam%g' % args.reg_weight
    if args.dataset in ('svhn',):
        resume_fn = os.path.join('models', args.dataset, stage1_name, 'best.ckpt')
    else:
        resume_fn = 'models/%s/best.ckpt' % stage1_name

    # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    # Compute the starting accuracy
    prec1, prec5 = validate(eval_loader, model, global_step, args.start_epoch, isMT = args.isMT)
    if args.isMT:
        ema_prec1, ema_prec5  = validate(eval_loader, ema_model, global_step, args.start_epoch, isMT = args.isMT)

    print('Resuming from:%s' % resume_fn)

    sel_acc = float('nan')

    feat_buffer = LabeledFeatureBuffer(args.buffer_size) \
        if args.reg_weight > 0 else None

    for epoch in range(args.start_epoch, args.epochs):
        # Extract features and update the pseudolabels (every update_interval epochs)
        if epoch % args.update_interval == 0:
            print('Extracting features...')
            feats,labels = extract_features(train_loader_noshuff, model, isMT = args.isMT)
            sel_acc = train_data.update_plabels(feats, k = args.dfs_k, max_iter = args.max_iter)
            print('selection accuracy: %.2f' % (sel_acc))
        
        #  Train for one epoch with the new pseudolabels
        if args.isMT:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args, ema_model=ema_model, feat_buffer=feat_buffer)
        else:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args, feat_buffer=feat_buffer)

        # Evaluate the model
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print("Evaluating the primary model:")
            prec1, prec5 = validate(eval_loader, model, global_step, epoch + 1, isMT = args.isMT)

            if args.isMT:
                print("Evaluating the EMA model:")
                ema_prec1, ema_prec5  = validate(eval_loader, ema_model, global_step, epoch + 1, isMT = args.isMT)
                is_best = ema_prec1 > best_prec1
                best_prec1 = max(ema_prec1, best_prec1)
            else:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)                
        else:
            is_best = False

        # Write to the log file and save the checkpoint
        if args.isMT:
            log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % 
                (epoch,
                train_meter['class_loss'].avg,
                train_meter['lr'].avg,
                train_meter['top1'].avg,
                train_meter['top5'].avg,
                prec1,
                prec5,
                ema_prec1,
                ema_prec5)
            )
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)
            elif is_best:
                best_path = os.path.join(checkpoint_path, 'best.ckpt')
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, best_path)
                print("--- best checkpoint saved to %s ---" % best_path)

        else:
            log.write('%d,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n' %
                (epoch,
                train_meter['class_loss'].avg,
                train_meter['lr'].avg,
                train_meter['top1'].avg,
                train_meter['top5'].avg,
                prec1,
                prec5,
                )
            )
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)
            elif is_best:
                best_path = os.path.join(checkpoint_path, 'best.ckpt')
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, best_path)
                print("--- best checkpoint saved to %s ---" % best_path)


if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()

    # Set the other settings
    args = load_args(args, isMT = args.isMT)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        print('[seed] fixed to %d (random/numpy/torch/cuda + cudnn deterministic)' % args.seed)

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled,args.label_split,args.gpu_id))

    main()
