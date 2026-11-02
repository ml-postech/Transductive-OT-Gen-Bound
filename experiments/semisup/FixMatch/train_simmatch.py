"""SimMatch training (single-GPU adaptation of mingkai-zheng/SimMatch), with the OT regularizer R.

Follows the official engine.py train() loop and simmatch.py main() hyperparameters.
Changes from official:
  1. Backbone = WideResNet (via models.simmatch_official.SimMatch with WRNWrap),
     with a configurable classifier depth (c_depth).
  2. Single-GPU path only -- distributed code (DDP samplers, all_gather, batch shuffle) removed.
  3. --reg_weight > 0 adds lambda * R (otbound.ssl.ssl_ot_regularizer):
         loss = loss_x + lambda_u * loss_u + lambda_in * loss_in + reg_weight * R
     R needs student features of the labelled and the weak unlabelled batch; the stock
     forward has no student pass on the weak view, so SimMatch.forward(...,
     return_reg_features=True) adds one (about 1/(2mu+1) extra compute).

Default hyperparameters mirror the official simmatch.py argparse:
  threshold=0.7, lambda-u=10, lambda-in=1, mu=5, st=0.1, tt=0.1, c-smooth=1.0, ema-m=0.999.

Checkpoint format (read by bound_utils.load_wrn):
  state_dict      = model.main.backbone.state_dict()      (backbone keys only)
  ema_state_dict  = model.ema.backbone.state_dict()
"""
import argparse
import logging
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS, set_dataset_defaults
from utils import AverageMeter, accuracy
from models.simmatch_official import SimMatch, WRNWrap

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root, for otbound
from otbound.ssl import ssl_ot_regularizer, LabeledFeatureBuffer

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def cosine_lr(step, total_steps, base_lr, warmup_steps=0):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class IndexedDataset(Dataset):
    """Wrap a Dataset so __getitem__ returns (img, target, idx)."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i


def build_label_tensor(labeled_dataset_idx, num_labeled, device):
    base = labeled_dataset_idx.base
    targets = torch.tensor(list(base.targets), dtype=torch.long)
    assert targets.shape[0] == num_labeled, (targets.shape, num_labeled)
    return targets.to(device)


def parse_args():
    p = argparse.ArgumentParser('SimMatch (official algorithm + WRN backbone)')
    p.add_argument('--gpu-id', default=0, type=int)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'svhn'])
    p.add_argument('--data-dir', default='', type=str)    # default: data/<dataset>/by-image
    p.add_argument('--label-file', default='', type=str)  # default: the 2% split in LP-DeepSSL/data-local/labels
    p.add_argument('--num-labeled', type=int, default=None)
    p.add_argument('--expand-labels', action='store_true')
    p.add_argument('--num-classes', type=int, default=10)
    p.add_argument('--arch', type=str, default='wideresnet', choices=['wideresnet'])
    p.add_argument('--model-depth', type=int, default=28)
    p.add_argument('--model-width', type=int, default=2)
    p.add_argument('--c-depth', type=int, default=1)
    p.add_argument('--total-steps', type=int, default=50000)
    p.add_argument('--eval-step', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.03)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--wdecay', type=float, default=5e-4)
    p.add_argument('--nesterov', action='store_true', default=True)
    # SimMatch (official defaults)
    p.add_argument('--mu', type=int, default=5)
    p.add_argument('--lambda-u', type=float, default=10.0)
    p.add_argument('--lambda-in', type=float, default=1.0)
    p.add_argument('--threshold', type=float, default=0.7)
    p.add_argument('--st', type=float, default=0.1)
    p.add_argument('--tt', type=float, default=0.1)
    p.add_argument('--c_smooth', type=float, default=1.0)
    p.add_argument('--ema-m', type=float, default=0.999)
    p.add_argument('--proj-dim', type=int, default=128)
    p.add_argument('--DA', action='store_true', default=False)
    # OT regularizer R
    p.add_argument('--reg_weight', type=float, default=0.0)  # lambda of R (0 = off)
    p.add_argument('--sinkhorn_eps', type=float, default=0.05)
    p.add_argument('--sinkhorn_iter', type=int, default=50)
    p.add_argument('--soft_temp', type=float, default=1.0)
    p.add_argument('--reg_n_sample', type=int, default=200)
    p.add_argument('--buffer_size', type=int, default=4)
    p.add_argument('--out', default='')  # default: results/semisup/<run name>
    p.add_argument('--seed', default=None, type=int)
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--local_rank', default=-1, type=int)
    args = p.parse_args()
    set_dataset_defaults(args)
    if not args.out:
        name = f'simmatch_{args.dataset}_d{args.model_depth}_w{args.model_width}_cd{args.c_depth}'
        if args.reg_weight > 0: name += f'_reg{args.reg_weight:g}'
        if args.seed is not None: name += f'_seed{args.seed}'
        args.out = os.path.join('results', 'semisup', name)
    return args


def main():
    args = parse_args()
    global best_acc

    device = torch.device('cuda', args.gpu_id)
    args.device = device
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.distributed = False

    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s  %(message)s',
                        datefmt='%m/%d %H:%M:%S', level=logging.INFO)
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out) if SummaryWriter is not None else None

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, args.data_dir)
    labeled_idx_dataset = IndexedDataset(labeled_dataset)
    bank_init_labels = build_label_tensor(labeled_idx_dataset, args.num_labeled, device)

    labeled_loader = DataLoader(
        labeled_idx_dataset, sampler=RandomSampler(labeled_idx_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers)

    args.epochs = math.ceil(args.total_steps / args.eval_step)

    def wrap_factory():
        return WRNWrap(args.model_depth, args.model_width, args.c_depth,
                       args.num_classes, dim=args.proj_dim)

    model = SimMatch(
        wrap_factory=wrap_factory,
        num_classes=args.num_classes,
        momentum=args.ema_m, dim=args.proj_dim,
        K=args.num_labeled, args=args).to(device)
    model.labels.copy_(bank_init_labels)

    n_total = sum(p.numel() for p in model.parameters())
    n_main = sum(p.numel() for p in model.main.parameters())
    n_ema = sum(p.numel() for p in model.ema.parameters())
    logger.info(
        f'Total params: {n_total/1e6:.2f}M  '
        f'(main {n_main/1e6:.2f}M, ema {n_ema/1e6:.2f}M)')

    optimizer = optim.SGD(model.main.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.wdecay,
                          nesterov=args.nesterov)

    arch_str = f'WRN d{args.model_depth}_w{args.model_width}_cd{args.c_depth}'
    logger.info(
        f'***** SimMatch ({arch_str}) '
        f'num_labeled={args.num_labeled}  total_steps={args.total_steps} '
        f'lr={args.lr} mu={args.mu} threshold={args.threshold} '
        f'lambda_u={args.lambda_u} lambda_in={args.lambda_in} '
        f'reg_weight={args.reg_weight} *****')

    train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer)


def train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer):
    global best_acc
    test_accs = []
    end = time.time()
    labeled_iter = iter(labeled_loader)
    step = 0  # global step

    use_reg = args.reg_weight > 0
    feat_buffer = (LabeledFeatureBuffer(args.buffer_size)
                   if use_reg else None)

    model.train()
    for epoch in range(args.epochs):
        losses = AverageMeter(); losses_x = AverageMeter()
        losses_u = AverageMeter(); losses_in = AverageMeter()
        losses_r = AverageMeter()
        mask_info = AverageMeter(); pseudo_acc = AverageMeter()
        batch_time = AverageMeter(); data_time = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))

        u_iter = iter(unlabeled_loader)
        batch_idx = 0
        while batch_idx < args.eval_step:
            try:
                images_u_pack, targets_u = next(u_iter)
            except StopIteration:
                u_iter = iter(unlabeled_loader)
                images_u_pack, targets_u = next(u_iter)

            try:
                images_x, targets_x, index = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                images_x, targets_x, index = next(labeled_iter)

            images_u_w, images_u_s = images_u_pack
            data_time.update(time.time() - end)

            images_x = images_x.to(args.device, non_blocking=True)
            images_u_w = images_u_w.to(args.device, non_blocking=True)
            images_u_s = images_u_s.to(args.device, non_blocking=True)
            targets_x = targets_x.to(args.device, non_blocking=True)
            targets_u = targets_u.to(args.device, non_blocking=True)
            index = index.to(args.device, non_blocking=True)

            for g in optimizer.param_groups:
                g['lr'] = cosine_lr(step, args.total_steps, args.lr, args.warmup)
            cur_lr = optimizer.param_groups[0]['lr']

            # R also needs the student features of the weak view
            if use_reg:
                (logits_x, pseudo_label, logits_u_s, loss_in,
                 phi_x, phi_u_w, logits_u_w) = model(
                    images_x, images_u_w, images_u_s,
                    labels=targets_x, index=index,
                    start_unlabel=(epoch > 0), args=args,
                    return_reg_features=True)
            else:
                logits_x, pseudo_label, logits_u_s, loss_in = model(
                    images_x, images_u_w, images_u_s,
                    labels=targets_x, index=index,
                    start_unlabel=(epoch > 0), args=args)

            max_probs, pseudo_targets = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
            loss_u = (torch.sum(-F.log_softmax(logits_u_s, dim=1) * pseudo_label.detach(),
                                dim=1) * mask).mean()
            loss_in = loss_in.mean()

            # OT regularizer R
            reg = torch.tensor(0.0, device=args.device)
            if use_reg:
                reg = ssl_ot_regularizer(
                    phi_x, targets_x, logits_x, phi_u_w, logits_u_w, args.num_classes,
                    buffer=feat_buffer, sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_iter=args.sinkhorn_iter, soft_temp=args.soft_temp,
                    n_sample=args.reg_n_sample)

            loss = (loss_x + args.lambda_u * loss_u
                    + args.lambda_in * loss_in + args.reg_weight * reg)

            losses.update(loss.item())
            losses_x.update(loss_x.item(), images_x.size(0))
            losses_u.update(loss_u.item(), images_u_w.size(0))
            losses_in.update(loss_in.item(), images_u_w.size(0))
            losses_r.update(reg.item())
            mask_info.update(mask.mean().item(), mask.size(0))

            with torch.no_grad():
                bool_mask = mask.bool()
                if bool_mask.any():
                    correct = (pseudo_label.max(1)[1][bool_mask] == targets_u[bool_mask]).float()
                    pseudo_acc.update(correct.mean().item() * 100, bool_mask.sum().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.momentum_update_ema()

            batch_time.update(time.time() - end); end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    f'Ep {epoch+1}/{args.epochs} It {batch_idx+1}/{args.eval_step} '
                    f'lr={cur_lr:.4f} L={losses.avg:.3f} '
                    f'Lx={losses_x.avg:.3f} Lu={losses_u.avg:.3f} Lin={losses_in.avg:.3f} '
                    f'Lr={losses_r.avg:.3f} '
                    f'msk={mask_info.avg:.2f} pAcc={pseudo_acc.avg:.1f}')
                p_bar.update()
            step += 1; batch_idx += 1

        if not args.no_progress:
            p_bar.close()

        test_loss, test_acc = test(args, test_loader, model, epoch)
        if args.writer is not None:
            args.writer.add_scalar('train/loss', losses.avg, epoch)
            args.writer.add_scalar('train/loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/loss_in', losses_in.avg, epoch)
            args.writer.add_scalar('train/loss_r', losses_r.avg, epoch)
            args.writer.add_scalar('train/mask', mask_info.avg, epoch)
            args.writer.add_scalar('train/pseudo_acc', pseudo_acc.avg, epoch)
            args.writer.add_scalar('test/acc', test_acc, epoch)
            args.writer.add_scalar('test/loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.main.backbone.state_dict(),
            'ema_state_dict': model.ema.backbone.state_dict(),
            'main_head_state_dict': model.main.head.state_dict(),
            'ema_head_state_dict': model.ema.head.state_dict(),
            'bank': model.bank.detach().cpu(),
            'bank_labels': model.labels.detach().cpu(),
            'acc': test_acc, 'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info(f'Best top-1 acc: {best_acc:.2f}')
        logger.info(f'Mean top-1 acc (last 20): {np.mean(test_accs[-20:]):.2f}')

    if args.writer is not None:
        args.writer.close()


def test(args, test_loader, model, epoch):
    losses = AverageMeter(); top1 = AverageMeter(); top5 = AverageMeter()
    if not args.no_progress:
        test_loader = tqdm(test_loader)
    test_model = model.ema

    test_model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            logits, _ = test_model(inputs)
            loss = F.cross_entropy(logits, targets)
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
        if not args.no_progress:
            test_loader.close()
    logger.info(f'top-1 acc: {top1.avg:.2f}  top-5 acc: {top5.avg:.2f}')
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
