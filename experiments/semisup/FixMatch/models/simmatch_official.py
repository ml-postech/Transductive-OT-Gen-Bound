"""SimMatch (CVPR 2022) — official algorithm with pluggable CNN backbone.

Adapted from https://github.com/mingkai-zheng/SimMatch/blob/main/models/simmatch.py with
a WideResNet backbone (WRNWrap) and single-GPU training: the distributed-only code
(all_gather, DDP batch shuffle, distribution alignment) is removed and EMAN-style EMA is
used.  return_reg_features=True also returns the student features needed by R.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wideresnet import build_wideresnet


class WRNWrap(nn.Module):
    """Backbone (WideResNet) + projection head."""

    def __init__(self, depth, widen_factor, c_depth, num_classes, dim=128, drop_rate=0.0):
        super().__init__()
        self.backbone = build_wideresnet(depth=depth,
                                         widen_factor=widen_factor,
                                         dropout=drop_rate,
                                         num_classes=num_classes,
                                         c_depth=c_depth)
        feat_dim = self.backbone.channels
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim),
        )

    def forward(self, x, return_phi=False):
        logits, phi = self.backbone(x, return_features=True)
        embedding = self.head(phi)
        embedding = F.normalize(embedding, dim=1)
        if return_phi:
            return logits, embedding, phi
        return logits, embedding


class SimMatch(nn.Module):
    def __init__(self, wrap_factory,
                 num_classes=10, momentum=0.999, dim=128, K=1200, args=None):
        super().__init__()
        self.momentum = momentum
        self.num_classes = num_classes

        self.main = wrap_factory()
        self.ema = wrap_factory()
        for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
            param_ema.data.copy_(param_main.data)
            param_ema.requires_grad = False

        # Memory bank: (dim, K) of normalized embeddings, plus K labels.
        self.register_buffer("bank",
                             F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("labels", torch.zeros(K, dtype=torch.long))

        if getattr(args, 'DA', False):
            raise NotImplementedError(
                'DA path needs distributed all_reduce; not adapted to single-GPU yet.')

    @torch.no_grad()
    def momentum_update_ema(self):
        """EMAN-style: copy num_batches_tracked verbatim, momentum-update the rest."""
        sd_main = self.main.state_dict()
        sd_ema = self.ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(sd_main.items(), sd_ema.items()):
            assert k_main == k_ema
            assert v_main.shape == v_ema.shape
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)

    @torch.no_grad()
    def _update_bank(self, k, labels, index):
        """k: (B, dim)  labels: (B,)  index: (B,) bank slot indices."""
        self.bank[:, index] = k.t()
        self.labels[index] = labels

    def forward(self, im_x, im_u_w=None, im_u_s=None,
                labels=None, index=None, start_unlabel=False, args=None,
                return_reg_features=False):
        if im_u_w is None and im_u_s is None:
            logits, _ = self.main(im_x)
            return logits

        batch_x = im_x.shape[0]
        batch_u = im_u_w.shape[0]
        bank = self.bank.clone().detach()

        # student: labeled + strong-aug unlabeled
        if return_reg_features:
            logits_q, feat_q, phi_q = self.main(
                torch.cat((im_x, im_u_s)), return_phi=True)
            phi_qx = phi_q[:batch_x]
        else:
            logits_q, feat_q = self.main(torch.cat((im_x, im_u_s)))
        logits_qx, logits_qu = logits_q[:batch_x], logits_q[batch_x:]
        feat_qx, feat_qu = feat_q[:batch_x], feat_q[batch_x:]

        # student on weak-aug unlabeled, for R
        if return_reg_features:
            logits_qu_w, _, phi_qu_w = self.main(im_u_w, return_phi=True)

        # teacher (EMA): labeled + weak-aug unlabeled
        with torch.no_grad():
            logits_k, feat_k = self.ema(torch.cat((im_x, im_u_w)))
            _, logits_ku = logits_k[:batch_x], logits_k[batch_x:]
            feat_kx, feat_ku = feat_k[:batch_x], feat_k[batch_x:]
            prob_ku_orig = F.softmax(logits_ku, dim=-1)

        if start_unlabel:
            with torch.no_grad():
                # teacher similarity over the bank, scaled by its class probability
                teacher_logits = feat_ku @ bank                        # (batch_u, K)
                teacher_prob_orig = F.softmax(teacher_logits / args.tt, dim=1)
                factor = prob_ku_orig.gather(
                    1, self.labels.expand([batch_u, -1]))              # (batch_u, K)
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if args.c_smooth < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros(
                        [bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(
                        1, self.labels.expand([bs, -1]), teacher_prob_orig)
                    prob_ku = (prob_ku_orig * args.c_smooth
                               + aggregated_prob * (1 - args.c_smooth))
                else:
                    prob_ku = prob_ku_orig

            student_logits = feat_qu @ bank                            # (batch_u, K)
            student_prob = F.softmax(student_logits / args.st, dim=1)
            loss_in = torch.sum(
                -teacher_prob.detach() * torch.log(student_prob + 1e-8), dim=1)
        else:
            loss_in = torch.tensor(0., device=im_x.device)
            prob_ku = prob_ku_orig

        # Bank update: EMA features on labeled, slot indices given by data loader.
        self._update_bank(feat_kx, labels, index)
        if return_reg_features:
            return logits_qx, prob_ku, logits_qu, loss_in, phi_qx, phi_qu_w, logits_qu_w
        return logits_qx, prob_ku, logits_qu, loss_in
