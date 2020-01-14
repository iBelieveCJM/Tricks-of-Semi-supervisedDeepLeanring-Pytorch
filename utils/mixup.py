#!coding:utf-8
import torch
from torch.nn import functional as F

import numpy as np

def mixup_one_target(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam*x + (1-lam)*x[index, :]
    mixed_y = lam*y + (1-lam)*y[index]
    return mixed_x, mixed_y, lam


def mixup_two_targets(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam*x + (1-lam)*x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_ce_loss_soft(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss for soft labels
    """
    mixup_loss_a = -torch.mean(torch.sum(targets_a* F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(targets_b* F.log_softmax(preds, dim=1), dim=1))

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss


def mixup_ce_loss_hard(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss
    """
    mixup_loss_a = F.nll_loss(F.log_softmax(preds, dim=1), targets_a)
    mixup_loss_b = F.nll_loss(F.log_softmax(preds, dim=1), targets_b)

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss


def mixup_ce_loss_with_softmax(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss
    """
    mixup_loss_a = -torch.mean(torch.sum(F.softmax(targets_a,1)* F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(F.softmax(targets_b,1)* F.log_softmax(preds, dim=1), dim=1))

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss


def mixup_mse_loss_with_softmax(preds, targets_a, targets_b, lam):
    """ mixed categorical mse loss
    """
    mixup_loss_a = F.mse_loss(F.softmax(preds,1), F.softmax(targets_a,1))
    mixup_loss_b = F.mse_loss(F.softmax(preds,1), F.softmax(targets_b,1))

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss
