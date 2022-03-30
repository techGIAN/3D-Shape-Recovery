import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision
import numpy as np
import random as rnd

class BaselineLoss(nn.Module):

    '''
        pr_d: The predicted depth
        gt_d: The ground truth depth
    '''
    def __init__(self, pr_d, gt_d):
        super(BaselineLoss, self).__init__()
    

    '''
        Scale Invariant Loss by Wang et al.
    '''
    def scale_invariant_loss(self, pr_d, gt_d):

        K = [2, 8, 32, 64]
        num1, num2, denom1, denom2 = 0, 0, 0, 0
        gt_d.backward(gradient=torch.tensor([1., 1.]))
        pr_d.backward(gradient=torch.tensor([1., 1.]))

        for k in K:
            num1 += torch.sum(torch.abs(pr_d[0].grad)**k)
            num2 += torch.sum(torch.abs(pr_d[1].grad)**k)
            denom1 += torch.sum(torch.abs(gt_d[0].grad)**k)
            denom2 += torch.sum(torch.abs(gt_d[0].grad)**k)
        s = (num1 + num2)/(denom1 + denom2)

        ssil = 0

        for k in K:
            ssil += torch.abs(s*pr_d[0].grad**k - gt_d[0].grad**k)
            ssil += torch.abs(s*pr_d[1].grad**k - gt_d[1].grad**k) 

        return ssil

    
    '''
        Scale Shift Invariant Loss by Ranftl et al.
    '''
    def scale_shift_invariant_loss(self, pr_d, gt_d):

        def t(d):
            return torch.median(d)

        def s(d):
            return torch.mean(torch.abs(d - t(d)))

        d_hat = (pr_d - t(pr_d))/s(pr_d)
        d_hat_star = (gt_d - t(gt_d))/s(gt_d)

        R = d_hat - d_hat_star
        R.backward(gradient=torch.tensor([1., 1.]))

        def lreg(d, dd):
            K = [1, 2, 4, 8] # K=4 scale levels, half image resolution at each level
            lreg = 0
            for k in K:
                lreg += torch.mean(torch.abs(R[0].grad**k) + torch.abs(R[1].grad**k))
            return lreg

        N = pr_d.size()
        alpha = 0.5 # given in their paper

        lssi = 0

        for n in range(N):
            lssi += scale_invariant_loss(d_hat**n, d_hat_star**n) + lreg(d_hat**n, d_hat_star**n)

        return lssi/N
