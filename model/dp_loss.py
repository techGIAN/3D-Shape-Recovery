import torch
import torch.nn as nn
import torch.nn.init
import torchvision
import numpy as np

class DepthPredictionLoss(nn.Module):
    
    '''
        pr_d: The predicted depth
        gt_d: The ground truth depth
    '''
    def __init__(self, pr_d, gt_d):
        super(DepthPredictionLoss, self).__init__()

        self.pr_d = torch.Tensor(pr_d)
        self.gt_d = torch.Tensor(gt_d)

        gt_d_trim = torch.clamp(self.gt_d, 0.1, 0.9)
        mean_trim = torch.mean(gt_d_trim)
        sd_trim = torch.std(gt_d_trim)
        self.d_bar = (self.gt_d - mean_trim) / sd_trim

        # Image-level Normalized Regression Loss
        self.ilnr = self.image_level_normalized_regression_loss()

        # Paiwise Normal Loss
        sample_pr_d_A, sample_gt_d_A = self.sample_pair_points(
            1000)  # paper uses 100K sampled points but we reduce to 1000 for simplicity and save time
        surface_normal_A = self.surface_normal(sample_pr_d_A, sample_gt_d_A)
        sample_pr_d_B, sample_gt_d_B = self.sample_pair_points(
            1000)  # paper uses 100K sampled points but we reduce to 1000 for simplicity and save time
        surface_normal_B = self.surface_normal(sample_pr_d_B, sample_gt_d_B)
        self.pwn = self.pairwise_normal_loss(surface_normal_A, surface_normal_B, sample_gt_d_A, sample_gt_d_B)

        # Multi-scale Gradient Loss
        self.msg = self.multi_scale_gradient_loss()

        self.overall_loss = self.overall_loss_func()

    '''
        Get the Image-level Normalized Regression Loss (ILNR) given the predicted and ground truth depths
        1. Trim out values that are within the 10% furthest away
        2. Apply normalization based on the mean and std of the remaining
        This is done to prevent outliers.
        3. Apply the proposed ILNR formula.
    '''
    def image_level_normalized_regression_loss(self):
        return torch.mean(
            torch.abs(self.pr_d - self.d_bar) + torch.abs(torch.tanh(self.pr_d / 100) - torch.tanh(self.d_bar / 100)))

    '''
        Sample [num] points from pr_d and gt_d
        where [num] is the number of points that we wish to sample.
    '''
    def sample_pair_points(self, num):
        idx = torch.multinomial(self.pr_d.flatten(), num_samples=num, replacement=True)
        return self.pr_d.flatten()[idx], self.gt_d.flatten()[idx]
    
    '''
        Follows Xian et al's Structure-guided rank loss, which can imporve edge sharpness
        the sampling method is followed but enforced on surface normal space
        this improves global and local geometric relations
    '''
    def surface_normal(self, sample_pr_d, sample_gt_d):
        gr_d = sample_gt_d.squeeze()
        pe_d = sample_pr_d.squeeze()
        filtered = (gr_d > 1e-8) & (pe_d > 1e-8)

        a0, a1 = np.polyfit(pe_d[filtered], gr_d[filtered], deg=1)  # linear least squares fit
        return a0 * pe_d + a1

    '''
        Get the PWN loss given the ground truths and the normals
    '''
    def pairwise_normal_loss(self, n_A, n_B, gt_A, gt_B):
        pwn = torch.mean(torch.abs(n_A * n_B - gt_A * gt_B))
        return pwn
    
    '''
        Assume pr_d and d_bar are requires_grad=True.
        Compute MSG
    '''
    def multi_scale_gradient_loss(self):
        N = self.pr_d.flatten().size()[0]

        msg = 0
        pred_d = self.pr_d.detach().clone()
        depth_b = self.d_bar.detach().clone()

        pred_d.requires_grad_()
        pred_d.retain_grad()
        pred_d.backward(gradient=torch.ones(pred_d.size()))

        depth_b.requires_grad_()
        depth_b.retain_grad()
        depth_b.backward(gradient=torch.ones(depth_b.size()))
        for k in [2, 8, 32, 64]:  # k values from the existing paper
            msg += torch.abs((pred_d.grad) ** k - (depth_b.grad) ** k).sum()

        return msg / N

    '''
        Returns the overall loss.
    '''
    def overall_loss_func(self):
        lambda_a, lambda_g = 1, 0.5  # given constants in the paper
        return self.pwn + lambda_a * self.ilnr + lambda_g * self.msg

    '''
        Accessor
    '''
    def get_loss(self):
        return self.overall_loss
