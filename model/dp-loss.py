import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision
import numpy as np
import random as rnd

class DepthPredictionLoss(nn.Module):
  
  '''
    pr_d: The predicted depth
    gt_d: The ground truth depth
  '''
  def __init__(self, pr_d, gt_d):
    super(DepthPredictionLoss, self).__init__()
    self.d_bar = None
    
    # Image-level Normalized Regression Loss
    ilnr = self.image_level_normalized_regression_loss(pr_d, gt_d)
    
    # Paiwise Normal Loss
    sampled_points =  self.sample_pair_points(pr_d, 100000) # 100K sampled points
    sample_gt_d_A = self.get_gt_d_of_sample(sampled_points, pr_d, gt_d)
    surface_normal_A = self.surface_normal(sample_gt_d_A)
    sampled_points =  self.sample_pair_points(pr_d, 100000) # 100K sampled points
    sample_gt_d_B = self.get_gt_d_of_sample(sampled_points, pr_d, gt_d)
    surface_normal_B = self.surface_normal(sample_gt_d_B)
    pwn = self.pairwise_normal_loss(surface_normal_A, surface_normal_B, sample_gt_d_A, sample_gt_d_B)
    
    # Multi-scale Gradient Loss
    msg = self.multi_scale_gradient_loss(gt_d, self.d_bar) # not sure about this
    
    self.overall_loss = self.overall_loss(pwn, ilnr, msg)

  '''
    Get the Image-level Normalized Regression Loss (ILNR) given the predicted and ground truth depths
    1. Trim out values that are within the 10% furthest away
    2. Apply normalization based on the mean and std of the remaining
    This is done to prevent outliers.
    3. Apply the proposed ILNR formula.
  '''
  def image_level_normalized_regression_loss(self, pr_d, gt_d):
    gt_d_trim = torch.stack([x for x in gt_d if 0.1 < gt_d < 0.9])
    d_bar = (gt_d - torch.mean(gt_d_trim))/torch.std(gt_d)
    self.d_bar = d_bar
    return torch.mean(torch.abs(pr_d - d_bar) + torch.abs(torch.tanh(pr_d/100) - torch.tanh(d_bar/100)))

  '''
    Sample [num] points from [pr_d]
    where [num] is the number of points that we wish to sample.
  '''
  def sample_pair_points(self, pr_d, num):
    idx = torch.multinomial(num_samples=num, replacement=False)
    return pr_d[idx]

  '''
    Get the ground truth of a sample [smp]
    [smp] is a list
  '''
  def get_gt_d_of_sample(self, smp, pr_d, gt_d):
    pos = [(pr_d == x).nonzero().item() for x in smp]   # get the pos list of the samples from  the pr_d
    return torch.stack([gt_d[x] for x in pos])

  '''
    Follows Xian et al's Structure-guided rank loss, which can imporve edge sharpness
    the sampling method is followed but enforced on surface normal space
    this improves global and local geometric relations
  '''
  def surface_normal(self, sample_gt_d)
    # to-do
    # requires 3D point cloud
    # least squares fit

  '''
    Get the PWN loss given the ground truths and the normals
  '''
  def pairwise_normal_loss(self, n_A, n_B, gt_A, gt_B):
    pwn = torch.mean(torch.abs(n_A*n_B - gt_A*gt_B))
    return pwn
  
  '''
    Assume gt_d and d_bar are requires_grad=True.
    Compute MSG
  '''
  def multi_scale_gradient_loss(self, gt_d, d_bar):
    
    K = [2, 8, 32, 64]    # k values from the existing paper
    N = gt_d.size()

    msg = 0
    gt_d.backward(gradient=torch.tensor([1., 1.]))
    d_bar.backward(gradient=torch.tensor([1., 1.]))
    for k in K:
        msg += torch.abs((gt_d[0].grad)**k - (d_bar[0].grad)**k) +  torch.abs((gt_d[1].grad)**k - (d_bar[1].grad)**k)

    return msg/N

  '''
    Returns the overall loss.
  '''
  def overall_loss(self, pwn, ilnr, msg):
    lambda_a, lambda_g = 1, 0.5     # given constants in the paper
    return self.pwn + lambda_a*self.ilnr + lambda_g*self.msg