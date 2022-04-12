import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize
import random
import math
import torch


def init_image_coor(height, width, u0, v0):
    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x.astype(np.float32)
    u_u0 = x - u0

    y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y.astype(np.float32)
    v_v0 = y - v0
    return u_u0, v_v0

def depth_to_pcd(depth, u_u0, v_v0, f, invalid_value=0):
    mask_invalid = depth <= invalid_value
    depth[mask_invalid] = 0.0
    x = u_u0 / f * depth
    y = v_v0 / f * depth
    z = depth
    pcd = np.stack([x, y, z], axis=2)
    return pcd, ~mask_invalid


def pcd_to_sparsetensor(pcd, mask_valid, voxel_size=0.01, num_points=100000):
    pcd_valid = pcd[mask_valid]
    block_ = pcd_valid
    block = np.zeros_like(block_)
    block[:, :3] = block_[:, :3]

    pc_ = np.round(block_[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = block

    # transfer point cloud to voxels
    inds = sparse_quantize(pc_,
                           feat_,
                           return_index=True,
                           return_invs=False)
    if len(inds) > num_points:
        inds = np.random.choice(inds, num_points, replace=False)

    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(feat, pc)
    feed_dict = [{'lidar': lidar}]
    inputs = sparse_collate_fn(feed_dict)
    return inputs

def pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f= 500.0, voxel_size=0.01, mask_side=None, num_points=100000):
    if mask_side is not None:
      mask_valid = mask_valid & mask_side
    pcd_valid = pcd[mask_valid]
    u_u0_valid = u_u0[mask_valid][:, np.newaxis] / f
    v_v0_valid = v_v0[mask_valid][:, np.newaxis] / f

    block_ = np.concatenate([pcd_valid, u_u0_valid, v_v0_valid], axis=1)
    block = np.zeros_like(block_)
    block[:, :] = block_[:, :]

    pc_ = np.round(block_[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = block

    # transfer point cloud to voxels
    inds = sparse_quantize(pc_,
                          feat_,
                          return_index=True,
                          return_invs=False)
    if len(inds) > num_points:
      inds = np.random.choice(inds, num_points, replace=False)

    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(feat, pc)
    feed_dict = [{'lidar': lidar}]
    inputs = sparse_collate_fn(feed_dict)
    return inputs

def refine_focal_one_step(depth, focal, model, u0, v0):
    u_u0, v_v0 = init_image_coor(depth.shape[0], depth.shape[1], u0=u0, v0=v0)
    pcd, mask_valid = depth_to_pcd(depth, u_u0, v_v0, f=focal, invalid_value=0)
    feed_dict = pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f=focal, voxel_size=0.005, mask_side=None)
    inputs = feed_dict['lidar'].cuda()

    outputs = model(inputs)
    return outputs

def refine_shift_one_step(depth_wshift, model, focal, u0, v0):
    # reconstruct PCD from depth
    u_u0, v_v0 = init_image_coor(depth_wshift.shape[0], depth_wshift.shape[1], u0=u0, v0=v0)
    pcd_wshift, mask_valid = depth_to_pcd(depth_wshift, u_u0, v_v0, f=focal, invalid_value=0)
    # input for the voxelnet
    feed_dict = pcd_to_sparsetensor(pcd_wshift, mask_valid, voxel_size=0.01)
    inputs = feed_dict['lidar'].cuda()

    outputs = model(inputs)
    return outputs

def refine_focal(depth, focal, model, u0, v0):
    last_scale = 1
    focal_tmp = np.copy(focal)
    for i in range(1):
        scale = refine_focal_one_step(depth, focal_tmp, model, u0, v0)
        focal_tmp = focal_tmp / scale.item()
        last_scale = last_scale * scale
    return torch.tensor([[last_scale]])

def refine_shift(depth_wshift, model, focal, u0, v0):
    depth_wshift_tmp = np.copy(depth_wshift)
    last_shift = 0
    for i in range(1):
        shift = refine_shift_one_step(depth_wshift_tmp, model, focal, u0, v0)
        shift = shift if shift.item() < 0.7 else torch.tensor([[0.7]])
        depth_wshift_tmp -= shift.item()
        last_shift += shift.item()
    return torch.tensor([[last_shift]])

def data_prepare(rgb, pred_depth, key, shift):
    '''
    key = 1: focal
    key = 2: depth
    key = 0: ground_truth
    '''
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax

    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

    if key == 1:
        u_u0, v_v0 = init_image_coor(pred_depth_norm.shape[0], pred_depth_norm.shape[1], u0=cam_u0, v0=cam_v0)
        alpha = shift
        pcd, mask_valid = depth_to_pcd(pred_depth_norm, u_u0, v_v0, f=proposed_scaled_focal*alpha, invalid_value=0)
        feed_dict = pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f=proposed_scaled_focal, voxel_size=0.005, mask_side=None)
        inputs = feed_dict['lidar'].cuda()
    elif key == 2:
        u_u0, v_v0 = init_image_coor(pred_depth_norm.shape[0], pred_depth_norm.shape[1], u0=cam_u0, v0=cam_v0)
        delta = shift
        pcd_wshift, mask_valid = depth_to_pcd(pred_depth_norm + delta, u_u0, v_v0, f=proposed_scaled_focal, invalid_value=0)
        # input for the voxelnet
        feed_dict = pcd_to_sparsetensor(pcd_wshift, mask_valid, voxel_size=0.01)
        inputs = feed_dict['lidar'].cuda()
    else:
        pass
    return inputs