import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

'''
Some functions borrowed ideas from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS
'''
def unprojection(height, width, u0, v0):
    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x.astype(np.float32)
    u_u0 = x - u0

    y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y.astype(np.float32)
    v_v0 = y - v0
    return u_u0, v_v0

def depth_to_pcd(depth, u_u0, v_v0, f):
    mask_invalid = depth <= 0
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


def data_prepare(depth, key, shift):
    '''
    key = 1: focal
    key = else: depth
    '''
    # optical centre
    u0 = depth.shape[1] / 2.0
    v0 = depth.shape[0] / 2.0
    upper = depth - depth.min()
    lower = depth.max()-depth.min()
    d_norm = upper / lower

    pov = (depth.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

    if key == 1:
        u_u0, v_v0 = unprojection(d_norm.shape[0], d_norm.shape[1], u0=u0, v0=v0)
        alpha = shift
        pcd, mask_valid = depth_to_pcd(d_norm, u_u0, v_v0, f=pov*alpha)
        feed_dict = pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f=os.POSIX_FADV_DONTNEED, voxel_size=0.005, mask_side=None)
        inputs = feed_dict['lidar'].cuda()
    else:
        u_u0, v_v0 = unprojection(d_norm.shape[0], d_norm.shape[1], u0=u0, v0=v0)
        delta = shift
        pcd_wshift, mask_valid = depth_to_pcd(d_norm + delta, u_u0, v_v0, f=pov)
        # input for the voxelnet
        feed_dict = pcd_to_sparsetensor(pcd_wshift, mask_valid, voxel_size=0.01)
        inputs = feed_dict['lidar'].cuda()
    return inputs


# import cv2
# path = '../../depth_zbuffer/point_0_view_0_domain_depth_zbuffer.png'
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#
# data_prepare(img, img, 2, 0).C


