# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import trimesh
import torch

from isdf import visualisation
from isdf.geometry import transform
from isdf.modules import sample


def get_frustum_normals(R_WC, H, W, fx, fy, cx, cy):
    c = np.array([0, W, W, 0])
    r = np.array([0, 0, H, H])
    x = (c - cx) / fx
    y = (r - cy) / fy
    corner_dirs_C = np.vstack((x, y, np.ones(4))).T
    corner_dirs_W = (R_WC * corner_dirs_C[..., None, :]).sum(axis=-1)

    frustum_normals = np.empty((4, 3))
    frustum_normals[0] = np.cross(corner_dirs_W[0], corner_dirs_W[1])
    frustum_normals[1] = np.cross(corner_dirs_W[1], corner_dirs_W[2])
    frustum_normals[2] = np.cross(corner_dirs_W[2], corner_dirs_W[3])
    frustum_normals[3] = np.cross(corner_dirs_W[3], corner_dirs_W[0])
    frustum_normals = frustum_normals / np.linalg.norm(
        frustum_normals, axis=1)[:, None]

    return frustum_normals


def check_inside_frustum(points, cam_center, frustum_normals):
    """ For a point to be within the frustrum, the projection on each normal
        vector must be positive.
        params:
    """
    pts = points - cam_center
    dots = np.dot(pts, frustum_normals.T)
    return (dots >= 0).all(axis=1)


def is_visible(points, T_WC, depth, H, W, fx, fy, cx, cy,
               trunc=0.2, use_projection=True):
    """ Are points visible to in this frame.
        Up to trunc metres behind the surface counts in visible region.
    """
    # forward project points
    K = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]])
    ones = np.ones([len(points), 1])
    homog_points = np.concatenate((points, ones), axis=-1)
    points_C = (np.linalg.inv(T_WC) @ homog_points.T)[:3]
    uv = K @ points_C
    z = uv[2]
    uv = uv[:2] / z
    uv = uv.T

    if use_projection:
        x_valid = np.logical_and(uv[:, 0] > 0, uv[:, 0] < W)
        y_valid = np.logical_and(uv[:, 1] > 0, uv[:, 1] < H)
        xy_valid = np.logical_and(x_valid, y_valid)
    else:  # use frustrum
        R_WC = T_WC[:3, :3]
        cam_center = T_WC[:3, 3]

        frustum_normals = get_frustum_normals(
            R_WC, H, W, fx, fy, cx, cy)

        xy_valid = check_inside_frustum(
            points, cam_center, frustum_normals)

    uv = uv.astype(int)
    depth_vals = depth[uv[xy_valid, 1], uv[xy_valid, 0]]
    max_depths = np.full(len(uv), -np.inf)
    max_depths[xy_valid] = depth_vals + trunc
    z_valid = np.logical_and(z > 0, z < max_depths)

    inside = np.logical_and(xy_valid, z_valid)

    return inside


def is_visible_torch(points, T_WC_batch, depth_batch, H, W, fx, fy, cx, cy,
                     trunc=0.2, use_projection=True):
    """ Are points visible to in this frame.
        Up to trunc metres behind the surface counts in visible region.
    """
    # forward project points
    K = torch.tensor(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]], device=points.device).float()
    ones = torch.ones([len(points), 1], device=points.device)
    homog_points = torch.cat((points, ones), dim=-1)

    T_CW = torch.linalg.inv(T_WC_batch)
    points_C = torch.matmul(T_CW, homog_points.transpose(0, 1))[:, :3, :]
    uv = torch.matmul(K, points_C)
    z = uv[:, 2]
    uv = uv[:, :2] / z[:, None]
    uv = uv.transpose(1, 2)

    if use_projection:
        x_valid = torch.logical_and(uv[..., 0] > 0, uv[..., 0] < W)
        y_valid = torch.logical_and(uv[..., 1] > 0, uv[..., 1] < H)
        xy_valid = torch.logical_and(x_valid, y_valid)
    else:  # use frustrum
        R_WC = T_WC_batch[:, :3, :3]
        cam_center = T_WC_batch[:, :3, 3]

        frustum_normals = get_frustum_normals(
            R_WC, H, W, fx, fy, cx, cy)

        xy_valid = check_inside_frustum(
            points, cam_center, frustum_normals)

    uv = uv.long()
    uv_xyvalid = uv[xy_valid]
    frame_ix = torch.arange(len(depth_batch), device=points.device)
    xy_valid_per_frame = xy_valid.sum(dim=-1)
    frame_ix = torch.repeat_interleave(frame_ix, xy_valid_per_frame)
    depth_vals = depth_batch[frame_ix, uv_xyvalid[:, 1], uv_xyvalid[:, 0]]
    max_depths = torch.full(xy_valid.shape, -np.inf, device=points.device)
    max_depths[xy_valid] = depth_vals + trunc
    z_valid = torch.logical_and(z > 0, z < max_depths)

    inside = torch.logical_and(xy_valid, z_valid)

    return inside


def test_inside_frustum(T_WC, depth):
    fx, fy = 600., 600.
    cx, cy = 600., 340.
    H, W = 680, 1200.

    # show camera
    scene = trimesh.Scene()
    visualisation.draw3D.draw_cams(1, T_WC, scene)

    # show random point cloud
    points = np.random.normal(0., 2., [1000, 3])
    visible = is_visible(points, T_WC, depth, H, W, fx, fy, cx, cy)
    cols = np.full(points.shape, [255, 0, 0])
    cols[visible] = [0, 255, 0]
    pc = trimesh.PointCloud(points, cols)
    scene.add_geometry(pc)

    # show rays
    sparse = 20
    dirs_C = transform.ray_dirs_C(
        1, int(H / sparse), int(W / sparse), fx / sparse,
        fy / sparse, cx / sparse, cy / sparse, 'cpu', depth_type='z')
    dirs_C = dirs_C.view(1, -1, 3)
    dirs_C = dirs_C.cpu().numpy()
    dirs_W = (T_WC[:3, :3] * dirs_C[..., None, :]).sum(axis=-1)
    n_rays = dirs_W.shape[1]
    sparse_depth = depth[::sparse, ::sparse]
    max_depth = torch.from_numpy(sparse_depth + 0.9).flatten()
    z_vals = sample.stratified_sample(
        0.2, max_depth, n_rays, 'cpu', n_bins=12)
    dirs_W = torch.from_numpy(dirs_W)
    dirs_W = dirs_W.squeeze()
    origins = torch.from_numpy(T_WC[:3, 3])
    origins = origins[None, :].repeat(n_rays, 1)
    rays_pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    rays_pc = rays_pc.reshape(-1, 3).numpy()
    visible_rays = is_visible(rays_pc, T_WC, depth, H, W, fx, fy, cx, cy)
    ray_col = np.full(rays_pc.shape, [255, 0, 0])
    ray_col[visible_rays] = [0, 255, 0]
    rays_tmpc = trimesh.PointCloud(rays_pc, ray_col)
    scene.add_geometry(rays_tmpc)

    # show frustum normals
    starts = T_WC[:3, 3][None, :].repeat(4, 0)
    frustum_normals = get_frustum_normals(T_WC[:3, :3], H, W, fx, fy, cx, cy)
    normal_ends = T_WC[:3, 3] + frustum_normals * 4
    normal_lines = np.concatenate(
        (starts[:, None, :], normal_ends[:, None, :]), axis=1)
    normal_paths = trimesh.load_path(normal_lines)
    normal_paths.colors = [[255, 255, 0, 255]] * 3
    scene.add_geometry(normal_paths)

    # show rays in corners of frame
    # ends = C + corner_dirs_W * 3
    # lines = np.concatenate((starts[:, None, :], ends[:, None, :]), axis=1)
    # paths = trimesh.load_path(lines)
    # paths.colors = [[0, 255, 0, 255]] * len(lines)
    # scene.add_geometry(paths)

    scene.show()


if __name__ == "__main__":

    test_inside_frustum()
