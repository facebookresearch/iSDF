# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from isdf.geometry import transform


def sample_pixels(
    n_rays, n_frames, h, w, device
):
    total_rays = n_rays * n_frames
    indices_h = torch.randint(0, h, (total_rays,), device=device)
    indices_w = torch.randint(0, w, (total_rays,), device=device)

    indices_b = torch.arange(n_frames, device=device)
    indices_b = indices_b.repeat_interleave(n_rays)

    return indices_b, indices_h, indices_w


def get_batch_data(
    depth_batch,
    T_WC_batch,
    dirs_C,
    indices_b,
    indices_h,
    indices_w,
    norm_batch=None,
    get_masks=False,
):
    """
    Get depth, ray direction and pose for the sampled pixels.
    Only render where depth is valid.
    """
    depth_sample = depth_batch[indices_b, indices_h, indices_w].view(-1)
    mask_valid_depth = depth_sample != 0

    norm_sample = None
    if norm_batch is not None:
        norm_sample = norm_batch[indices_b,
                                 indices_h,
                                 indices_w, :].view(-1, 3)
        mask_invalid_norm = torch.isnan(norm_sample[..., 0])
        mask_valid_depth = torch.logical_and(
            mask_valid_depth, ~mask_invalid_norm)
        norm_sample = norm_sample[mask_valid_depth]

    depth_sample = depth_sample[mask_valid_depth]

    indices_b = indices_b[mask_valid_depth]
    indices_h = indices_h[mask_valid_depth]
    indices_w = indices_w[mask_valid_depth]

    T_WC_sample = T_WC_batch[indices_b]
    dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(-1, 3)

    masks = None
    if get_masks:
        masks = torch.zeros(depth_batch.shape, device=depth_batch.device)
        masks[indices_b, indices_h, indices_w] = 1

    return (
        dirs_C_sample,
        depth_sample,
        norm_sample,
        T_WC_sample,
        masks,
        indices_b,
        indices_h,
        indices_w
    )


def stratified_sample(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.

    If n_stratified_samples is passed then use fixed number of bins,
    else if bin_length is passed use fixed bin size.
    """
    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            bin_limits = torch.linspace(
                0, 1, n_bins + 1,
                device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            if isinstance(min_depth, torch.Tensor):
                bin_limits = bin_limits + min_depth[:, None]
            else:
                bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            bin_limits = torch.linspace(
                min_depth,
                max_depth,
                n_bins + 1,
                device=device,
            )[None, :]
            bin_length = (max_depth - min_depth) / (n_bins)

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments

    return z_vals


def sample_along_rays(
    T_WC,
    min_depth,
    max_depth,
    n_stratified_samples,
    n_surf_samples,
    dirs_C,
    gt_depth=None,
    grad=False,
):
    with torch.set_grad_enabled(grad):
        # rays in world coordinate
        origins, dirs_W = transform.origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        n_rays = dirs_W.shape[0]

        # stratified sampling along rays # [total_n_rays, n_stratified_samples]
        z_vals = stratified_sample(
            min_depth, max_depth,
            n_rays, T_WC.device,
            n_stratified_samples,
            bin_length=None,
        )

        # if gt_depth is given, first sample at surface then around surface
        if gt_depth is not None and n_surf_samples > 0:
            surface_z_vals = gt_depth
            offsets = torch.normal(
                torch.zeros(gt_depth.shape[0], n_surf_samples - 1), 0.1
            ).to(z_vals.device)
            near_surf_z_vals = gt_depth[:, None] + offsets
            if not isinstance(min_depth, torch.Tensor):
                min_depth = torch.full(near_surf_z_vals.shape, min_depth).to(
                    z_vals.device)[..., 0]
            near_surf_z_vals = torch.clamp(
                near_surf_z_vals,
                min_depth[:, None],
                max_depth[:, None],
            )
            z_vals = torch.cat(
                (surface_z_vals[:, None], near_surf_z_vals, z_vals), dim=1)

        # point cloud of 3d sample locations
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])

    return pc, z_vals


"""
For integrated PE.
"""

# diag = False
# mean, covs = embedding.cast_rays(
#     z_vals, origins, dirs_W, diag=diag)
# if do_sdf_grad:
#     mean.requires_grad_()
# samples = (mean, covs)

# import trimesh
# scene = trimesh.Scene([
#     trimesh.PointCloud(origins[0:1].cpu(), [255, 0, 0]),
#     trimesh.PointCloud(pc[0].detach().cpu()),
#     trimesh.PointCloud(mean[0].detach().cpu(), [0, 255, 0]),
# ])
# if diag is False:
#     e, v = torch.linalg.eig(covs[0])
#     print(e)
# for i in range(3):
#     starts, ends = mean[0].detach().clone(), mean[0].detach().clone()
#     if diag:
#         starts[:, i] += covs[0, :, i]
#         ends[:, i] -= covs[0, :, i]
#     else:
#         starts += v[:, :, i].real * e[:, i].real[:, None]
#         ends -= v[:, :, i].real * e[:, i].real[:, None]
#     lines = torch.cat((starts[:, None, :], ends[:, None, :]), dim=1)

#     scene.add_geometry(trimesh.load_path(lines.cpu()))

# scene.show()
# import ipdb; ipdb.set_trace()

# surf_dists = None
# if gt_depth is not None:
#     z_to_euclidean_depth = dirs_C.norm(dim=-1)
#     ray_target = gt_depth.flatten()[:, None] - z_vals
#     surf_dists = z_to_euclidean_depth[:, None] * ray_target
#     surf_dists = surf_dists.squeeze(-1)
