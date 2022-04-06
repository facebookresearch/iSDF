# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from isdf.geometry import transform

cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def bounds_ray(depth_sample, z_vals, dirs_C_sample, T_WC_sample, do_grad):
    bounds = depth_sample[:, None] - z_vals
    z_to_euclidean_depth = dirs_C_sample.norm(dim=-1)
    bounds = z_to_euclidean_depth[:, None] * bounds

    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)

    return bounds, grad


def bounds_normal(
    depth_sample, z_vals, dirs_C_sample, norm_sample, normal_trunc_dist,
    T_WC_sample, do_grad,
):
    ray_bounds = bounds_ray(depth_sample, z_vals, dirs_C_sample)

    costheta = torch.abs(cosSim(-dirs_C_sample, norm_sample))

    # only apply correction out to truncation distance
    sub = normal_trunc_dist * (1. - costheta)
    normal_bounds = ray_bounds - sub[:, None]

    trunc_ixs = ray_bounds < normal_trunc_dist
    trunc_vals = (ray_bounds * costheta[:, None])[trunc_ixs]
    normal_bounds[trunc_ixs] = trunc_vals

    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)

    return normal_bounds, grad


def grad_ray(T_WC_sample, dirs_C_sample, n_samples):
    """ Returns the negative of the viewing direction vector """
    _, dirs_W = transform.origin_dirs_W(T_WC_sample, dirs_C_sample)
    grad = - dirs_W[:, None, :].repeat(1, n_samples, 1)

    return grad


def bounds_pc(pc, z_vals, depth_sample, do_grad=True):
    with torch.set_grad_enabled(False):
        surf_pc = pc[:, 0]
        diff = pc[:, :, None] - surf_pc
        dists = diff.norm(dim=-1)
        dists, closest_ixs = dists.min(axis=-1)
        behind_surf = z_vals > depth_sample[:, None]
        dists[behind_surf] *= -1
        bounds = dists

        grad = None
        if do_grad:
            ix1 = torch.arange(
                diff.shape[0])[:, None].repeat(1, diff.shape[1])
            ix2 = torch.arange(
                diff.shape[1])[None, :].repeat(diff.shape[0], 1)
            grad = diff[ix1, ix2, closest_ixs]
            grad = grad[:, 1:]
            grad = grad / grad.norm(dim=-1)[..., None]
            # flip grad vectors behind the surf
            grad[behind_surf[:, 1:]] *= -1

        # # vis gradient vector
        # surf_pc_tm = trimesh.PointCloud(
        #     surf_pc.reshape(-1, 3).cpu(), colors=[255, 0, 0])
        # pc_tm = trimesh.PointCloud(pc[:, 1:].reshape(-1, 3).cpu())
        # closest_surf_pts = surf_pc[closest_ixs].reshape(-1, 3)
        # lines = torch.cat((
        #     closest_surf_pts[:, None, :],
        #     pc.reshape(-1, 3)[:, None, :]), dim=1)
        # paths = trimesh.load_path(lines.cpu())
        # trimesh.Scene([surf_pc_tm, pc_tm, paths]).show()

    return bounds, grad


def bounds(
    method,
    dirs_C_sample,
    depth_sample,
    T_WC_sample,
    z_vals,
    pc,
    normal_trunc_dist,
    norm_sample,
    do_grad=True,
):
    """ do_grad: compute approximate gradient vector. """
    assert method in ["ray", "normal", "pc"]

    if method == "ray":
        bounds, grad = bounds_ray(
            depth_sample, z_vals, dirs_C_sample, T_WC_sample, do_grad
        )

    elif method == "normal":
        bounds, grad = bounds_normal(
            depth_sample, z_vals, dirs_C_sample,
            norm_sample, normal_trunc_dist, T_WC_sample, do_grad)

    else:
        bounds, grad = bounds_pc(pc, z_vals, depth_sample, do_grad)

    return bounds, grad


def sdf_loss(sdf, bounds, t, loss_type="L1"):
    """
        params:
        sdf: predicted sdf values.
        bounds: upper bound on abs(sdf)
        t: truncation distance up to which the sdf value is directly supevised.
        loss_type: L1 or L2 loss.
    """
    free_space_loss_mat, trunc_loss_mat = full_sdf_loss(sdf, bounds)

    free_space_ixs = bounds > t
    free_space_loss_mat[~free_space_ixs] = 0.
    trunc_loss_mat[free_space_ixs] = 0.

    sdf_loss_mat = free_space_loss_mat + trunc_loss_mat

    if loss_type == "L1":
        sdf_loss_mat = torch.abs(sdf_loss_mat)
    elif loss_type == "L2":
        sdf_loss_mat = torch.square(sdf_loss_mat)
    else:
        raise ValueError("Must be L1 or L2")

    return sdf_loss_mat, free_space_ixs


def full_sdf_loss(sdf, target_sdf, free_space_factor=5.0):
    """
    For samples that lie in free space before truncation region:
        loss(sdf_pred, sdf_gt) =  { max(0, sdf_pred - sdf_gt), if sdf_pred >= 0
                                  { exp(-sdf_pred) - 1, if sdf_pred < 0

    For samples that lie in truncation region:
        loss(sdf_pred, sdf_gt) = sdf_pred - sdf_gt
    """

    free_space_loss_mat = torch.max(
        torch.nn.functional.relu(sdf - target_sdf),
        torch.exp(-free_space_factor * sdf) - 1.
    )
    trunc_loss_mat = sdf - target_sdf

    return free_space_loss_mat, trunc_loss_mat


def tsdf_loss(sdf, target_sdf, trunc_dist):
    """
    tsdf loss from: https://arxiv.org/pdf/2104.04532.pdf
    SDF values in truncation region are scaled in range [0, 1].
    """
    free_space_loss_mat = sdf - torch.ones(sdf.shape, device=sdf.device)
    trunc_loss_mat = sdf - target_sdf / trunc_dist

    return free_space_loss_mat, trunc_loss_mat


def tot_loss(
    sdf_loss_mat, grad_loss_mat, eik_loss_mat,
    free_space_ixs, bounds, eik_apply_dist,
    trunc_weight, grad_weight, eik_weight,
):
    sdf_loss_mat[~free_space_ixs] *= trunc_weight
    # print("zero losses",
    #       sdf_loss_mat.numel() - sdf_loss_mat.nonzero().shape[0])

    losses = {"sdf_loss": sdf_loss_mat.mean().item()}
    tot_loss_mat = sdf_loss_mat

    # surface normal loss
    if grad_loss_mat is not None:
        tot_loss_mat = tot_loss_mat + grad_weight * grad_loss_mat
        losses["grad_loss"] = grad_loss_mat.mean().item()

    # eikonal loss
    if eik_loss_mat is not None:
        eik_loss_mat[bounds < eik_apply_dist] = 0.
        eik_loss_mat = eik_loss_mat * eik_weight
        tot_loss_mat = tot_loss_mat + eik_loss_mat
        losses["eikonal_loss"] = eik_loss_mat.mean().item()

    tot_loss = tot_loss_mat.mean()
    losses["total_loss"] = tot_loss

    return tot_loss, tot_loss_mat, losses


def approx_loss(full_loss, binary_masks, W, H, factor=8):
    w_block = W // factor
    h_block = H // factor
    loss_approx = full_loss.view(-1, factor, h_block, factor, w_block)
    loss_approx = loss_approx.sum(dim=(2, 4))
    actives = binary_masks.view(-1, factor, h_block, factor, w_block)
    actives = actives.sum(dim=(2, 4))
    actives[actives == 0] = 1.0
    loss_approx = loss_approx / actives

    return loss_approx


def frame_avg(
    total_loss_mat, depth_batch, indices_b, indices_h, indices_w,
    W, H, loss_approx_factor, binary_masks,
):

    # frame average losses
    full_loss = torch.zeros(
        depth_batch.shape, device=depth_batch.device)
    full_loss[indices_b, indices_h, indices_w] = total_loss_mat.sum(
        -1).detach()

    loss_approx = approx_loss(
        full_loss, binary_masks, W, H,
        factor=loss_approx_factor
    )
    factor = loss_approx.shape[1]
    frame_sum = loss_approx.sum(dim=(1, 2))
    frame_avg_loss = frame_sum / (factor * factor)

    return loss_approx, frame_avg_loss
