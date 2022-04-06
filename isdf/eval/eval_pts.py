# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import os
from torchvision import transforms

from isdf.modules import sample
from isdf.eval import metrics
from isdf.datasets import dataset, image_transforms
from isdf.datasets import sdf_util
from isdf.geometry import transform


def sub_eval(
    sdf_fn,
    vis_and_vox_pts, vis_not_vox_pts,
    vis_and_vox_sdf, vis_not_vox_sdf,
):
    n_vox = vis_and_vox_pts.shape[0]

    pts = np.concatenate((vis_and_vox_pts, vis_not_vox_pts))
    gt_sdf = np.concatenate((vis_and_vox_sdf, vis_not_vox_sdf))

    # vis region is all visible points
    # vox region is voxblox mapped region
    sdf = sdf_fn(pts)

    vox_sdf = sdf[:n_vox]
    vox_gt_sdf = gt_sdf[:n_vox]

    outs = []
    for (pred, gt) in zip([sdf, vox_sdf], [gt_sdf, vox_gt_sdf]):

        diff = pred - gt
        diff = np.abs(diff)
        l1_sdf = diff.mean().item()

        bins_loss = metrics.binned_losses(diff, gt)

        # chomp cost difference
        epsilons = [1., 1.5, 2.]
        l1_chomp_costs = [
            np.abs(
                metrics.chomp_cost(pred, epsilon=epsilon) -
                metrics.chomp_cost(gt, epsilon=epsilon)
            ).mean() for epsilon in epsilons
        ]

        out = {
            "av_l1": l1_sdf,
            "binned_l1": bins_loss,
            "l1_chomp_costs": l1_chomp_costs,
        }
        outs.append(out)

    sub_res = {
        "vis": outs[0],
        "vox": outs[1],
    }

    return sub_res


def eval_grad(sdf_interp, pts, delta, is_gt_sdf):
    grad = np.zeros(pts.shape)

    for i in range(3):
        for dx in [-1, 1]:

            offset = np.zeros(3)
            offset[i] += dx * delta
            offset_pts = pts + offset[None, :]

            if is_gt_sdf:
                sdf, valid_mask = sdf_util.eval_sdf_interp(
                    sdf_interp, offset_pts, handle_oob='mask')
                valid_mask = np.logical_and(sdf != 0., valid_mask)
                sdf[~valid_mask] = np.nan
            else:
                sdf = sdf_util.eval_sdf_interp(
                    sdf_interp, offset_pts, handle_oob='fill', oob_val=np.nan)

            grad[:, i] += dx * sdf

    grad /= (2 * delta)

    valid_mask = ~np.isnan(np.linalg.norm(grad, axis=1))

    return grad, valid_mask


def fixed_pts_eval(
    sdf_fn, t, eval_pts_dir, seq_dir, dataset_format,
    cached_dataset, dirs_C, gt_sdf_interp, eval_pts_root_vol,
    seq_len, grad_fn=None,
):
    t_str = f"{t:.3f}"
    pts_dir = os.path.join(eval_pts_dir, t_str)

    max_ix = min(np.floor(t * 30), seq_len)
    frame_ixs = np.arange(0, max_ix).astype(int)
    sample = cached_dataset[frame_ixs]
    depth_batch = torch.FloatTensor(sample["depth"])
    T_WC_batch = torch.FloatTensor(sample["T"])

    res = {
        "time": t,
    }

    # Load masks
    masks_dir = eval_pts_dir + t_str
    surf_valid_gt_sdf = np.load(masks_dir + "/surf_valid_gt_sdf.npy")
    surf_valid_vox_sdf = np.load(masks_dir + "/surf_valid_vox_sdf.npy")
    vis_valid_gt_sdf = np.load(masks_dir + "/vis_valid_gt_sdf.npy")
    vis_valid_vox_sdf = np.load(masks_dir + "/vis_valid_vox_sdf.npy")
    vis_valid_gt_grad = np.load(masks_dir + "/vis_valid_gt_grad.npy")
    vis_valid_vox_grad = np.load(masks_dir + "/vis_valid_vox_grad.npy")
    assert surf_valid_gt_sdf.sum() == surf_valid_vox_sdf.shape[0]
    assert vis_valid_gt_sdf.sum() == vis_valid_vox_sdf.shape[0]
    assert vis_valid_gt_grad.sum() == vis_valid_vox_grad.shape[0]

    # visible region

    pts = sample_visible_region(
        t_str, depth_batch, T_WC_batch, dataset_format, dirs_C)
    pts = pts[vis_valid_gt_sdf]
    gt_sdf, _ = sdf_util.eval_sdf_interp(gt_sdf_interp, pts, handle_oob='mask')

    sub_res = sub_eval(
        sdf_fn, pts[vis_valid_vox_sdf], pts[~vis_valid_vox_sdf],
        gt_sdf[vis_valid_vox_sdf], gt_sdf[~vis_valid_vox_sdf],
    )
    res['rays'] = sub_res

    # visible region grad

    pts_all = sample_visible_region(
        t_str, depth_batch, T_WC_batch, dataset_format, dirs_C)
    gt_grad_all, _ = eval_grad(gt_sdf_interp, pts_all, 0.01, is_gt_sdf=True)
    pts = pts_all[vis_valid_gt_grad]
    gt_grad = gt_grad_all[vis_valid_gt_grad]

    vox_region_gt_valid_grad = vis_valid_gt_grad[vis_valid_gt_sdf][
        vis_valid_vox_sdf]
    vox_grad_pts = pts_all[vis_valid_gt_sdf][vis_valid_vox_sdf][
        vox_region_gt_valid_grad]
    gt_grad_vox = gt_grad_all[vis_valid_gt_sdf][vis_valid_vox_sdf][
        vox_region_gt_valid_grad]

    if grad_fn is None:  # for gpu fusion eval
        vis_grad_1, valid_grad_1 = eval_grad(
            sdf_fn, pts, 0.07, is_gt_sdf=False)
        vis_grad_1[~valid_grad_1] = np.array([1., 0., 0.])
        vis_grad_2, valid_grad_2 = eval_grad(
            sdf_fn, pts, 0.14, is_gt_sdf=False)
        vis_grad_2[~valid_grad_2] = np.array([1., 0., 0.])

        vox_grad_1, valid_grad_1 = eval_grad(
            sdf_fn, vox_grad_pts, 0.07, is_gt_sdf=False)
        vox_grad_1[~valid_grad_1] = np.array([1., 0., 0.])
        vox_grad_2, valid_grad_2 = eval_grad(
            sdf_fn, vox_grad_pts, 0.14, is_gt_sdf=False)
        vox_grad_2[~valid_grad_2] = np.array([1., 0., 0.])

    else:  # for iSDF eval
        vis_grad_1 = grad_fn(pts)
        vis_grad_2 = vis_grad_1
        vox_grad_1 = grad_fn(vox_grad_pts)
        vox_grad_2 = vox_grad_1

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    cosdist_vis_1 = 1 - cos(torch.tensor(vis_grad_1), torch.tensor(gt_grad))
    cosdist_vis_1 = cosdist_vis_1.mean().item()
    cosdist_vis_2 = 1 - cos(torch.tensor(vis_grad_2), torch.tensor(gt_grad))
    cosdist_vis_2 = cosdist_vis_2.mean().item()

    cosdist_vox_1 = 1 - cos(
        torch.tensor(vox_grad_1), torch.tensor(gt_grad_vox))
    cosdist_vox_1 = cosdist_vox_1.mean().item()
    cosdist_vox_2 = 1 - cos(
        torch.tensor(vox_grad_2), torch.tensor(gt_grad_vox))
    cosdist_vox_2 = cosdist_vox_2.mean().item()

    res['rays']['vox']['av_cossim'] = [cosdist_vox_1, cosdist_vox_1]
    res['rays']['vis']['av_cossim'] = [cosdist_vis_1, cosdist_vis_2]

    # surface region

    pts = sample_surface(t_str, depth_batch, T_WC_batch, dirs_C)
    pts = pts[surf_valid_gt_sdf]
    gt_sdf, _ = sdf_util.eval_sdf_interp(gt_sdf_interp, pts, handle_oob='mask')

    sub_res = sub_eval(
        sdf_fn, pts[surf_valid_vox_sdf], pts[~surf_valid_vox_sdf],
        gt_sdf[surf_valid_vox_sdf], gt_sdf[~surf_valid_vox_sdf],
    )
    res['visible_surf'] = sub_res

    # objects eval

    obj_bounds = None
    obj_bounds_file = seq_dir + '/obj_bounds.txt'
    if os.path.exists(obj_bounds_file):
        obj_bounds = load_obj_bounds(obj_bounds_file)

    obj_res = []
    if obj_bounds is not None:
        for i, bounds in enumerate(obj_bounds):

            obj_files = [x for x in os.listdir(pts_dir) if f'obj{i}' in x]
            if len(obj_files) == 0:
                continue

            valid_gt_sdf = np.load(masks_dir + f"/obj{i}_valid_gt_sdf.npy")
            valid_vox_sdf = np.load(masks_dir + f"/obj{i}_valid_vox_sdf.npy")

            pts = object_eval_pts(bounds)
            pts = pts[valid_gt_sdf]
            gt_sdf, gt_valid_mask = sdf_util.eval_sdf_interp(
                gt_sdf_interp, pts, handle_oob='mask')

            vis_and_vox_pts = pts[valid_vox_sdf]
            vis_not_vox_pts = pts[~valid_vox_sdf]
            vis_and_vox_sdf = gt_sdf[valid_vox_sdf]
            vis_not_vox_sdf = gt_sdf[~valid_vox_sdf]

            n_vox = vis_and_vox_pts.shape[0]

            pts = np.concatenate((vis_and_vox_pts, vis_not_vox_pts))
            gt_sdf = np.concatenate((vis_and_vox_sdf, vis_not_vox_sdf))

            # vis region is all visible points
            # vox region is voxblox mapped region
            sdf = sdf_fn(pts)

            vox_sdf = sdf[:n_vox]
            vox_gt_sdf = gt_sdf[:n_vox]

            diff = sdf - gt_sdf
            diff = np.abs(diff)
            l1_sdf = diff.mean().item()

            vox_diff = vox_sdf - vox_gt_sdf
            vox_diff = np.abs(vox_diff)
            vox_l1_sdf = vox_diff.mean().item()

            sub_res = {
                "vis": {"av_l1": l1_sdf},
                "vox": {"av_l1": vox_l1_sdf},
            }
            obj_res.append(sub_res)

        res["objects"] = obj_res

    # full volume eval

    seq = [x for x in seq_dir.split('/') if x != ""][-1]

    if dataset_format == "replicaCAD":
        vol_pts_file = eval_pts_root_vol + "full_vol/replicaCAD.npy"
        gt_sdf_file = eval_pts_root_vol + f"full_vol/gt_{seq}.npy"
    if dataset_format == "ScanNet":
        vol_pts_file = eval_pts_root_vol + f"full_vol/{seq}.npy"
        gt_sdf_file = eval_pts_root_vol + f"full_vol/gt_{seq}.npy"

    pts = np.load(vol_pts_file)
    gt_sdf = np.load(gt_sdf_file)

    sdf = sdf_fn(pts)

    diff = sdf - gt_sdf
    diff = np.abs(diff)
    l1_sdf = diff.mean().item()

    bins_loss = metrics.binned_losses(diff, gt_sdf)

    # chomp cost difference
    epsilons = [1., 1.5, 2.]
    l1_chomp_costs = [
        np.abs(
            metrics.chomp_cost(sdf, epsilon=epsilon) -
            metrics.chomp_cost(gt_sdf, epsilon=epsilon)
        ).mean().item() for epsilon in epsilons
    ]

    out = {
        "av_l1": l1_sdf,
        "binned_l1": bins_loss,
        "l1_chomp_costs": l1_chomp_costs,
    }

    res["vol"] = out

    return res


def load_obj_bounds(obj_bounds_file):

    obj_bounds = np.loadtxt(obj_bounds_file)
    obj_bounds = obj_bounds.reshape(-1, 2, 3)

    # box 20cms around and above the object
    obj_bounds[:, 1] += 0.08
    obj_bounds[:, 0, 0] -= 0.08
    obj_bounds[:, 0, 2] -= 0.08

    return obj_bounds


def object_eval_pts(bounds):

    obj_samples = 10000

    np.random.seed(0)
    offsets = np.random.rand(obj_samples, 3)
    extents = bounds[1] - bounds[0]
    pts = bounds[0] + offsets * extents[None, :]

    return pts


def set_seed(t_str):
    seed = float(t_str) * 1e3
    torch.manual_seed(seed)


def sample_visible_region(
    t_str, depth_batch, T_WC_batch, dataset_format, dirs_C,
):
    dist_behind_surf = 0.1
    if dataset_format == "ScanNet":
        dist_behind_surf = 0.

    return sample_rays(
        t_str, depth_batch, T_WC_batch, dist_behind_surf, dirs_C,
        sample_surface=False,
    )


def sample_surface(
    t_str, depth_batch, T_WC_batch, dirs_C,
):
    return sample_rays(
        t_str, depth_batch, T_WC_batch, 0., dirs_C,
        sample_surface=True,
    )


def sample_rays(
    t_str, depth_batch, T_WC_batch,
    dist_behind_surf, dirs_C,
    sample_surface,
):
    set_seed(t_str)

    samples = 200000
    min_depth = 0.1
    device = depth_batch.device

    rays_per_frame = samples // depth_batch.shape[0]
    indices_b, indices_h, indices_w = sample.sample_pixels(
        rays_per_frame,
        depth_batch.shape[0],
        depth_batch.shape[1],
        depth_batch.shape[2],
        device=device)

    depth_sample = depth_batch[indices_b, indices_h, indices_w].view(-1)
    mask_valid_depth = depth_sample != 0
    depth_sample = depth_sample[mask_valid_depth]

    indices_b = indices_b[mask_valid_depth]
    indices_h = indices_h[mask_valid_depth]
    indices_w = indices_w[mask_valid_depth]

    T_WC_sample = T_WC_batch[indices_b]
    dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(-1, 3)

    origins, dirs_W = transform.origin_dirs_W(T_WC_sample, dirs_C_sample)

    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)
    n_rays = dirs_W.shape[0]

    if sample_surface is False:
        max_depth = depth_sample + dist_behind_surf
        z_vals = sample.stratified_sample(
            min_depth, max_depth, n_rays, device, n_stratified_samples=1)
    else:
        z_vals = depth_sample[:, None]

    pts = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    pts = pts.squeeze()

    return pts


def get_cache_dataset(seq_dir, dataset_format, scannet_dir):
    # Use gt traj file for evaluation
    traj_file = seq_dir + "/traj.txt"

    if dataset_format == "replicaCAD":
        ims_file = seq_dir + "results/"
        col_ext = ".png"
        inv_depth_scale = 1. / 3276.75
    elif dataset_format == "ScanNet":
        col_ext = ".jpg"
        inv_depth_scale = 1. / 1000.
        ims_file = scannet_dir

    rgb_transform = transforms.Compose([image_transforms.BGRtoRGB()])
    depth_transform = transforms.Compose(
        [image_transforms.DepthScale(inv_depth_scale),
         image_transforms.DepthFilter(12.0)])

    keep_ixs = None
    # seq = [x for x in seq_dir.split('/') if x != ""][-1]
    n_frames = np.loadtxt(traj_file).shape[0]
    keep_ixs = np.arange(0, n_frames, 5)

    cached_dataset = dataset.SceneCache(
        dataset_format,
        ims_file,
        traj_file,
        rgb_transform=rgb_transform,
        depth_transform=depth_transform,
        col_ext=col_ext,
        keep_ixs=keep_ixs,
    )

    return cached_dataset


def get_dirs_C(dataset_format, scannet_dir):
    if dataset_format == "replicaCAD":
        H, W = 680, 1200
        fx, fy = W / 2., W / 2.
        cx, cy = W / 2. - 0.5, H / 2. - 0.5
    elif dataset_format == "ScanNet":
        intrinsic_file = scannet_dir + "/frames/intrinsic/intrinsic_depth.txt"
        K = np.loadtxt(intrinsic_file)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        H, W = 480, 640
    dirs_C = transform.ray_dirs_C(
        1, H, W, fx, fy, cx, cy, "cpu", depth_type="z")

    return dirs_C
