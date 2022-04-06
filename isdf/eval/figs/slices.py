# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
If slices are saved for iSDF, this script generates slices
for the other methods and creates plots comparing the slices.
"""


import numpy as np
import torch
import cv2
import os
import json
import time
import trimesh
import imgviz
import matplotlib.pylab as plt

from isdf.datasets import sdf_util
from isdf.eval import metrics, eval_pts, plot_utils
from isdf.geometry import frustum, transform


def load_ours(exp_dir, n_slices, dataset_format, eval_t=None):
    # load saved images
    slice_dir = exp_dir + "/z_slice/"

    ours = []
    gt = []
    diff = []
    ours_chomp = []
    gt_chomp = []
    diff_chomp = []

    prefix = ""
    if eval_t is not None:
        prefix = f"{eval_t:.3f}_"
    else:
        times = [
            x.split('_')[0] for x in os.listdir(slice_dir)
            if x[-10:] == 'pred_0.png'
        ]
        times.remove('pred')
        prefix = times[np.argmax(np.array(times).astype(float))] + '_'

    for i in range(n_slices):
        ours.append(
            cv2.imread(slice_dir + prefix + f"pred_{i}.png"))
        gt.append(
            cv2.imread(slice_dir + prefix + f"gt_{i}.png"))
        diff.append(
            cv2.imread(slice_dir + prefix + f"diff_{i}.png"))
        ours_chomp.append(
            cv2.imread(slice_dir + prefix + f"pred_chomp_{i}.png"))
        gt_chomp.append(
            cv2.imread(slice_dir + prefix + f"gt_chomp_{i}.png"))
        diff_chomp.append(
            cv2.imread(slice_dir + prefix + f"diff_chomp_{i}.png"))

    # assemble image
    if dataset_format == "ScanNet":
        gt = [np.flip(x, axis=1) for x in gt]
        ours = [np.flip(x, axis=1) for x in ours]
        diff = [np.flip(x, axis=1) for x in diff]
        gt_chomp = [np.flip(x, axis=1) for x in gt_chomp]
        ours_chomp = [np.flip(x, axis=1) for x in ours_chomp]
        diff_chomp = [np.flip(x, axis=1) for x in diff_chomp]

    gt = np.hstack((gt))
    ours = np.hstack((ours))
    diff = np.hstack((diff))

    gt_chomp = np.hstack((gt_chomp))
    ours_chomp = np.hstack((ours_chomp))
    diff_chomp = np.hstack((diff_chomp))

    return gt, ours, diff, gt_chomp, ours_chomp, diff_chomp


def slices_pc(
    gt_mesh, n_slices, dataset_format,
    grid_dim=256, z_ixs=None
):
    if dataset_format == "ScanNet":
        up = np.array([0., 0., 1.])
    elif dataset_format == "replicaCAD":
        up = np.array([0., 1., 0.])

    inv_bounds_transform, extents = trimesh.bounds.oriented_bounds(gt_mesh)
    bounds_transform = np.linalg.inv(inv_bounds_transform)
    scene_scale = extents / (2 * 0.9)
    grid_pc = transform.make_3D_grid(
        occ_range=[-1.0, 1.0],
        dim=grid_dim,
        device="cpu",
        transform=torch.FloatTensor(bounds_transform),
        scale=torch.FloatTensor(scene_scale),
    )
    grid_pc = grid_pc.view(-1, 3)

    if z_ixs is None:
        z_ixs = torch.linspace(30, grid_dim - 30, n_slices)
        z_ixs = torch.round(z_ixs).long()

    up_ix = np.argmax(np.abs(np.matmul(up, bounds_transform[:3, :3])))
    grid_up = bounds_transform[:3, up_ix]
    up_aligned = np.dot(grid_up, up) > 0

    pc = grid_pc.reshape(grid_dim, grid_dim, grid_dim, 3)
    pc = torch.index_select(pc, up_ix, z_ixs)

    if not up_aligned:
        print('Not up aligned, flipping')
        indices = np.arange(len(z_ixs))[::-1]
        indices = torch.FloatTensor(indices.copy()).int()
        pc = torch.index_select(pc, up_ix, indices)

    scales = np.concatenate([scene_scale[:up_ix], scene_scale[up_ix + 1:]])
    im_size = 256 * scales / scales.min()
    im_size = im_size.astype(int)

    return pc, im_size, up_ix, up


def get_slices(
    sdf_fn, gt_sdf_fn, n_slices, mesh_gt, dataset_format,
    prob_fn=None
):
    pc, im_size, up_ix, up = slices_pc(mesh_gt, n_slices, dataset_format)

    grid_shape = pc.shape[:-1]

    pred_sdf = sdf_util.eval_sdf_interp(
        sdf_fn, pc, handle_oob='fill', oob_val=0.)
    if prob_fn is not None:
        prob = sdf_util.eval_sdf_interp(
            prob_fn, pc, handle_oob='fill', oob_val=0.)
        unmapped = prob == 0
        pred_sdf[unmapped] = np.nan
    cmap = sdf_util.get_colormap()
    col = cmap.to_rgba(pred_sdf.flatten(), alpha=1., bytes=False)
    col = col.reshape(*grid_shape, 4)
    col = (col * 255).astype(np.uint8)[..., :3]

    chomp = metrics.chomp_cost(pred_sdf, epsilon=2.)
    col_chomp = imgviz.depth2rgb(
        chomp.reshape(256, -1), min_value=0., max_value=1.5)
    col_chomp = col_chomp.reshape(*grid_shape, 3)

    gt_sdf = sdf_util.eval_sdf_interp(
        gt_sdf_fn, pc, handle_oob='fill', oob_val=0.)
    gt_sdf = gt_sdf.reshape(*grid_shape)
    diff = np.abs(gt_sdf - pred_sdf)
    diff = diff.reshape(256, -1)
    diff_viz = imgviz.depth2rgb(diff, min_value=0., max_value=0.5)
    diff_viz = diff_viz.reshape(*grid_shape, 3)

    col = [
        cv2.resize(np.take(col, i, up_ix), im_size[::-1])
        for i in range(n_slices)
    ]
    chomp = [
        cv2.resize(np.take(col_chomp, i, up_ix), im_size[::-1])
        for i in range(n_slices)
    ]
    diff = [
        cv2.resize(np.take(diff_viz, i, up_ix), im_size[::-1])
        for i in range(n_slices)
    ]

    if dataset_format == "ScanNet":
        col = [np.flip(x, axis=1) for x in col]
        chomp = [np.flip(x, axis=1) for x in chomp]
        diff = [np.flip(x, axis=1) for x in diff]

    if im_size[0] > 256:
        col = [c.transpose(1, 0, 2) for c in col]
        chomp = [c.transpose(1, 0, 2) for c in chomp]
        diff = [c.transpose(1, 0, 2) for c in diff]

    sdf = np.hstack((col))
    chomp = np.hstack((chomp))
    diff = np.hstack((diff))

    sdf[sdf.sum(-1) == 0] = 255
    chomp[chomp.sum(-1) == 0] = 255

    sdf = sdf[..., ::-1]
    chomp = chomp[..., ::-1]
    diff = diff[..., ::-1]

    # grad, valid_grad = eval_pts.eval_grad(
    #     sdf_fn, pc.reshape(-1, 3), 0.055, False)
    # grad = grad.reshape(pc.shape)
    # valid_grad = valid_grad.reshape(pc.shape[:-1]).astype(np.uint8) * 255
    # valid_grad_vis = [cv2.resize(np.take(valid_grad, i, up_ix), im_size[::-1]) for i in range(n_slices)]
    # valid_grad_vis = np.hstack((valid_grad_vis))
    # valid_grad_vis = valid_grad_vis[..., None].repeat(3, 2)
    # vis = np.vstack((sdf, valid_grad_vis))
    # cv2.imshow('s', vis); cv2.waitKey(0)

    ixs = [0, 1, 2]
    ixs.remove(up_ix)
    z_vals = np.take(np.take(pc, 0, ixs[-1]), 0, ixs[-2])
    z_vals = (z_vals * np.array([0, 1, 0])).sum(-1).numpy()

    return sdf, chomp, diff, z_vals


def compute_voxblox_slices(
    exp_dir, n_slices, dataset_format, mesh_gt, gt_sdf_interp, eval_t=None
):
    sdf_interp = plot_utils.get_voxblox_sdf_interp(
        exp_dir, mesh_gt, eval_t=eval_t)
    return get_slices(
        sdf_interp, gt_sdf_interp, n_slices, mesh_gt, dataset_format
    )


def compute_gpu_fusion_slices(
    exp_dir, n_slices, dataset_format, mesh, gt_sdf_interp, eval_t=None
):
    sdf_fn = plot_utils.get_gpuf_sdf_interp(exp_dir, eval_t=eval_t)
    prob_fn = plot_utils.get_gpuf_prob_interp(exp_dir)
    return get_slices(
        sdf_fn, gt_sdf_interp, n_slices, mesh, dataset_format,
        prob_fn=prob_fn
    )


def mask_scannet_visible(mesh_gt, params):
    n_slices = 6

    pc, im_size, up_ix, up = slices_pc(mesh_gt, n_slices, "ScanNet")

    traj = np.loadtxt(params['seq'] + "/traj.txt")

    depth_dir = os.path.join(params['scannet_dir'], "frames", "depth/")
    depth_batch = []
    for j in range(len(traj)):
        depth_file = depth_dir + str(j) + ".png"
        depth = cv2.imread(depth_file, -1)
        depth = depth.astype(float) / 1000.0
        depth_batch.append(depth)

    depth_batch = np.array(depth_batch)

    H = 480
    W = 640
    intrinsic_file = params['scannet_dir'] + \
        "/frames/intrinsic/intrinsic_depth.txt"
    K = np.loadtxt(intrinsic_file)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = pc.reshape(-1, 3).to(device)
    traj = torch.FloatTensor(traj).reshape(-1, 4, 4).to(device)
    depth_batch = torch.FloatTensor(depth_batch).to(device)

    bsize = 10000
    batches = int(np.ceil(pc.shape[0] / bsize))

    with torch.no_grad():
        visible = []
        for b in range(batches):
            pts_batch = pc[b * bsize: (b + 1) * bsize]

            visible_b = frustum.is_visible_torch(
                pts_batch, traj, depth_batch,
                H, W, fx, fy, cx, cy,
                trunc=0., use_projection=True)
            visible_b = visible_b.sum(dim=0) > 0
            visible.append(visible_b)
    visible = torch.cat(visible)
    visible = visible.reshape(n_slices, 256, 256).cpu().numpy()

    visible = [
        cv2.resize(np.take(visible, j, up_ix).astype(float), im_size[::-1])
        for j in range(n_slices)
    ]
    visible = [np.flip(x, axis=1) for x in visible]
    visible = np.array(visible)

    if im_size[0] > 256:
        visible = np.array([c.transpose(1, 0) for c in visible])

    visible = np.hstack((visible))
    return visible


def sdf_panel(
    dataset_format, mesh, gt_sdf_interp,
    ours_dir, voxblox_dir=None, gpuf_dir=None,
    eval_t=None, n_slices=6,
    visible_mask=None, last=False,
):
    t1 = time.time()
    gt, ours, diff, gt_chomp, ours_chomp, diff_chomp = load_ours(
        ours_dir, n_slices, dataset_format, eval_t=eval_t)
    ours_time = time.time() - t1

    if voxblox_dir is not None:
        t1 = time.time()
        vx_sdf, vx_chomp, vx_diff, z_vals = compute_voxblox_slices(
            voxblox_dir, n_slices, dataset_format, mesh, gt_sdf_interp,
            eval_t=eval_t)
        vox_time = time.time() - t1

    if gpuf_dir is not None:
        t1 = time.time()
        gf_sdf, gf_chomp, gf_diff, z_vals = compute_gpu_fusion_slices(
            gpuf_dir, n_slices, dataset_format, mesh, gt_sdf_interp,
            eval_t=eval_t)
        gpuf_time = time.time() - t1

    print("Times:", ours_time, vox_time, gpuf_time)

    # shade non visible region
    if visible_mask is not None:
        gt[visible_mask == 0] = [150, 150, 150]

    sdf_rows = [gt, ours]

    if last and dataset_format == "ScanNet" and voxblox_dir is not None:
        vx_sdf[visible_mask == 0] = [150, 150, 150]
        gf_sdf[visible_mask == 0] = [150, 150, 150]
        sdf_rows.append(vx_sdf)
    elif voxblox_dir is not None:
        ixs = diff.sum(axis=-1) == 255 * 3
        vx_sdf[ixs] = [150, 150, 150]
        gf_sdf[ixs] = [150, 150, 150]
        sdf_rows.append(vx_sdf)

    if gpuf_dir is not None:
        sdf_rows.append(gf_sdf)

    vis_sdf = np.vstack((sdf_rows))
    vis_sdf = vis_sdf.astype(np.uint8)
    return vis_sdf, z_vals


def all_seqs_panel(
    seqs, dataset_formats,
    ours_root, voxblox_root, gpuf_root, seq_ixs=None
):
    n_slices = 6
    seq_cols = []

    slice_ixs = [1, 1, 1, 2, 2, 1, 2, 2, 0, 3, 3, 2]

    for i in range(len(seqs)):
        if seq_ixs is not None:
            if i not in seq_ixs:
                continue

        dataset_format = dataset_formats[i]
        seq = seqs[i]
        slice_ix = slice_ixs[i]

        print("Saving slices for", seq)

        ours_dir = ours_root + seq + "_0/"
        voxblox_dir = voxblox_root + seq + "/"
        gpuf_dir = gpuf_root + seq + "/"

        with open(voxblox_dir + "/params.json", 'r') as f:
            params = json.load(f)
        mesh_file = os.path.dirname(params["gt_sdf_dir"][:-1]) + "/mesh.obj"
        mesh = trimesh.exchange.load.load(mesh_file, process=False)

        gt_sdf_interp, sdf_dims, sdf_transform = plot_utils.load_gt_sdf(
            params['gt_sdf_dir'])

        print("loaded mesh and gt sdf")

        gt, ours, diff, _, _, _ = load_ours(
            ours_dir, n_slices, dataset_format)

        vx_sdf, _, _, _ = compute_voxblox_slices(
            voxblox_dir, n_slices, dataset_format, mesh, gt_sdf_interp)

        gf_sdf, _, _, _ = compute_gpu_fusion_slices(
            gpuf_dir, n_slices, dataset_format, mesh, gt_sdf_interp)

        if dataset_format == "ScanNet":  # shade non visible region
            visible_mask = mask_scannet_visible(mesh, params)
            gt[visible_mask == 0] = [150, 150, 150]
            vx_sdf[visible_mask == 0] = [150, 150, 150]
            gf_sdf[visible_mask == 0] = [150, 150, 150]
        else:
            ixs = diff.sum(axis=-1) == 255 * 3
            vx_sdf[ixs] = [150, 150, 150]
            gf_sdf[ixs] = [150, 150, 150]

        w = gt.shape[1] // n_slices
        gt = gt[:, slice_ix * w: (slice_ix + 1) * w]
        ours = ours[:, slice_ix * w: (slice_ix + 1) * w]
        vx_sdf = vx_sdf[:, slice_ix * w: (slice_ix + 1) * w]
        gf_sdf = gf_sdf[:, slice_ix * w: (slice_ix + 1) * w]

        pad_shape = np.array(gt.shape)
        pad_shape[0] = 13
        pad = np.full(pad_shape, 255).astype(np.uint8)

        seq_vis = np.vstack((gt, pad, ours, pad, vx_sdf, pad, gf_sdf))
        seq_cols.append(seq_vis)

    pad_shape = np.array(seq_cols[0].shape)
    pad_shape[1] = 13
    pad = np.full(pad_shape, 255).astype(np.uint8)
    seq_cols = [np.hstack((x, pad)) for x in seq_cols[:-1]] + [seq_cols[-1]]

    viz = np.hstack((seq_cols))

    return viz


def save_info(
    save_dir, ours_root, voxblox_root, gpuf_root,
    # z_vals
):
    """ Save useful info about the plot. """

    # with open(save_dir + "/dirs.txt", 'w') as f:
    #     f.write(ours_root + "\n")
    #     f.write(voxblox_root + "\n")
    #     f.write(gpuf_root + "\n")

    # np.savetxt(save_dir + "/z_vals.txt", z_vals)

    fig, ax = plt.subplots(figsize=(20, 1.8))
    cmap = sdf_util.get_colormap()
    plt.colorbar(cmap, ax=ax, orientation='horizontal')
    ax.remove()
    plt.savefig(save_dir + "/sdf_cmap.png", bbox_inches='tight')

    # fig, ax = plt.subplots(figsize=(2, 3))
    # cmap = sdf_util.get_colormap()
    # fig.colorbar(cmap, ax=ax)
    # cbar = fig.colorbar(ax, ticks=[-2, 0, 2])
    # cbar.ax.set_yticklabels(['-2', '0', '2'])
    # ax.remove()
    # plt.savefig(save_dir + "/sdf_cmap_vert.png", bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(2, 3))
    cmap = sdf_util.get_cost_colormap()
    plt.colorbar(cmap, ax=ax)
    ax.remove()
    plt.savefig(save_dir + "/chomp_cmap.png", bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(2, 3))
    cmap = sdf_util.get_cost_colormap(range=[0, 0.5])
    plt.colorbar(cmap, ax=ax)
    ax.remove()
    plt.savefig(save_dir + "/diff_cmap.png", bbox_inches='tight')


# Directories --------------------------------------------

incSDF_root = "/home/joe/projects/incSDF/incSDF/"

save_dir = incSDF_root + "res/figs/slices"

ours_root = incSDF_root + "res/iSDF/slices/"
voxblox_root = incSDF_root + "res/voxblox/gt_traj/0.055/"
gpuf_root = incSDF_root + "res/gpu_fusion/7cm_unocc/"


# Save slices --------------------------------------------

replicaCAD_seqs = ['apt_2_nav', 'apt_2_obj', 'apt_2_mnp',
                   'apt_3_nav', 'apt_3_obj', 'apt_3_mnp']
scanNet_seqs = ['scene0010_00', 'scene0030_00', 'scene0031_00',
                'scene0004_00', 'scene0005_00', 'scene0009_00']
seqs = replicaCAD_seqs + scanNet_seqs

dataset_formats = ["replicaCAD"] * len(replicaCAD_seqs) + \
                  ["ScanNet"] * len(scanNet_seqs)


# # One slice at the end of all sequences ----------------------

viz = all_seqs_panel(
    seqs, dataset_formats,
    ours_root, voxblox_root, gpuf_root,
)
cv2.imshow("slices", viz)
cv2.waitKey(0)

# [apt_3_nav, apt_3_obj, apt_3_mnp, scene30, scene04, scene05]
main_ixs = [3, 4, 5, 7, 9, 10]

# [apt_2_nav, apt_2_obj, apt_2_mnp, scene10, scene31, scene09]
supp_ixs = [0, 1, 2, 6, 8, 11]

# viz = all_seqs_panel(
#     seqs, dataset_formats,
#     ours_root, voxblox_root, gpuf_root,
#     main_ixs,
# )
# cv2.imwrite(f"{save_dir}/slices_main.png", viz)

# viz = all_seqs_panel(
#     seqs, dataset_formats,
#     ours_root, voxblox_root, gpuf_root,
#     supp_ixs,
# )
# cv2.imwrite(f"{save_dir}/slices_supp.png", viz)


# All slices at all times for all seqs ------------------------


# for i in range(2, 3):
#     seq = seqs[i]
#     dataset_format = dataset_formats[i]

#     print("Saving slices for", seq)

#     ours_dir = ours_root + seq + "_0/"
#     voxblox_dir = voxblox_root + seq + "/"
#     gpuf_dir = gpuf_root + seq + "/"

#     # os.makedirs(f"{save_dir}/{seq}")

#     # Get eval times and mesh

#     with open(voxblox_dir + "/res.json", 'r') as f:
#         res = json.load(f)['sdf_eval']
#     if 'bins_lb' in res.keys():
#         res.pop('bins_lb')
#     if 'bins_ub' in res.keys():
#         res.pop('bins_ub')
#     eval_times = [res[k]['time'] for k in res.keys()]
#     eval_times = [float(x) for x in eval_times]
#     eval_times.sort()

#     mesh = None

#     if voxblox_dir is not None:

#         with open(voxblox_dir + "/params.json", 'r') as f:
#             params = json.load(f)
#         mesh_file = os.path.dirname(params["gt_sdf_dir"][:-1]) + "/mesh.obj"
#         mesh = trimesh.exchange.load.load(mesh_file, process=False)

#         visible_mask = None
#         if dataset_formats[i] == "ScanNet":
#             visible_mask = mask_scannet_visible(mesh, params)

#         gt_sdf_interp, sdf_dims, sdf_transform = plot_utils.load_gt_sdf(
#             params['gt_sdf_dir'])

#         print("loaded mesh and gt sdf")

#     for eval_t in eval_times[-2:]:
#         print(eval_t)
#         last = eval_t == eval_times[-1]

#         vis_sdf, z_vals = sdf_panel(
#             dataset_formats[i],
#             mesh,
#             gt_sdf_interp,
#             ours_dir,
#             voxblox_dir=voxblox_dir,
#             gpuf_dir=gpuf_dir,
#             eval_t=eval_t,
#             n_slices=6,
#             visible_mask=visible_mask,
#             last=last,
#         )

#         cv2.imwrite(f"{save_dir}/{seq}/{eval_t:.3f}.png", vis_sdf)


# save_info(save_dir, ours_root, voxblox_root, gpuf_root)
