# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
import torch
import trimesh
from scipy.interpolate import RegularGridInterpolator

from isdf.datasets import sdf_util
from isdf.modules import embedding, fc_map


def load_model(load_file, gt_mesh, device):

    inv_bounds_transform, extents = trimesh.bounds.oriented_bounds(gt_mesh)

    positional_encoding = embedding.PostionalEncoding(
        min_deg=0,
        max_deg=5,
        scale=0.05937489,
        transform=torch.FloatTensor(inv_bounds_transform).to(device),
    )

    sdf_map = fc_map.SDFMap(
        positional_encoding,
        hidden_size=256,
        hidden_layers_block=2,
        scale_output=0.14,
    ).to(device)

    checkpoint = torch.load(load_file)
    sdf_map.load_state_dict(checkpoint["model_state_dict"])

    return sdf_map


def load_res(exp_name):
    with open(exp_name + '/res.json', 'r') as f:
        res = json.load(f)
    return res


def load_gt_sdf(gt_sdf_dir):
    gt_sdf_file = gt_sdf_dir + "/sdf.npy"
    # stage_sdf_file = gt_sdf_dir + "/stage_sdf.npy"
    sdf_transf_file = gt_sdf_dir + "/transform.txt"

    sdf_grid = np.load(gt_sdf_file)
    sdf_transform = np.loadtxt(sdf_transf_file)
    gt_sdf_interp = sdf_util.sdf_interpolator(
        sdf_grid, sdf_transform)
    sdf_dims = torch.tensor(sdf_grid.shape)

    return gt_sdf_interp, sdf_dims, sdf_transform


def get_seq_time(
    exp,
    data_dir="/home/joe/projects/incSDF/data/ReplicaCAD-seqs/"
):
    config_dir = exp + "/config.json"
    if os.path.exists(config_dir):
        with open(config_dir) as json_file:
            config = json.load(json_file)
        seq_name = config['dataset']['seq_dir'].split('/')[-2]
    else:
        config_dir = exp + "/params.json"
        with open(config_dir) as json_file:
            config = json.load(json_file)
        seq_name = config['seq'].split('/')[-2]

    kf_dir = os.path.join(data_dir, seq_name, "results/")
    n_frames = len(os.listdir(kf_dir)) // 3
    return n_frames / 30


def get_last_eval_t(ours_root, exp):

    conf_file = os.path.join(ours_root, exp, "config.json")
    with open(conf_file, 'r') as f:
        config = json.load(f)

    eval_pts_dir = "/home/joe/projects/incSDF/incSDF/data/eval_pts/vox/"
    if config["model"]["frac_time_perception"] == 1.:
        eval_pts_dir += "0.055/"
    elif config["model"]["frac_time_perception"] == 0.75:
        eval_pts_dir += "0.063/"
    elif config["model"]["frac_time_perception"] == 0.5:
        eval_pts_dir += "0.078/"
    elif config["model"]["frac_time_perception"] == 0.25:
        eval_pts_dir += "0.11/"
    else:
        raise ValueError(
            'Frace perception time not in [0.25, 0.5, 0.75, 1.]')
    eval_pts_dir += [
        x for x in config["dataset"]["seq_dir"].split('/') if x != ""][-1] +\
        "/eval_pts/"
    eval_times = [float(x) for x in os.listdir(eval_pts_dir)]
    eval_times.sort()

    return eval_times[-1]


def get_gpuf_sdf_interp(exp_dir, eval_t=None):
    if eval_t is None:
        sdf_file = "final_sdf.txt"  # Use final SDF saved at the end
    else:
        sdf_file = f"{eval_t:.3f}.txt"

    sdf, transform = sdf_util.read_sdf_gpufusion(
        os.path.join(exp_dir, sdf_file),
        os.path.join(exp_dir, "transform.txt"))
    sdf_interp = sdf_util.sdf_interpolator(sdf, transform)

    return sdf_interp


def get_gpuf_prob_interp(exp_dir):
    sdf, transform = sdf_util.read_sdf_gpufusion(
        os.path.join(exp_dir, "final_sdf.txt"),
        os.path.join(exp_dir, "transform.txt"))
    prob = np.loadtxt(os.path.join(exp_dir, "final_prob.txt"))
    prob = prob.reshape(sdf.shape)
    prob_interp = sdf_util.sdf_interpolator(prob, transform)

    return prob_interp


def get_voxblox_sdf_interp(exp_dir, mesh_gt, eval_t=None):

    with open(exp_dir + "/params.json", 'r') as f:
        params = json.load(f)

    # create axis aligned grid at same resolution
    vsm = params['voxel_size']
    bounds = mesh_gt.bounds.copy()
    start = bounds[0] - bounds[0] % vsm + vsm / 2 - 20 * vsm
    end = bounds[1] + 20 * vsm
    x = np.arange(start[0], end[0], step=vsm)
    y = np.arange(start[1], end[1], step=vsm)
    z = np.arange(start[2], end[2], step=vsm)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid = np.concatenate(
        (xx[..., None], yy[..., None], zz[..., None]), axis=-1)

    with open(exp_dir + "/res.json", 'r') as f:
        res = json.load(f)['sdf_eval']

    if 'bins_lb' in res.keys():
        res.pop('bins_lb')
    if 'bins_ub' in res.keys():
        res.pop('bins_ub')

    eval_times = [res[k]['time'] for k in res.keys()]
    if eval_t is None:
        seq = [x for x in exp_dir.split("/") if x != ""][-1]
        last = -1
        if seq in ["scene0004_00", "apt_2_mnp"]:
            last = -2
        key = list(res.keys())[last]
    else:
        key = list(res.keys())[eval_times.index(eval_t)]
    sdf_file = key + ".npy"

    sdf = np.loadtxt(os.path.join(exp_dir, "out", sdf_file))

    sdf[:, :3] = np.round(sdf[:, :3] * 2 / vsm) * vsm / 2

    grid_sdf = np.concatenate(
        (grid, np.full([*grid.shape[:-1], 1], np.nan)), axis=-1)
    grid_ixs = np.rint((sdf[:, :3] - start) / vsm).astype(int)
    check = np.logical_and(grid_ixs < grid.shape[:-1], grid_ixs >= 0)
    check = check.sum(-1) == 3
    grid_ixs = grid_ixs[check]
    sdf = sdf[check]
    grid_sdf[grid_ixs[:, 0], grid_ixs[:, 1], grid_ixs[:, 2], 3] = sdf[:, 3]

    # # visualise voxblox mapped region
    # vox_bool_grid = ~np.isnan(grid_sdf[..., 3])
    # for z_ix in range(grid.shape[1]):
    #     vox_slice = vox_bool_grid[:, z_ix]
    #     vox_slice = vox_slice[..., None].repeat(3, 2).astype(float)
    #     cv2.imshow("im", vox_slice)
    #     cv2.waitKey(0)

    sdf_interp = RegularGridInterpolator((x, y, z), grid_sdf[..., 3])

    return sdf_interp
