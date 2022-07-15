# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import torch


def start_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
        end = None
    return start, end


def end_timing(start, end):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        end = time.perf_counter()
        elapsed_time = end - start
        # Convert to milliseconds to have the same units
        # as torch.cuda.Event.elapsed_time
        elapsed_time = elapsed_time * 1000
    return elapsed_time


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    rec_points_kd_tree = KDTree(rec_points)
    distances, _ = rec_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def accuracy_comp(mesh_gt, mesh_rec, samples=200000):

    rec_pc = trimesh.sample.sample_surface(mesh_rec, samples)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, samples)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    acc = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    comp = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)

    return acc, comp


def aligned_ate(t1, t2):
    ate = np.linalg.norm(t1 - t2, axis=1)
    ate_rmse = np.sqrt((ate * ate).sum() / len(ate))

    return ate_rmse


def trimesh_sample(mesh_file, pc):
    gt_mesh = trimesh.load(mesh_file)

    # trimesh defines outside mesh as negative and vice versa
    numpy_pc = pc.cpu().detach().numpy()
    # Puts all vertices in a KDtree and finds nearest vertex
    # for each point. Then computes signed dist
    gt_sdf = trimesh.proximity.signed_distance(gt_mesh, numpy_pc)

    return gt_sdf


def chomp_cost(sdf, epsilon=2.0):
    """ CHOMP collision cost.
        equation 21 - https://www.ri.cmu.edu/pub_files/2013/5/CHOMP_IJRR.pdf
        Input is sdf samples along the trajectory to be evaluated.
    """
    cost = - sdf + epsilon / 2.
    cost[sdf > 0] = 1 / (2 * epsilon) * (sdf[sdf > 0] - epsilon)**2
    cost[sdf > epsilon] = 0.

    return cost


def linear_cost(sdf, epsilon=1.5):
    """ Linear collision cost.
    """
    cost = - sdf + epsilon
    cost[sdf > epsilon] = 0.

    return cost


def get_obj_eval_bounds(
    obj_bounds_file, up_ix, expand_m=0.08, expand_down=False
):
    """ Return bounds for box around object used for evaluation. """
    obj_bounds = np.loadtxt(obj_bounds_file)
    obj_bounds = obj_bounds.reshape(-1, 2, 3)

    # box 20cms around and above the object
    obj_bounds[:, 1] += expand_m
    obj_bounds[:, 0, :up_ix] -= expand_m
    obj_bounds[:, 0, up_ix + 1:] -= expand_m
    if expand_down:
        obj_bounds[:, 0, up_ix] -= expand_m

    return obj_bounds


def binned_losses(
    sdf_diff, gt_sdf,
    bin_limits=np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99])
):
    """
        Sort loss into bins at different distances from the surface.
        sdf_diff: np array, absolute difference between predicted and gt sdf
    """
    if isinstance(gt_sdf, torch.Tensor):
        bins_lb = torch.tensor(bin_limits[:-1]).to(gt_sdf.device)
        bins_ub = torch.tensor(bin_limits[1:]).to(gt_sdf.device)
        locical_op = torch.logical_and
    else:
        bins_lb = bin_limits[:-1]
        bins_ub = bin_limits[1:]
        locical_op = np.logical_and

    lb_masks = gt_sdf > bins_lb[:, None]
    ub_masks = gt_sdf < bins_ub[:, None]
    masks = locical_op(lb_masks, ub_masks)

    masked_diffs = sdf_diff * masks
    bins_loss = masked_diffs.sum(1)
    bins_loss = bins_loss / masks.sum(1)

    return bins_loss.tolist()
