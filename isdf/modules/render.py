# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from isdf.geometry import transform
from isdf.modules import fc_map


def sdf_render_depth(z_vals, sdf):
    """
    Basic method for rendering depth from SDF using samples along a ray.
    Assumes z_vals are ordered small -> large.
    """
    # assert (z_vals[0].sort()[1].cpu() == torch.arange(len(z_vals[0]))).all()

    n = sdf.size(1)  # n_samples
    inside = sdf < 0
    ixs = torch.arange(n, 0, -1).to(sdf.device)
    mul = inside * ixs
    max_ix = mul.argmax(dim=1)

    arange = torch.arange(z_vals.size(0))
    depths = z_vals[arange, max_ix] + sdf[arange, max_ix]

    # if no zero crossing found
    depths[max_ix == sdf.shape[1] - 1] = 0.

    # print("number of rays without zero crossing found",
    #       (depths == 0.).sum().item(), "out of",
    #       depths.numel())

    return depths


# Compute surface normals in the camera frame
def render_normals(T_WC, render_depth, sdf_map, dirs_C):
    origins, dirs_W = transform.origin_dirs_W(T_WC, dirs_C)
    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)

    pc = origins + (dirs_W * (render_depth.flatten()[:, None]))
    pc.requires_grad_()
    sdf = sdf_map(pc)
    sdf_grad = fc_map.gradient(pc, sdf)

    surface_normals_W = - sdf_grad / \
        (sdf_grad.norm(dim=1, keepdim=True) + 1e-4)
    R_CW = T_WC[:, :3, :3].inverse()
    surface_normals_C = (R_CW * surface_normals_W[..., None, :]).sum(dim=-1)

    surface_normals_C = surface_normals_C.view(
        render_depth.shape[0], render_depth.shape[1], 3)

    return surface_normals_C


def render_weighted(weights, vals, dim=-1, normalise=False):
    """
    General rendering function using weighted sum.
    """
    weighted_vals = weights * vals
    render = weighted_vals.sum(dim=dim)
    if normalise:
        n_samples = weights.size(dim)
        render = render / n_samples

    return render
