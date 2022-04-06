# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


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
