# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import json
import os
import trimesh
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
import scipy
from scipy import ndimage
from scipy.spatial.transform import Rotation as R


# Input / output functions -----------------------------------


def read_sdf_binary(sdf_dir):
    """ Read SDF saved in binary format with params file.
    """
    params_file = os.path.join(sdf_dir, "parameters.json")
    sdf_file = os.path.join(sdf_dir, "volume.sdf")

    with open(params_file, 'r') as f:
        params = json.load(f)

    with open(sdf_file, 'rb') as f:
        sdf = np.fromfile(f, np.float32)

    dims = params['voxelDim'][::-1]

    sdf = sdf.reshape(dims)
    sdf = np.transpose(sdf, (2, 1, 0))

    transform = np.linalg.inv(np.array(params["T_voxel_sdf"]))
    sdf = - sdf  # as outside of room is considered free space
    return sdf, transform


def read_sdf_txt(sdf_dir):
    """ Read SDF and transform files output from Habitat-sim.
    """
    sdf_fname = os.path.join(sdf_dir, "sdf.txt")
    transf_fname = os.path.join(sdf_dir, "transform.txt")

    sdf = []
    with open(sdf_fname, 'r') as f:
        for line in f.readlines():
            sdf.append(float(line))

    with open(transf_fname, 'r') as f:
        dims = [int(e) for e in f.readline().split()[1:]]
        vsm = [float(e) for e in f.readline().split()[1:]]
        offset = [float(e) for e in f.readline().split()[1:]]

    transform = np.eye(4)
    transform[:3, 3] = offset
    transform[np.diag_indices_from(transform[:3, :3])] = vsm

    sdf = np.array(sdf).reshape(dims)
    sdf *= vsm[0]
    sdf = - sdf  # inside room is free space

    return sdf, transform


def read_sdf_gpufusion(sdf_file, transform_file):
    """ Read SDF and transform from output of GPU fusion """
    with open(transform_file, 'r') as f:
        dims = [int(e) for e in f.readline().split()[1:]]
        vsm = [float(e) for e in f.readline().split()[1:]]
        offset = [float(e) for e in f.readline().split()[1:]]

    transform = np.eye(4)
    transform[:3, 3] = offset
    transform[np.diag_indices_from(transform[:3, :3])] = vsm

    sdf = np.loadtxt(sdf_file)
    sdf = sdf.reshape(dims)

    return sdf, transform


# Voxel grid util fns -------------------------------------------


def robotics_2_graphics_coords(mesh):
    rot = R.from_euler('x', -90, degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    mesh.apply_transform(T)
    return mesh


def merge_sdfs(base_sdf, base_transf, merge_sdf, merge_transf):
    """ Merge 2 aligned voxel grids.
        Merges merge_sdf into base_sdf.
        Grids must have the same voxel size and offset.
        If merge_sdf extends outside base_sdf ignore this region.

        Return: updated base_sdf after merging merge_sdf.
    """
    vsm = base_transf[0, 0]
    assert vsm == merge_transf[0, 0], "Voxel sizes are different"

    # Compute indices where grids overlap

    base_start_ix = (merge_transf[:3, 3] - base_transf[:3, 3]) / vsm
    base_end_ix = base_start_ix + merge_sdf.shape

    check = base_start_ix - np.round(base_start_ix)
    assert np.linalg.norm(check) < 1e-5, "Grids are not aligned"

    merge_start_ix = np.maximum(
        np.zeros_like(base_start_ix), -base_start_ix)
    merge_end_ix = base_sdf.shape - base_end_ix
    coords = np.argwhere(merge_end_ix >= 0)
    merge_end_ix[coords] = np.array(merge_sdf.shape)[coords]

    base_end_ix = np.minimum(base_sdf.shape, base_end_ix)
    base_start_ix[base_start_ix < 0] = 0

    base_start_ix = np.round(base_start_ix).astype(int)
    base_end_ix = np.round(base_end_ix).astype(int)
    merge_start_ix = np.round(merge_start_ix).astype(int)
    merge_end_ix = np.round(merge_end_ix).astype(int)

    # Take minimum value to join SDF grids

    merge_sdf_inrange = merge_sdf[
        merge_start_ix[0]:merge_end_ix[0],
        merge_start_ix[1]:merge_end_ix[1],
        merge_start_ix[2]:merge_end_ix[2]]

    base_sdf_inrange = base_sdf[
        base_start_ix[0]:base_end_ix[0],
        base_start_ix[1]:base_end_ix[1],
        base_start_ix[2]:base_end_ix[2]]

    base_sdf[base_start_ix[0]:base_end_ix[0],
             base_start_ix[1]:base_end_ix[1],
             base_start_ix[2]:base_end_ix[2]] = np.minimum(
        base_sdf_inrange, merge_sdf_inrange)

    return base_sdf


def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]

    return x, y, z


def sdf_grid2pc(sdf_grid, transform):
    x, y, z = get_grid_pts(sdf_grid.shape, transform)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    pc = np.concatenate(
        (xx[..., None], yy[..., None], zz[..., None], sdf_grid[..., None]),
        axis=-1)

    return pc


def sdf_interpolator(sdf_grid, transform):
    x, y, z = get_grid_pts(sdf_grid.shape, transform)

    sdf_interp = scipy.interpolate.RegularGridInterpolator(
        (x, y, z), sdf_grid)

    return sdf_interp


def eval_sdf_interp(sdf_interp, pc, handle_oob='except', oob_val=0.):
    """ param:
        handle_oob: dictates what to do with out of bounds points. Must
        take either 'except', 'mask' or 'fill'.
    """

    reshaped = False
    if pc.ndim != 2:
        reshaped = True
        pc_shape = pc.shape[:-1]
        pc = pc.reshape(-1, 3)

    if handle_oob == 'except':
        sdf_interp.bounds_error = True
    elif handle_oob == 'mask':
        dummy_val = 1e99
        sdf_interp.bounds_error = False
        sdf_interp.fill_value = dummy_val
    elif handle_oob == 'fill':
        sdf_interp.bounds_error = False
        sdf_interp.fill_value = oob_val
    else:
        assert True, "handle_oob must take a recognised value."

    sdf = sdf_interp(pc)

    if reshaped:
        sdf = sdf.reshape(pc_shape)

    if handle_oob == 'mask':
        valid_mask = sdf != dummy_val
        return sdf, valid_mask

    return sdf


class SDFTriInterp:
    # Follows https://spie.org/samples/PM159.pdf
    def __init__(self, sdf_grid, transform):
        x, y, z = get_grid_pts(sdf_grid.shape, transform)
        self.vsm = transform[0, 0]
        self.start = transform[:3, 3]
        self.x0 = transform[0, 3]
        self.y0 = transform[1, 3]
        self.z0 = transform[2, 3]

        self.dims = sdf_grid.shape
        self.grid = sdf_grid

    def __call__(self, pts):
        indices = (pts - self.start) // self.vsm
        indices = indices.astype(int)
        assert (indices < self.dims).all(), "Point outside of grid"

        deltas = (pts % self.vsm) / self.vsm

        qvec = np.array([
            np.ones(len(pts)),
            deltas[:, 0],
            deltas[:, 1],
            deltas[:, 2],
            deltas[:, 0] * deltas[:, 1],
            deltas[:, 1] * deltas[:, 2],
            deltas[:, 2] * deltas[:, 0],
            deltas[:, 0] * deltas[:, 1] * deltas[:, 2],
        ])

        xind, yind, zind = indices[:, 0], indices[:, 1], indices[:, 2]

        p000 = self.grid[xind, yind, zind]
        p100 = self.grid[xind + 1, yind, zind]
        p010 = self.grid[xind, yind + 1, zind]
        p001 = self.grid[xind, yind, zind + 1]
        p101 = self.grid[xind + 1, yind, zind + 1]
        p110 = self.grid[xind + 1, yind + 1, zind]
        p011 = self.grid[xind, yind + 1, zind + 1]
        p111 = self.grid[xind + 1, yind + 1, zind + 1]

        c0 = p000
        c1 = p100 - p000
        c2 = p010 - p000
        c3 = p001 - p000
        c4 = p110 - p010 - p100 + p000
        c5 = p011 - p001 - p010 + p000
        c6 = p101 - p001 - p100 + p000
        c7 = p111 - p011 - p101 - p110 + p100 + p001 + p010 - p000

        C = np.vstack((c0, c1, c2, c3, c4, c5, c6, c7))

        vals = (qvec * C).sum(axis=0)
        return vals


def get_colormap(sdf_range=[-2, 2], surface_cutoff=0.01):
    white = np.array([1., 1., 1., 1.])
    sdf_range[1] += surface_cutoff - (sdf_range[1] % surface_cutoff)
    sdf_range[0] -= surface_cutoff - (-sdf_range[0] % surface_cutoff)

    positive_n_cols = int(sdf_range[1] / surface_cutoff)
    viridis = cm.get_cmap('viridis', positive_n_cols)
    positive_colors = viridis(np.linspace(0.2, 1, int(positive_n_cols)))
    positive_colors[0] = white

    negative_n_cols = int(-sdf_range[0] / surface_cutoff)
    redpurple = cm.get_cmap('RdPu', negative_n_cols).reversed()
    negative_colors = redpurple(np.linspace(0., 0.7, negative_n_cols))
    negative_colors[-1] = white

    colors = np.concatenate(
        (negative_colors, white[None, :], positive_colors), axis=0)
    sdf_cmap = ListedColormap(colors)

    norm = mpl.colors.Normalize(sdf_range[0], sdf_range[1])
    sdf_cmap_fn = cm.ScalarMappable(norm=norm, cmap=sdf_cmap)
    # plt.colorbar(sdf_cmap_fn)
    # plt.show()
    return sdf_cmap_fn


def get_cost_colormap(range=[0, 1.5]):
    norm = mpl.colors.Normalize(range[0], range[1])
    cmap_fn = cm.ScalarMappable(norm=norm, cmap="jet")

    return cmap_fn


# Mesh to SDF functions -------------------------------------------


def voxelize_subdivide(mesh,
                       pitch,
                       origin_voxel=np.zeros(3),
                       max_iter=10,
                       edge_factor=2.0):
    """
    Adapted from trimesh function allow for shifts in the origin
    of the SDF grid. i.e. there doesn't need to be a voxel with
    centere at [0, 0, 0].

    Voxelize a surface by subdividing a mesh until every edge is
    shorter than: (pitch / edge_factor)
    Parameters
    -----------
    mesh:        Trimesh object
    pitch:       float, side length of a single voxel cube
    max_iter:    int, cap maximum subdivisions or None for no limit.
    edge_factor: float,
    Returns
    -----------
    VoxelGrid instance representing the voxelized mesh.
    """
    max_edge = pitch / edge_factor

    if max_iter is None:
        longest_edge = np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]],
            axis=1).max()
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)

    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    v, f = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=max_iter)

    # convert the vertices to their voxel grid position
    hit = (v - origin_voxel) / pitch

    # Provided edge_factor > 1 and max_iter is large enough, this is
    # sufficient to preserve 6-connectivity at the level of voxels.
    hit = np.round(hit).astype(int)

    # remove duplicates
    unique, inverse = trimesh.grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = hit[unique]

    origin_index = occupied_index.min(axis=0)
    origin_position = origin_voxel + origin_index * pitch

    return trimesh.voxel.base.VoxelGrid(
        trimesh.voxel.encoding.SparseBinaryEncoding(
            occupied_index - origin_index),
        transform=trimesh.transformations.scale_and_translate(
            scale=pitch, translate=origin_position)
    )


def sdf_from_occupancy(occ_map, voxel_size):
    # Convert occupancy field to sdf field
    inv_occ_map = 1 - occ_map

    # Get signed distance from occupancy map and inv map
    map_dist = ndimage.distance_transform_edt(inv_occ_map)
    inv_map_dist = ndimage.distance_transform_edt(occ_map)

    sdf = map_dist - inv_map_dist

    # metric units
    sdf = sdf.astype(float)
    sdf = sdf * voxel_size

    return sdf


def sdf_from_mesh(mesh, voxel_size, extend_factor=0.15,
                  origin_voxel=np.zeros(3)):
    # Convert mesh to occupancy field
    voxels = voxelize_subdivide(
        mesh, voxel_size, origin_voxel=origin_voxel)
    voxels = voxels.fill()
    occ_map = voxels.matrix
    transform = voxels.transform

    # Extend voxel grid around object
    extend = np.array(occ_map.shape) * extend_factor
    extend = np.repeat(extend, 2).reshape(3, 2)
    extend = np.round(extend).astype(int)
    occ_map = np.pad(occ_map, extend)
    transform[:3, 3] -= extend[:, 0] * voxel_size

    sdf = sdf_from_occupancy(occ_map, voxel_size)

    return sdf, transform


def sdf_from_mesh_gridgiven(mesh, transform, dims):
    """ Compute SDF from the mesh.
        Output SDF is in a voxel grid specified by transform and dims.
    """
    # Convert mesh to occupancy field
    voxel_size = transform[0, 0]
    origin_voxel = transform[:3, 3] % transform[0, 0]
    voxels = voxelize_subdivide(mesh, voxel_size, origin_voxel=origin_voxel)
    voxels = voxels.fill()
    occ_map = voxels.matrix
    occ_transform = voxels.transform

    # Place object occupancy map in full empty voxel grid
    base_grid = np.zeros(dims).astype(bool)

    base_start_ix = (occ_transform[:3, 3] - transform[:3, 3]) / voxel_size
    base_end_ix = base_start_ix + occ_map.shape

    check = base_start_ix - np.round(base_start_ix)
    assert np.linalg.norm(check) < 1e-5, "Grids are not aligned"

    occ_start_ix = np.maximum(
        np.zeros_like(base_start_ix), -base_start_ix)
    occ_end_ix = base_grid.shape - base_end_ix
    coords = np.argwhere(occ_end_ix >= 0)
    occ_end_ix[coords] = np.array(occ_map.shape)[coords]

    base_end_ix = np.minimum(base_grid.shape, base_end_ix)
    base_start_ix[base_start_ix < 0] = 0

    base_start_ix = np.round(base_start_ix).astype(int)
    base_end_ix = np.round(base_end_ix).astype(int)
    occ_start_ix = np.round(occ_start_ix).astype(int)
    occ_end_ix = np.round(occ_end_ix).astype(int)

    # If object is fully within the specified grid, then full
    # occ_map will be inrange.
    occ_inrange = occ_map[
        occ_start_ix[0]:occ_end_ix[0],
        occ_start_ix[1]:occ_end_ix[1],
        occ_start_ix[2]:occ_end_ix[2]]

    base_grid[base_start_ix[0]:base_end_ix[0],
              base_start_ix[1]:base_end_ix[1],
              base_start_ix[2]:base_end_ix[2]] = occ_inrange

    sdf = sdf_from_occupancy(base_grid, voxel_size)

    return sdf, transform


# def test_mesh_to_sdf(mesh_path):
#     """ Test mesh_to_sdf library.
#         https://github.com/marian42/mesh_to_sdf
#     """
#     from isdf.visualisation import sdf_viewer
#     from mesh_to_sdf import mesh_to_voxels

#     mesh = trimesh.load(mesh_path)

#     d = 64
#     voxels = mesh_to_voxels(
#         mesh, d, pad=False, check_result=False,
#         sign_method="depth")

#     grid2world = np.array(
#         [[2. / d, 0., 0., -1.],
#          [0., 2. / d, 0., -1.],
#          [0., 0., 2. / d, -1.],
#          [0., 0., 0., 1.]])

#     vertices = mesh.vertices - mesh.bounding_box.centroid
#     vertices *= 2 / np.max(mesh.bounding_box.extents)
#     scaled_mesh = mesh.copy()
#     scaled_mesh.vertices = vertices

#     sdf_viewer.SDFViewer(
#         mesh=scaled_mesh, sdf_grid=voxels,
#         grid2world=grid2world, colormap=True)
