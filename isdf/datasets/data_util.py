# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import trimesh


class FrameData:
    def __init__(
        self,
        frame_id=None,
        im_batch=None,
        im_batch_np=None,
        depth_batch=None,
        depth_batch_np=None,
        T_WC_batch=None,
        T_WC_batch_np=None,
        normal_batch=None,
        frame_avg_losses=None,
        T_WC_track=None,
        T_WC_gt=None,
    ):
        super(FrameData, self).__init__()

        self.frame_id = frame_id
        self.im_batch = im_batch
        self.im_batch_np = im_batch_np
        self.depth_batch = depth_batch
        self.depth_batch_np = depth_batch_np
        self.T_WC_batch = T_WC_batch
        self.T_WC_batch_np = T_WC_batch_np

        self.normal_batch = normal_batch
        self.frame_avg_losses = frame_avg_losses

        # for pose refinement
        self.T_WC_track = T_WC_track
        self.T_WC_gt = T_WC_gt

        self.count = 0 if frame_id is None else len(frame_id)

    def add_frame_data(self, data, replace):
        """
        Add new FrameData to existing FrameData.
        """
        self.frame_id = expand_data(
            self.frame_id, data.frame_id, replace)

        self.im_batch = expand_data(
            self.im_batch, data.im_batch, replace)
        self.im_batch_np = expand_data(
            self.im_batch_np, data.im_batch_np, replace)

        self.depth_batch = expand_data(
            self.depth_batch, data.depth_batch, replace)
        self.depth_batch_np = expand_data(
            self.depth_batch_np, data.depth_batch_np, replace)

        self.T_WC_batch = expand_data(
            self.T_WC_batch, data.T_WC_batch, replace)
        self.T_WC_batch_np = expand_data(
            self.T_WC_batch_np, data.T_WC_batch_np, replace)

        self.normal_batch = expand_data(
            self.normal_batch, data.normal_batch, replace)

        device = data.im_batch.device
        empty_dist = torch.zeros(
            [data.im_batch.shape[0]], device=device)
        self.frame_avg_losses = expand_data(
            self.frame_avg_losses, empty_dist, replace)

        if data.T_WC_gt is not None:
            self.T_WC_gt = expand_data(
                self.T_WC_gt, data.T_WC_gt, replace)

    def __len__(self):
        return 0 if self.frame_id is None else len(self.frame_id)


def expand_data(batch, data, replace=False):
    """
    Add new FrameData attribute to exisiting FrameData attribute.
    Either concatenate or replace last row in batch.
    """
    cat_fn = np.concatenate
    if torch.is_tensor(data):
        cat_fn = torch.cat

    if batch is None:
        batch = data

    else:
        if replace is False:
            batch = cat_fn((batch, data))
        else:
            batch[-1] = data[0]

    return batch


def scene_properties(mesh_path):

    scene_mesh = trimesh.exchange.load.load(mesh_path, process=False)
    T_extent_to_scene, bound_scene_extents = trimesh.bounds.oriented_bounds(
        scene_mesh)
    T_extent_to_scene = np.linalg.inv(T_extent_to_scene)

    scene_center = scene_mesh.bounds.mean(axis=0)

    return T_extent_to_scene, bound_scene_extents, scene_center


def save_trajectory(traj, file_name, format="replica", timestamps=None):
    traj_file = open(file_name, "w")

    if format == "replica":
        for idx, T_WC in enumerate(traj):
            time = timestamps[idx]
            traj_file.write('{} '.format(time))
            np.savetxt(traj_file, T_WC[:3, :].reshape([1, 12]), fmt="%f")
    elif format == "TUM":
        for idx, T_WC in enumerate(traj):
            quat = trimesh.transformations.quaternion_from_matrix(T_WC[:3, :3])
            quat = np.roll(quat, -1)
            trans = T_WC[:3, 3]
            time = timestamps[idx]

            traj_file.write('{} '.format(time))
            np.savetxt(traj_file, trans.reshape([1, 3]), fmt="%f", newline=" ")
            np.savetxt(traj_file, quat.reshape([1, 4]), fmt="%f",)

    traj_file.close()
