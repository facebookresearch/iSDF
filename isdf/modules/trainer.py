# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pylab as plt
import torch

import torch.optim as optim
from torchvision import transforms
import trimesh
import imgviz
import json
import cv2
import copy
import os
from scipy import ndimage
from scipy.spatial import KDTree

from isdf.datasets import (
    dataset, image_transforms, sdf_util, data_util
)
from isdf.datasets.data_util import FrameData
from isdf.modules import (
    fc_map, embedding, render, sample, loss
)
from isdf import geometry, visualisation
from isdf.eval import metrics, eval_pts
from isdf.visualisation import draw, draw3D
from isdf.eval.metrics import start_timing, end_timing


class Trainer():
    def __init__(
        self,
        device,
        config_file,
        chkpt_load_file=None,
        incremental=True,
        grid_dim=200,
    ):
        super(Trainer, self).__init__()

        self.device = device
        self.incremental = incremental
        self.tot_step_time = 0.
        self.last_is_keyframe = False
        self.steps_since_frame = 0
        self.optim_frames = 0

        self.gt_depth_vis = None
        self.gt_im_vis = None

        # eval params
        self.gt_sdf_interp = None
        self.stage_sdf_interp = None
        self.sdf_dims = None
        self.sdf_transform = None

        self.grid_dim = grid_dim
        self.new_grid_dim = None
        self.chunk_size = 100000

        with open(config_file) as json_file:
            self.config = json.load(json_file)

        self.frames = FrameData()  # keyframes

        self.set_params()
        self.set_cam()
        self.set_directions()
        self.load_data()

        # scene params for visualisation
        self.scene_center = None
        self.inv_bounds_transform = None
        self.active_idxs = None
        self.active_pixels = None
        if self.gt_scene:
            scene_mesh = trimesh.exchange.load.load(
                self.scene_file, process=False)
            self.set_scene_properties(scene_mesh)
        if self.dataset_format == "realsense_franka_offline":
            self.set_scene_properties()

        self.load_networks()
        if chkpt_load_file is not None:
            self.load_checkpoint(chkpt_load_file)
        self.sdf_map.train()

        # for evaluation
        if self.do_eval:
            if self.sdf_transf_file is not None:
                self.load_gt_sdf()
        self.cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # Init functions ---------------------------------------

    def get_latest_frame_id(self):
        return int(self.tot_step_time * self.fps)

    def set_scene_properties(self, scene_mesh = None):
        # if self.live:
        #     # bounds is axis algined, camera initial pose defines origin
        #     ax_aligned_box = scene_mesh.bounding_box
        #     T_extent_to_scene, bounds_extents = \
        #         trimesh.bounds.oriented_bounds(ax_aligned_box)
        # else:
        # bounds_transform is a transformation matrix which moves
        # the center of the bounding box of the mesh to the origin.
        # bounds_extents is the extents of the mesh once transformed
        # with bounds_transform
        if "realsense_franka" in self.dataset_format:
            # define our own workspace bounds 
            T_extent_to_scene = trimesh.transformations.rotation_matrix(np.deg2rad(self.config["workspace"]["rotate_z"]), [0, 0, 1]) # flip workspace
            T_extent_to_scene[:3, 3] = np.array(self.config["workspace"]["offset"])
            bounds_extents = np.array(self.config["workspace"]["extents"])
            self.scene_center = np.array(self.config["workspace"]["center"])
        else:
            T_extent_to_scene, bounds_extents = \
                trimesh.bounds.oriented_bounds(scene_mesh)
            self.scene_center = scene_mesh.bounds.mean(axis=0)

        self.inv_bounds_transform = torch.from_numpy(
            T_extent_to_scene).float().to(self.device)
        self.bounds_transform_np = np.linalg.inv(T_extent_to_scene)
        self.bounds_transform = torch.from_numpy(
            self.bounds_transform_np).float().to(self.device)

        # Need to divide by range_dist as it will scale the grid which
        # is created in range = [-1, 1]
        # Also divide by 0.9 so extents are a bit larger than gt mesh
        grid_range = [-1.0, 1.0]
        range_dist = grid_range[1] - grid_range[0]
        self.scene_scale_np = bounds_extents / (range_dist * 0.9)
        self.scene_scale = torch.from_numpy(
            self.scene_scale_np).float().to(self.device)
        self.inv_scene_scale = 1. / self.scene_scale

        self.grid_pc = geometry.transform.make_3D_grid(
            grid_range,
            self.grid_dim,
            self.device,
            transform=self.bounds_transform,
            scale=self.scene_scale,
        )
        self.grid_pc = self.grid_pc.view(-1, 3).to(self.device)

        self.up_ix = np.argmax(np.abs(np.matmul(
            self.up, self.bounds_transform_np[:3, :3])))
        self.grid_up = self.bounds_transform_np[:3, self.up_ix]
        self.up_aligned = np.dot(self.grid_up, self.up) > 0

        self.crop_dist = 0.1 if "franka" in self.dataset_format else 0.25

    def set_params(self):
        # Dataset
        # require dataset format, depth scale and camera params
        self.dataset_format = self.config["dataset"]["format"]
        self.live = False
        if self.dataset_format in ["arkit", "realsense", "realsense_franka"]:
            self.live = True
        if "realsense_franka" in self.dataset_format:
            self.ext_calib = self.config["ext_calib"]
        else: 
            self.ext_calib = None
        self.inv_depth_scale = 1. / self.config["dataset"]["depth_scale"]
        self.distortion_coeffs = []
        if self.dataset_format == "ScanNet":
            self.set_scannet_cam_params(
                self.config["dataset"]["intrinsics_file"])
        else:
            self.fx = self.config["dataset"]["camera"]["fx"]
            self.fy = self.config["dataset"]["camera"]["fy"]
            self.cx = self.config["dataset"]["camera"]["cx"]
            self.cy = self.config["dataset"]["camera"]["cy"]
            self.H = self.config["dataset"]["camera"]["h"]
            self.W = self.config["dataset"]["camera"]["w"]
            if "k1" in self.config["dataset"]["camera"]:
                self.distortion_coeffs.append(self.config["dataset"]["camera"]["k1"])
            if "k2" in self.config["dataset"]["camera"]:
                self.distortion_coeffs.append(self.config["dataset"]["camera"]["k2"])
            if "p1" in self.config["dataset"]["camera"]:
                self.distortion_coeffs.append(self.config["dataset"]["camera"]["p1"])
            if "p2" in self.config["dataset"]["camera"]:
                self.distortion_coeffs.append(self.config["dataset"]["camera"]["p2"])
            if "k3" in self.config["dataset"]["camera"]:
                self.distortion_coeffs.append(self.config["dataset"]["camera"]["k3"])

        self.gt_scene = False
        self.fps = 30  # this can be set to anything when in live mode
        if not self.live:
            self.seq_dir = self.config["dataset"]["seq_dir"]
            self.seq = [x for x in self.seq_dir.split('/') if x != ''][-1]
            self.ims_file = self.seq_dir
            if self.dataset_format != "realsense_franka_offline":
                self.ims_file = os.path.join(self.ims_file, "results")
            self.fps = self.config["dataset"]["fps"]

            self.obj_bounds_file = None

            if os.path.exists(self.seq_dir + "/obj_bounds.txt"):
                self.obj_bounds_file = self.seq_dir + "/obj_bounds.txt"
            if "gt_sdf_dir" in self.config["dataset"]:
                gt_sdf_dir = self.config["dataset"]["gt_sdf_dir"]
                self.scene_file = gt_sdf_dir + "mesh.obj"
                self.gt_sdf_file = gt_sdf_dir + "/1cm/sdf.npy"
                self.stage_sdf_file = gt_sdf_dir + "/1cm/stage_sdf.npy"
                self.sdf_transf_file = gt_sdf_dir + "/1cm/transform.txt"
                self.gt_scene = True
            self.scannet_dir = None
            if self.dataset_format == "ScanNet":
                self.scannet_dir = self.config["dataset"]["scannet_dir"]
            if "im_indices" in self.config["dataset"]:
                self.indices = self.config["dataset"]["im_indices"]
            self.noisy_depth = False
            if "noisy_depth" in self.config["dataset"]:
                self.noisy_depth = bool(self.config["dataset"]["noisy_depth"])
            self.traj_file = self.seq_dir + "/traj.txt"
            # assert os.path.exists(self.traj_file)

        self.gt_traj = None

        self.n_steps = self.config["trainer"]["steps"]

        # Model
        self.do_active = bool(self.config["model"]["do_active"])
        # scaling applied to network output before interpreting value as sdf
        self.scale_output = self.config["model"]["scale_output"]
        # noise applied to network output
        self.noise_std = self.config["model"]["noise_std"]
        self.noise_kf = self.config["model"]["noise_kf"]
        self.noise_frame = self.config["model"]["noise_frame"]
        # sliding window size for optimising latest frames
        self.window_size = self.config["model"]["window_size"]
        self.hidden_layers_block = self.config["model"]["hidden_layers_block"]
        self.hidden_feature_size = self.config["model"]["hidden_feature_size"]
        # multiplier for time spent doing training vs time elapsed
        # to simulate scenarios with e.g. 50% perception time, 50% planning
        self.frac_time_perception = \
            self.config["model"]["frac_time_perception"]
        # optimisation steps per kf
        self.iters_per_kf = self.config["model"]["iters_per_kf"]
        self.iters_per_frame = self.config["model"]["iters_per_frame"]
        # thresholds for adding frame to keyframe set
        self.kf_dist_th = self.config["model"]["kf_dist_th"]
        self.kf_pixel_ratio = self.config["model"]["kf_pixel_ratio"]

        embed_config = self.config["model"]["embedding"]
        # scaling applied to coords before embedding
        self.scale_input = embed_config["scale_input"]
        self.n_embed_funcs = embed_config["n_embed_funcs"]
        # boolean to use gaussian embedding
        self.gauss_embed = bool(embed_config["gauss_embed"])
        self.gauss_embed_std = embed_config["gauss_embed_std"]
        self.optim_embedding = bool(embed_config["optim_embedding"])

        # Evaluation
        self.do_vox_comparison = (
            bool(self.config["eval"]["do_vox_comparison"])
            and "eval_pts_root" in self.config["eval"])
        self.do_eval = self.config["eval"]["do_eval"]
        self.eval_freq_s = self.config["eval"]["eval_freq_s"]
        self.sdf_eval = bool(self.config["eval"]["sdf_eval"])
        self.mesh_eval = bool(self.config["eval"]["mesh_eval"])
        self.eval_times = []
        if self.do_vox_comparison and "eval_pts_root" in self.config["eval"]:
            self.eval_pts_root = self.config["eval"]["eval_pts_root"]
            self.eval_pts_dir = self.config["eval"]["eval_pts_root"]

            self.eval_pts_dir += "/vox/"
            if self.frac_time_perception == 1.:
                self.eval_pts_dir += "0.055/"
            elif self.frac_time_perception == 0.75:
                self.eval_pts_dir += "0.063/"
            elif self.frac_time_perception == 0.5:
                self.eval_pts_dir += "0.078/"
            elif self.frac_time_perception == 0.25:
                self.eval_pts_dir += "0.11/"
            else:
                raise ValueError(
                    'Frace perception time not in [0.25, 0.5, 0.75, 1.]')
            self.eval_pts_dir += [
                x for x in self.seq_dir.split('/') if x != ""][-1] +\
                "/eval_pts/"
            self.eval_times = [float(x) for x in os.listdir(self.eval_pts_dir)]
            self.eval_times.sort()
            print("eval pts dir", self.eval_pts_dir)

            self.cached_dataset = eval_pts.get_cache_dataset(
                self.seq_dir, self.dataset_format, self.scannet_dir)

        # save
        self.save_period = self.config["save"]["save_period"]
        self.save_times = np.arange(
            self.save_period, 2000, self.save_period).tolist()
        self.save_checkpoints = bool(self.config["save"]["save_checkpoints"])
        self.save_slices = bool(self.config["save"]["save_slices"])
        self.save_meshes = bool(self.config["save"]["save_meshes"])

        # Loss
        self.bounds_method = self.config["loss"]["bounds_method"]
        assert self.bounds_method in ["ray", "normal", "pc"]
        self.loss_type = self.config["loss"]["loss_type"]
        assert self.loss_type in ["L1", "L2"]
        self.trunc_weight = self.config["loss"]["trunc_weight"]
        # distance at which losses transition (see paper for details)
        self.trunc_distance = self.config["loss"]["trunc_distance"]
        self.eik_weight = self.config["loss"]["eik_weight"]
        # where to apply the eikonal loss
        self.eik_apply_dist = self.config["loss"]["eik_apply_dist"]
        self.grad_weight = self.config["loss"]["grad_weight"]
        self.orien_loss = bool(self.config["loss"]["orien_loss"])

        self.do_normal = False
        if self.bounds_method == "normal" or self.grad_weight != 0:
            self.do_normal = True

        # optimiser
        self.learning_rate = self.config["optimiser"]["lr"]
        self.weight_decay = self.config["optimiser"]["weight_decay"]

        # Sampling
        self.max_depth = self.config["sample"]["depth_range"][1]
        self.min_depth = self.config["sample"]["depth_range"][0]
        self.dist_behind_surf = self.config["sample"]["dist_behind_surf"]
        self.n_rays = self.config["sample"]["n_rays"]
        self.n_rays_is_kf = self.config["sample"]["n_rays_is_kf"]
        # num stratified samples per ray
        self.n_strat_samples = self.config["sample"]["n_strat_samples"]
        # num surface samples per ray
        self.n_surf_samples = self.config["sample"]["n_surf_samples"]

    def set_scannet_cam_params(self, file):
        info = {}
        with open(file, 'r') as f:
            for line in f.read().splitlines():
                split = line.split(' = ')
                info[split[0]] = split[1]
        self.fx = float(info['fx_depth'])
        self.fy = float(info['fy_depth'])
        self.cx = float(info['mx_depth'])
        self.cy = float(info['my_depth'])
        self.H = int(info['depthHeight'])
        self.W = int(info['depthWidth'])

    def set_cam(self):
        reduce_factor = 16
        self.H_vis = self.H // reduce_factor
        self.W_vis = self.W // reduce_factor
        self.fx_vis = self.fx / reduce_factor
        self.fy_vis = self.fy / reduce_factor
        self.cx_vis = self.cx / reduce_factor
        self.cy_vis = self.cy / reduce_factor

        reduce_factor_up = 8
        self.H_vis_up = self.H // reduce_factor_up
        self.W_vis_up = self.W // reduce_factor_up
        self.fx_vis_up = self.fx / reduce_factor_up
        self.fy_vis_up = self.fy / reduce_factor_up
        self.cx_vis_up = self.cx / reduce_factor_up
        self.cy_vis_up = self.cy / reduce_factor_up

        self.loss_approx_factor = 8
        w_block = self.W // self.loss_approx_factor
        h_block = self.H // self.loss_approx_factor
        increments_w = torch.arange(
            self.loss_approx_factor, device=self.device) * w_block
        increments_h = torch.arange(
            self.loss_approx_factor, device=self.device) * h_block
        c, r = torch.meshgrid(increments_w, increments_h)
        c, r = c.t(), r.t()
        self.increments_single = torch.stack((r, c), dim=2).view(-1, 2)

        # base radius for integrated PE
        # norm_dirs_C = self.dirs_C / torch.linalg.norm(
        #     self.dirs_C, dim=-1, keepdims=True)
        # dx = torch.sqrt(torch.sum(
        #     (self.dirs_C[:, 1, 0, :] - self.dirs_C[:, 0, 0, :])**2, -1))
        # self.pix_radius = dx.item() * 2 / np.sqrt(12)

    def set_directions(self):
        self.dirs_C = geometry.transform.ray_dirs_C(
            1,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.device,
            depth_type="z",
        )

        self.dirs_C_vis = geometry.transform.ray_dirs_C(
            1,
            self.H_vis,
            self.W_vis,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

        self.dirs_C_vis_up = geometry.transform.ray_dirs_C(
            1,
            self.H_vis_up,
            self.W_vis_up,
            self.fx_vis_up,
            self.fy_vis_up,
            self.cx_vis_up,
            self.cy_vis_up,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

    def load_networks(self):
        positional_encoding = embedding.PostionalEncoding(
            min_deg=0,
            max_deg=self.n_embed_funcs,
            scale=self.scale_input,
            transform=self.inv_bounds_transform,
        )

        self.sdf_map = fc_map.SDFMap(
            positional_encoding,
            hidden_size=self.hidden_feature_size,
            hidden_layers_block=self.hidden_layers_block,
            scale_output=self.scale_output,
        ).to(self.device)

        self.optimiser = optim.AdamW(
            self.sdf_map.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def load_checkpoint(self, checkpoint_load_file):
        checkpoint = torch.load(checkpoint_load_file)
        self.sdf_map.load_state_dict(checkpoint["model_state_dict"])
        # self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_gt_sdf(self):
        sdf_grid = np.load(self.gt_sdf_file)
        if self.dataset_format == "ScanNet":
            sdf_grid = np.abs(sdf_grid)
        self.sdf_transform = np.loadtxt(self.sdf_transf_file)
        self.gt_sdf_interp = sdf_util.sdf_interpolator(
            sdf_grid, self.sdf_transform)
        self.sdf_dims = torch.tensor(sdf_grid.shape)

    # Data methods ---------------------------------------

    def load_data(self):

        rgb_transform = transforms.Compose(
            [image_transforms.BGRtoRGB()])
        depth_transform = transforms.Compose(
            [image_transforms.DepthScale(self.inv_depth_scale),
             image_transforms.DepthFilter(self.max_depth)])

        camera_matrix = None
        noisy_depth = None
        if self.dataset_format == "ScanNet":
            dataset_class = dataset.ScanNetDataset
            col_ext = ".jpg"
            self.up = np.array([0., 0., 1.])
            ims_file = self.scannet_dir
        elif self.dataset_format == "replica":
            dataset_class = dataset.ReplicaDataset
            col_ext = ".jpg"
            self.up = np.array([0., 1., 0.])
            ims_file = self.ims_file
        elif self.dataset_format == "replicaCAD":
            dataset_class = dataset.ReplicaDataset
            col_ext = ".png"
            self.up = np.array([0., 1., 0.])
            ims_file = self.ims_file
            noisy_depth = self.noisy_depth
        elif self.dataset_format == "arkit":
            dataset_class = dataset.ARKit
            col_ext = None
            self.up = np.array([0., 0., 1.])
            ims_file = None
            self.traj_file = None
        elif self.dataset_format in ["realsense", "realsense_franka"]:
            dataset_class = dataset.ROSSubscriber
            col_ext = None
            self.up = np.array([0., 0., 1.])
            ims_file =  self.ext_calib # extrinsic calib 
            self.traj_file = None
            camera_matrix = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        elif self.dataset_format == "realsense_franka_offline":
            dataset_class = dataset.RealsenseFrankaOffline
            col_ext = ".jpg"
            self.up = np.array([0., 0., 1.])
            ims_file = self.ims_file
            camera_matrix = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

        self.scene_dataset = dataset_class(
            ims_file,
            traj_file=self.traj_file,
            rgb_transform=rgb_transform,
            depth_transform=depth_transform,
            col_ext=col_ext,
            noisy_depth=noisy_depth,
            distortion_coeffs=self.distortion_coeffs,
            camera_matrix=camera_matrix,
        )

        if self.incremental is False:
            if "im_indices" not in self.config["dataset"]:
                if "n_views" in self.config["dataset"]:
                    n_views = self.config["dataset"]["n_views"]
                    if n_views > 0:
                        n_dataset = len(self.scene_dataset)
                        if self.config["dataset"]["random_views"]:
                            self.indices = np.random.choice(np.arange(0, n_dataset), size=n_views, replace=False)
                        else:
                            self.indices = np.linspace(0, n_dataset, n_views, dtype = int, endpoint= False)
            print("Frame indices", self.indices)
            self.last_is_keyframe = True
            idxs = self.indices
            frame_data = self.get_data(idxs)
            self.add_data(frame_data)

    def get_data(self, idxs):
        frames_data = FrameData()

        for idx in idxs:
            sample = self.scene_dataset[idx]

            im_np = sample["image"][None, ...]
            depth_np = sample["depth"][None, ...]
            T_np = sample["T"][None, ...]

            im = torch.from_numpy(im_np).float().to(self.device) / 255.
            depth = torch.from_numpy(depth_np).float().to(self.device)
            T = torch.from_numpy(T_np).float().to(self.device)

            data = FrameData(
                frame_id=np.array([idx]),
                im_batch=im,
                im_batch_np=im_np,
                depth_batch=depth,
                depth_batch_np=depth_np,
                T_WC_batch=T,
                T_WC_batch_np=T_np,
            )
            if self.do_normal:
                pc = geometry.transform.pointcloud_from_depth_torch(
                    depth[0], self.fx, self.fy, self.cx, self.cy)
                normals = geometry.transform.estimate_pointcloud_normals(pc)
                data.normal_batch = normals[None, :]
            if self.gt_traj is not None:
                data.T_WC_gt = self.gt_traj[idx][None, ...]
            frames_data.add_frame_data(data, replace=False)

        return frames_data

    def add_data(self, data, replace=False):
        # if last frame isn't a keyframe then the new frame
        # replaces last frame in batch.
        replace = self.last_is_keyframe is False

        self.frames.add_frame_data(data, replace)

        if self.last_is_keyframe:
            print("New keyframe. KF ids:", self.frames.frame_id[:-1])

    def add_frame(self, frame_data):
        if self.last_is_keyframe:
            self.frozen_sdf_map = copy.deepcopy(self.sdf_map)

        self.add_data(frame_data)
        self.steps_since_frame = 0
        self.last_is_keyframe = False
        self.optim_frames = self.iters_per_frame
        self.noise_std = self.noise_frame

    # Keyframe methods ----------------------------------

    def is_keyframe(self, T_WC, depth_gt):
        sample_pts = self.sample_points(
            depth_gt, T_WC, n_rays=self.n_rays_is_kf, dist_behind_surf=0.8)

        pc = sample_pts["pc"]
        z_vals = sample_pts["z_vals"]
        depth_sample = sample_pts["depth_sample"]

        with torch.set_grad_enabled(False):
            sdf = self.frozen_sdf_map(pc, noise_std=self.noise_std)

        z_vals, ind1 = z_vals.sort(dim=-1)
        ind0 = torch.arange(z_vals.shape[0])[:, None].repeat(
            1, z_vals.shape[1])
        sdf = sdf[ind0, ind1]

        view_depth = render.sdf_render_depth(z_vals, sdf)

        loss = torch.abs(view_depth - depth_sample) / depth_sample

        below_th = loss < self.kf_dist_th
        size_loss = below_th.shape[0]
        below_th_prop = below_th.sum().float() / size_loss
        is_keyframe = below_th_prop.item() < self.kf_pixel_ratio

        print(
            "Proportion of loss below threshold",
            below_th_prop.item(),
            "for KF should be less than",
            self.kf_pixel_ratio,
            " ---> is keyframe:",
            is_keyframe
        )

        return is_keyframe

    def check_keyframe_latest(self):
        """
        returns whether or not to add a new frame.
        """
        add_new_frame = False

        if self.last_is_keyframe:
            # Latest frame is already a keyframe. We have now
            # finished the extra steps and want to add a new frame
            add_new_frame = True

        else:
            # Check if latest frame should be a keyframe.
            T_WC = self.frames.T_WC_batch[-1].unsqueeze(0)
            depth_gt = self.frames.depth_batch[-1].unsqueeze(0)
            self.last_is_keyframe = self.is_keyframe(T_WC, depth_gt)

            time_since_kf = self.tot_step_time - self.frames.frame_id[-2] / 30.
            if time_since_kf > 5. and not self.live:
                print("More than 5 seconds since last kf, so add new")
                self.last_is_keyframe = True

            if self.last_is_keyframe:
                self.optim_frames = self.iters_per_kf
                self.noise_std = self.noise_kf
            else:
                add_new_frame = True

        return add_new_frame

    def select_keyframes(self):
        """
        Use most recent two keyframes then fill rest of window
        based on loss distribution across the remaining keyframes.
        """
        n_frames = len(self.frames)
        limit = n_frames - 2
        denom = self.frames.frame_avg_losses[:-2].sum()
        loss_dist = self.frames.frame_avg_losses[:-2] / denom
        loss_dist_np = loss_dist.cpu().numpy()

        select_size = self.window_size - 2

        rand_ints = np.random.choice(
            np.arange(0, limit),
            size=select_size,
            replace=False,
            p=loss_dist_np)

        last = n_frames - 1
        idxs = [*rand_ints, last - 1, last]

        return idxs

    def clear_keyframes(self):
        self.frames = FrameData()  # keyframes
        self.gt_depth_vis = None
        self.gt_im_vis = None

    # Main training methods ----------------------------------

    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
        norm_batch=None,
        active_loss_approx=None,
        n_rays=None,
        dist_behind_surf=None,
        n_strat_samples=None,
        n_surf_samples=None,
    ):
        """
        Sample points by first sampling pixels, then sample depths along
        the backprojected rays.
        """
        if n_rays is None:
            n_rays = self.n_rays
        if dist_behind_surf is None:
            dist_behind_surf = self.dist_behind_surf
        if n_strat_samples is None:
            n_strat_samples = self.n_strat_samples
        if n_surf_samples is None:
            n_surf_samples = self.n_surf_samples

        n_frames = depth_batch.shape[0]
        if active_loss_approx is None:
            indices_b, indices_h, indices_w = sample.sample_pixels(
                n_rays, n_frames, self.H, self.W, device=self.device)
        else:
            # indices_b, indices_h, indices_w = \
            #     active_sample.active_sample_pixels(
            #         n_rays, n_frames, self.H, self.W, device=self.device,
            #         loss_approx=active_loss_approx,
            #         increments_single=self.increments_single
            #     )
            raise Exception('Active sampling not currently supported.')

        get_masks = active_loss_approx is None
        (
            dirs_C_sample,
            depth_sample,
            norm_sample,
            T_WC_sample,
            binary_masks,
            indices_b,
            indices_h,
            indices_w
        ) = sample.get_batch_data(
            depth_batch,
            T_WC_batch,
            self.dirs_C,
            indices_b,
            indices_h,
            indices_w,
            norm_batch=norm_batch,
            get_masks=get_masks,
        )

        max_depth = depth_sample + dist_behind_surf
        pc, z_vals = sample.sample_along_rays(
            T_WC_sample,
            self.min_depth,
            max_depth,
            n_strat_samples,
            n_surf_samples,
            dirs_C_sample,
            gt_depth=depth_sample,
            grad=False,
        )

        sample_pts = {
            "depth_batch": depth_batch,
            "pc": pc,
            "z_vals": z_vals,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "dirs_C_sample": dirs_C_sample,
            "depth_sample": depth_sample,
            "T_WC_sample": T_WC_sample,
            "norm_sample": norm_sample,
            "binary_masks": binary_masks,
        }
        return sample_pts

    def sdf_eval_and_loss(
        self,
        sample,
        do_avg_loss=True,
    ):
        pc = sample["pc"]
        z_vals = sample["z_vals"]
        indices_b = sample["indices_b"]
        indices_h = sample["indices_h"]
        indices_w = sample["indices_w"]
        dirs_C_sample = sample["dirs_C_sample"]
        depth_sample = sample["depth_sample"]
        T_WC_sample = sample["T_WC_sample"]
        norm_sample = sample["norm_sample"]
        binary_masks = sample["binary_masks"]
        depth_batch = sample["depth_batch"]

        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0
        if do_sdf_grad:
            pc.requires_grad_()

        sdf = self.sdf_map(pc, noise_std=self.noise_std)

        sdf_grad = None
        if do_sdf_grad:
            sdf_grad = fc_map.gradient(pc, sdf)

        # compute bounds

        bounds, grad_vec = loss.bounds(
            self.bounds_method,
            dirs_C_sample,
            depth_sample,
            T_WC_sample,
            z_vals,
            pc,
            self.trunc_distance,
            norm_sample,
            do_grad=True,
        )

        # compute loss

        sdf_loss_mat, free_space_ixs = loss.sdf_loss(
            sdf, bounds, self.trunc_distance, loss_type=self.loss_type)

        eik_loss_mat = None
        if self.eik_weight != 0:
            eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)

        grad_loss_mat = None
        if self.grad_weight != 0:
            pred_norms = sdf_grad[:, 0]
            surf_loss_mat = 1 - self.cosSim(pred_norms, norm_sample)

            grad_vec[torch.where(grad_vec[..., 0].isnan())] = \
                norm_sample[torch.where(grad_vec[..., 0].isnan())[0]]
            grad_loss_mat = 1 - self.cosSim(grad_vec, sdf_grad[:, 1:])
            grad_loss_mat = torch.cat(
                (surf_loss_mat[:, None], grad_loss_mat), dim=1)

            if self.orien_loss:
                grad_loss_mat = (grad_loss_mat > 1).float()

        total_loss, total_loss_mat, losses = loss.tot_loss(
            sdf_loss_mat, grad_loss_mat, eik_loss_mat,
            free_space_ixs, bounds, self.eik_apply_dist,
            self.trunc_weight, self.grad_weight, self.eik_weight,
        )

        loss_approx, frame_avg_loss = None, None
        if do_avg_loss:
            loss_approx, frame_avg_loss = loss.frame_avg(
                total_loss_mat, depth_batch, indices_b, indices_h, indices_w,
                self.W, self.H, self.loss_approx_factor, binary_masks)

        # # # for plot
        # z_to_euclidean_depth = dirs_C_sample.norm(dim=-1)
        # ray_target = depth_sample[:, None] - z_vals
        # ray_target = z_to_euclidean_depth[:, None] * ray_target

        # # apply correction based on angle between ray and normal
        # costheta = torch.abs(self.cosSim(-dirs_C_sample, norm_sample))
        # # only apply correction out to truncation distance
        # sub = self.trunc_distance * (1. - costheta)
        # normal_target_fs = ray_target - sub[:, None]
        # # target_normal = ray_target.clone()
        # target_normal = normal_target_fs
        # ixs = target_normal < self.trunc_distance
        # target_normal[ixs] = (ray_target * costheta[:, None])[ixs]

        # self.check_gt_sdf(
        #     depth_sample, z_vals, dirs_C_sample, pc,
        #     ray_target, target_sdf, target_normal)

        return (
            total_loss,
            losses,
            loss_approx,
            frame_avg_loss,
        )

    def check_gt_sdf(self, depth_sample, z_vals, dirs_C_sample, pc,
                     target_ray, target_pc, target_normal):
                     # origins, dirs_W):
        # reorder in increasing z vals
        z_vals, indices = z_vals.sort(dim=-1)
        row_ixs = torch.arange(pc.shape[0])[:, None].repeat(1, pc.shape[1])

        pc = pc[row_ixs, indices]
        target_ray = target_ray[row_ixs, indices]
        target_pc = target_pc[row_ixs, indices]
        if target_normal is not None:
            target_normal = target_normal[row_ixs, indices]

        z2euc_sample = torch.norm(dirs_C_sample, dim=-1)
        z_vals = z_vals * z2euc_sample[:, None]

        scene = trimesh.Scene(trimesh.load(self.scene_file))

        with torch.set_grad_enabled(False):
            j = 0
            fig, ax = plt.subplots(3, 1, figsize=(11, 10))

            for i in [9, 19, 23]:  # range(0, 100):
                gt_sdf = sdf_util.eval_sdf_interp(
                    self.gt_sdf_interp,
                    pc[i].reshape(-1, 3).detach().cpu().numpy(),
                    handle_oob='fill', oob_val=np.nan)

                x = z_vals[i].cpu()
                lw = 2.5
                ax[j].hlines(0, x[0], x[-1], color="gray", linestyle="--")
                ax[j].plot(
                    x, gt_sdf, label="True signed distance",
                    color="C1", linewidth=lw
                )
                ax[j].plot(
                    x, target_ray[i].cpu(), label="Ray",
                    color="C3", linewidth=lw
                )
                if target_normal is not None:
                    ax[j].plot(
                        x, target_normal[i].cpu(), label="Normal",
                        color="C2", linewidth=lw
                    )
                ax[j].plot(
                    x, target_pc[i].cpu(), label="Batch distance",
                    color="C0", linewidth=lw
                )

                # print("diffs", target_sdf[i].cpu() - gt_sdf)
                if j == 2:
                    ax[j].set_xlabel("Distance along ray, d [m]", fontsize=24)
                    ax[j].set_yticks([0, 4, 8])
                # ax[j].set_ylabel("Signed distance (m)", fontsize=21)
                ax[j].tick_params(axis='both', which='major', labelsize=24)
                # ax[j].set_xticks(fontsize=20)
                # ax[j].set_yticks(fontsize=20)
                # if j == 0:
                #     ax[j].legend(fontsize=20)
                j += 1

            fig.text(
                0.05, 0.5, 'Signed distance [m]',
                va='center', rotation='vertical', fontsize=24
            )
            # plt.tight_layout()
            plt.show()

            # intersection = dirs_W[i] * depth_sample[i] + origins[i]
            # int_pc = trimesh.PointCloud(
            #     intersection[None, :].cpu(), [255, 0, 0, 255])
            # scene.add_geometry(int_pc)

            # pts = pc[i].detach().cpu().numpy()
            # colormap_fn = sdf_util.get_colormap()
            # col = colormap_fn.to_rgba(gt_sdf, alpha=1., bytes=False)
            # tm_pc = trimesh.PointCloud(pts, col)
            # scene.add_geometry(tm_pc)

        scene.show()

    def step(self):
        start, end = start_timing()

        depth_batch = self.frames.depth_batch
        T_WC_batch = self.frames.T_WC_batch
        norm_batch = self.frames.normal_batch if self.do_normal else None

        if len(self.frames) > self.window_size and self.incremental:
            idxs = self.select_keyframes()
            # print("selected frame ids", self.frames.frame_id[idxs[:-1]])
        else:
            idxs = np.arange(T_WC_batch.shape[0])
        self.active_idxs = idxs

        depth_batch = depth_batch[idxs]
        T_WC_select = T_WC_batch[idxs]

        sample_pts = self.sample_points(
            depth_batch, T_WC_select, norm_batch=norm_batch)
        self.active_pixels = {
            'indices_b': sample_pts['indices_b'],
            'indices_h': sample_pts['indices_h'],
            'indices_w': sample_pts['indices_w'],
        }

        total_loss, losses, active_loss_approx, frame_avg_loss = \
            self.sdf_eval_and_loss(sample_pts, do_avg_loss=True)

        self.frames.frame_avg_losses[idxs] = frame_avg_loss

        total_loss.backward()
        self.optimiser.step()
        for param_group in self.optimiser.param_groups:
            params = param_group["params"]
            for param in params:
                param.grad = None

        # if self.do_active:
        #     sample_pts = self.sample_points(
        #         depth_batch, T_WC_select, norm_batch=norm_batch,
        #         active_loss_approx=active_loss_approx)

        #     loss_active, _, _, _ = \
        #         self.sdf_eval_and_loss(sample_pts, do_avg_loss=False)

        #     loss_active.backward()
        #     self.optimiser.step()
        #     for param_group in self.optimiser.param_groups:
        #         params = param_group["params"]
        #         for param in params:
        #             param.grad = None

            # loss_approx = loss_approx[-1].detach().cpu().numpy()
            # loss_approx_viz = imgviz.depth2rgb(loss_approx)
            # loss_approx_viz = cv2.cvtColor(loss_approx_viz, cv2.COLOR_RGB2BGR)
            # loss_approx_viz = cv2.resize(loss_approx_viz, (200, 200),
            #                              interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("loss_approx_viz", loss_approx_viz)
            # cv2.waitKey(1)

        step_time = end_timing(start, end)
        time_s = step_time / 1000.
        self.tot_step_time += (1 / self.frac_time_perception) * time_s
        self.steps_since_frame += 1

        return losses, step_time

    # Visualisation methods -----------------------------------

    def update_vis_vars(self):
        depth_batch_np = self.frames.depth_batch_np
        im_batch_np = self.frames.im_batch_np

        if self.gt_depth_vis is None:
            updates = depth_batch_np.shape[0]
        else:
            diff_size = depth_batch_np.shape[0] - \
                self.gt_depth_vis.shape[0]
            updates = diff_size + 1

        for i in range(updates, 0, -1):
            prev_depth_gt = depth_batch_np[-i]
            prev_im_gt = im_batch_np[-i]
            prev_depth_gt_resize = imgviz.resize(
                prev_depth_gt, width=self.W_vis,
                height=self.H_vis,
                interpolation="nearest")[None, ...]
            prev_im_gt_resize = imgviz.resize(
                prev_im_gt, width=self.W_vis,
                height=self.H_vis)[None, ...]

            replace = False
            if i == updates:
                replace = True

            self.gt_depth_vis = data_util.expand_data(
                self.gt_depth_vis,
                prev_depth_gt_resize,
                replace=replace)
            self.gt_im_vis = data_util.expand_data(
                self.gt_im_vis,
                prev_im_gt_resize,
                replace=replace)

    def latest_frame_vis(self, do_render=True):
        start, end = start_timing()

        # get latest frame from camera
        if self.live:
            data = self.scene_dataset[0]
            image = data['image']
            depth = data['depth']
            T_WC_np = data['T']
        else:
            image = self.frames.im_batch_np[-1]
            depth = self.frames.depth_batch_np[-1]
            T_WC_np = self.frames.T_WC_batch_np[-1]

        w = self.W_vis_up * 2
        h = self.H_vis_up * 2
        image = cv2.resize(image, (w, h))
        depth = cv2.resize(depth, (w, h))
        depth_viz = imgviz.depth2rgb(
            depth, min_value=self.min_depth, max_value=self.max_depth)
        # depth_viz[depth == 0] = [0, 255, 0]

        rgbd_vis = np.hstack((image, depth_viz))

        if not do_render:
            return rgbd_vis, None, T_WC_np
        else:
            T_WC = torch.FloatTensor(T_WC_np).to(self.device)[None, ...]

            with torch.set_grad_enabled(False):
                # efficient depth and normals render
                # valid_depth = depth != 0.0
                # depth_sample = torch.FloatTensor(depth).to(self.device)[valid_depth]
                # max_depth = depth_sample + 0.5  # larger max depth for depth render
                # dirs_C = self.dirs_C_vis[0, valid_depth.flatten()]

                pc, z_vals = sample.sample_along_rays(
                    T_WC,
                    self.min_depth,
                    self.max_depth,
                    n_stratified_samples=20,
                    n_surf_samples=0,
                    dirs_C=self.dirs_C_vis,
                    gt_depth=None,  # depth_sample
                )

                sdf = self.sdf_map(pc)
                # sdf = fc_map.chunks(pc, self.chunk_size, self.sdf_map)
                depth_vals_vis = render.sdf_render_depth(z_vals, sdf)

                depth_up = torch.nn.functional.interpolate(
                    depth_vals_vis.view(1, 1, self.H_vis, self.W_vis),
                    size=[self.H_vis_up, self.W_vis_up],
                    mode='bilinear', align_corners=True
                )
                depth_up = depth_up.view(-1)

                pc_up, z_vals_up = sample.sample_along_rays(
                    T_WC,
                    depth_up - 0.1,
                    depth_up + 0.1,
                    n_stratified_samples=12,
                    n_surf_samples=12,
                    dirs_C=self.dirs_C_vis_up,
                )
                sdf_up = self.sdf_map(pc_up)
                depth_vals = render.sdf_render_depth(z_vals_up, sdf_up)

            surf_normals_C = render.render_normals(
                T_WC, depth_vals[None, ...], self.sdf_map, self.dirs_C_vis_up)

            # render_depth = torch.zeros(self.H_vis, self.W_vis)
            # render_depth[valid_depth] = depth_vals.detach().cpu()
            # render_depth = render_depth.numpy()
            render_depth = depth_vals.view(self.H_vis_up, self.W_vis_up).cpu().numpy()
            render_depth_viz = imgviz.depth2rgb(
                render_depth, min_value=self.min_depth, max_value=self.max_depth)

            surf_normals_C = (- surf_normals_C + 1.0) / 2.0
            surf_normals_C = torch.clip(surf_normals_C, 0., 1.)
            # normals_viz = torch.zeros(self.H_vis, self.W_vis, 3)
            # normals_viz[valid_depth] = surf_normals_C.detach().cpu()
            normals_viz = surf_normals_C.view(self.H_vis_up, self.W_vis_up, 3).detach().cpu()
            normals_viz = (normals_viz.numpy() * 255).astype(np.uint8)

            render_vis = np.hstack((normals_viz, render_depth_viz))
            w_up = int(render_vis.shape[1] * 2)
            h_up = int(render_vis.shape[0] * 2)
            render_vis = cv2.resize(render_vis, (w_up, h_up))

            elapsed = end_timing(start, end)
            print("Time for depth and normal render", elapsed)
            return rgbd_vis, render_vis, T_WC_np

    def keyframe_vis(self, reduce_factor=2):
        start, end = start_timing()

        h, w = self.frames.im_batch_np.shape[1:3]
        h = int(h / reduce_factor)
        w = int(w / reduce_factor)

        kf_vis = []
        for i, kf in enumerate(self.frames.im_batch_np):
            kf = cv2.resize(kf, (w, h))
            kf = cv2.cvtColor(kf, cv2.COLOR_BGR2RGB)

            pad_color = [255, 255, 255]
            if self.active_idxs is not None and self.active_pixels is not None:
                if i in self.active_idxs:
                    pad_color = [0, 0, 139]

                    # show sampled pixels
                    act_inds_mask = self.active_pixels['indices_b'] == i
                    h_inds = self.active_pixels['indices_h'][act_inds_mask]
                    w_inds = self.active_pixels['indices_w'][act_inds_mask]
                    mask = np.zeros([self.H, self.W])
                    mask[h_inds.cpu().numpy(), w_inds.cpu().numpy()] = 1
                    mask = ndimage.binary_dilation(mask, iterations=6)
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (w, h)).astype(np.bool)
                    kf[mask, :] = [0, 0, 139]

            kf = cv2.copyMakeBorder(
                kf, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=pad_color)
            kf = cv2.copyMakeBorder(
                kf, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            kf_vis.append(kf)

        kf_vis = np.hstack(kf_vis)
        elapsed = end_timing(start, end)
        print("Time for kf vis", elapsed)
        return kf_vis

    def frames_vis(self):
        view_depths = self.render_depth_vis()
        view_normals = self.render_normals_vis(view_depths)
        view_depths = view_depths.cpu().numpy()
        view_normals = view_normals.detach().cpu().numpy()
        gt_depth_ims = self.gt_depth_vis
        im_batch_np = self.gt_im_vis

        views = []
        for batch_i in range(len(self.frames)):
            depth = view_depths[batch_i]
            depth_viz = imgviz.depth2rgb(
                depth, min_value=self.min_depth, max_value=self.max_depth)

            gt = gt_depth_ims[batch_i]
            gt_depth = imgviz.depth2rgb(
                gt, min_value=self.min_depth, max_value=self.max_depth)

            loss = np.abs(gt - depth)
            loss[gt == 0] = 0
            loss_viz = imgviz.depth2rgb(loss)

            normals = view_normals[batch_i]
            normals = (- normals + 1.0) / 2.0
            normals = np.clip(normals, 0., 1.)
            normals = (normals * 255).astype(np.uint8)

            visualisations = [gt_depth, depth_viz, loss_viz, normals]
            if im_batch_np is not None:
                visualisations.append(im_batch_np[batch_i])

            viz = np.vstack(visualisations)
            views.append(viz)

        viz = np.hstack(views)
        return viz

    def render_depth_vis(self):
        view_depths = []

        depth_gt = self.frames.depth_batch_np
        T_WC_batch = self.frames.T_WC_batch
        if self.frames.T_WC_track:
            T_WC_batch = self.frames.T_WC_track

        with torch.set_grad_enabled(False):
            for batch_i in range(len(self.frames)):  # loops through frames
                T_WC = T_WC_batch[batch_i].unsqueeze(0)

                depth_sample = depth_gt[batch_i]
                depth_sample = cv2.resize(
                    depth_sample, (self.W_vis, self.H_vis))
                depth_sample = torch.FloatTensor(depth_sample).to(self.device)

                # larger max depth for depth render
                max_depth = (depth_sample + 0.8).flatten()
                pc, z_vals = sample.sample_along_rays(
                    T_WC,
                    self.min_depth,
                    max_depth,
                    self.n_strat_samples,
                    n_surf_samples=0,
                    dirs_C=self.dirs_C_vis[0],
                    gt_depth=None,
                    grad=False,
                )

                sdf = self.sdf_map(pc)

                view_depth = render.sdf_render_depth(z_vals, sdf)
                view_depth = view_depth.view(self.H_vis, self.W_vis)
                view_depths.append(view_depth)

            view_depths = torch.stack(view_depths)
        return view_depths

    def render_normals_vis(self, view_depths):
        view_normals = []

        T_WC_batch = self.frames.T_WC_batch
        if self.frames.T_WC_track:
            T_WC_batch = self.frames.T_WC_track

        for batch_i in range(len(self.frames)):  # loops through frames
            T_WC = T_WC_batch[batch_i].unsqueeze(0)
            view_depth = view_depths[batch_i]

            surf_normals_C = render.render_normals(
                T_WC, view_depth, self.sdf_map, self.dirs_C_vis[0])
            view_normals.append(surf_normals_C)

        view_normals = torch.stack(view_normals)
        return view_normals

    def draw_3D(
        self,
        show_pc=False,
        show_mesh=False,
        draw_cameras=False,
        show_gt_mesh=False,
        camera_view=True,
    ):  
        start, end = start_timing()

        scene = trimesh.Scene()
        scene.set_camera()
        scene.camera.focal = (self.fx, self.fy)
        scene.camera.resolution = (self.W, self.H)

        T_WC_np = self.frames.T_WC_batch_np
        if self.frames.T_WC_track:
            T_WC_np = self.frames.T_WC_track.cpu().numpy()

        if draw_cameras:
            n_frames = len(self.frames)
            cam_scale = 0.25 if "franka" in self.dataset_format else 1.0
            draw3D.draw_cams(
                n_frames, T_WC_np, scene, color=(0.0, 1.0, 0.0, 1.0), cam_scale = cam_scale)

            if self.frames.T_WC_gt:  # show gt and input poses too
                draw3D.draw_cams(
                    n_frames, self.frames.T_WC_gt, scene,
                    color=(1.0, 0.0, 1.0, 0.8), cam_scale = cam_scale)
                draw3D.draw_cams(
                    n_frames, self.frames.T_WC_batch_np, scene,
                    color=(1., 0., 0., 0.8), cam_scale = cam_scale)

            if self.incremental:
                trajectory_gt = self.frames.T_WC_batch_np[:, :3, 3]
                if self.frames.T_WC_gt is not None:
                    trajectory_gt = self.frames.T_WC_gt[:, :3, 3]
                visualisation.draw3D.draw_trajectory(
                    trajectory_gt, scene, color=(1.0, 0.0, 0.0)
                )

        if show_pc:
            if self.gt_depth_vis is None:
                self.update_vis_vars()  # called in self.mesh_rec
            pcs_cam = geometry.transform.backproject_pointclouds(
                self.gt_depth_vis, self.fx_vis, self.fy_vis,
                self.cx_vis, self.cy_vis)
            pc_w, colors = draw3D.draw_pc(
                n_frames,
                pcs_cam,
                T_WC_np,
                self.gt_im_vis,
            )
            pc = trimesh.PointCloud(pc_w, colors=colors)
            scene.add_geometry(pc, geom_name='depth_pc')

        if show_mesh:
            try:
                sdf_mesh = self.mesh_rec()
                scene.add_geometry(sdf_mesh, geom_name="rec_mesh")
            except ValueError: # ValueError: Surface level must be within volume data range.
                print("ValueError: Surface level must be within volume data range.")
                pass

        if show_gt_mesh:
            gt_mesh = trimesh.load(self.scene_file)
            gt_mesh.visual.material.image.putalpha(50)
            scene.add_geometry(gt_mesh)

        if not camera_view and self.scene_center is not None:
            if "realsense_franka" in self.dataset_format:
                cam_pos = self.scene_center + self.up * 1 + np.array([1, -1, 0.])
            else:
                cam_pos = self.scene_center + self.up * 12 + np.array([3., 0., 0.])
            R, t = geometry.transform.look_at(
                cam_pos, self.scene_center, -self.up)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            scene.camera_transform = geometry.transform.to_trimesh(T)
        else:
            view_idx = -1
            scene.camera_transform = geometry.transform.to_trimesh(
                T_WC_np[view_idx])
            scene.camera_transform = (
                scene.camera_transform
                @ trimesh.transformations.translation_matrix([0, 0, 0.1]))

        elapsed = end_timing(start, end)
        print(f'Time to draw scene: {elapsed}ms')
        return scene

    def draw_obj_3D(self, show_gt_mesh=True):
        if self.obj_bounds_file is not None:

            scene = trimesh.Scene()
            scene.set_camera()
            scene.camera.focal = (self.fx, self.fy)
            scene.camera.resolution = (self.W, self.H)

            if show_gt_mesh:
                gt_mesh = trimesh.load(self.scene_file)
                gt_mesh.visual.material.image.putalpha(50)

            obj_bounds = metrics.get_obj_eval_bounds(
                self.obj_bounds_file, self.up_ix,
                expand_m=0.2, expand_down=True)

            for i, bounds in enumerate(obj_bounds):

                x = torch.linspace(bounds[0, 0], bounds[1, 0], 128)
                y = torch.linspace(bounds[0, 1], bounds[1, 1], 128)
                z = torch.linspace(bounds[0, 2], bounds[1, 2], 128)
                xx, yy, zz = torch.meshgrid(x, y, z)
                pc = torch.cat(
                    (xx[..., None], yy[..., None], zz[..., None]), dim=3)
                pc = pc.view(-1, 3).to(self.device)

                with torch.set_grad_enabled(False):
                    sdf = fc_map.chunks(
                        pc, self.chunk_size, self.sdf_map,
                        # surf_dists=gt_dist,
                    )
                T = np.eye(4)
                T[:3, 3] = bounds[0] + 0.5 * (bounds[1] - bounds[0])
                sdf = sdf.view(128, 128, 128)
                obj_mesh = draw3D.draw_mesh(
                    sdf, 0.5 * (bounds[1] - bounds[0]), T)
                obj_mesh.visual.face_colors = [160, 160, 160, 160]
                scene.add_geometry(obj_mesh)

                if show_gt_mesh:
                    box = trimesh.primitives.Box(
                        extents=bounds[1] - bounds[0], transform=T)
                    crop = gt_mesh.slice_plane(
                        box.facets_origin, -box.facets_normal)
                    crop.visual.face_colors = [0, 160, 50, 160]
                    scene.add_geometry(crop)

            scene.set_camera()

            return scene
        return None

    def get_sdf_grid(self):
        with torch.set_grad_enabled(False):

            # gt_dist = sdf_util.eval_sdf_interp(
            #     self.gt_sdf_interp, self.grid_pc.cpu().numpy(),
            #     handle_oob='fill')
            # gt_dist = torch.FloatTensor(gt_dist).to(self.device)

            sdf = fc_map.chunks(
                self.grid_pc,
                self.chunk_size,
                self.sdf_map,
                # surf_dists=gt_dist,
            )

            dim = self.grid_dim
            sdf = sdf.view(dim, dim, dim)

        return sdf

    def get_sdf_grid_pc(self, include_gt=False, mask_near_pc=False):
        sdf_grid = self.get_sdf_grid()
        grid_pc = self.grid_pc.reshape(
            self.grid_dim, self.grid_dim, self.grid_dim, 3)
        sdf_grid_pc = torch.cat((grid_pc, sdf_grid[..., None]), dim=-1)
        sdf_grid_pc = sdf_grid_pc.detach().cpu().numpy()

        if include_gt and self.gt_sdf_interp is not None:
            self.gt_sdf_interp.bounds_error = False
            self.gt_sdf_interp.fill_value = 0.0
            gt_sdf = self.gt_sdf_interp(self.grid_pc.cpu())
            gt_sdf = gt_sdf.reshape(
                self.grid_dim, self.grid_dim, self.grid_dim)
            sdf_grid_pc = np.concatenate(
                (sdf_grid_pc, gt_sdf[..., None]), axis=-1)
            self.gt_sdf_interp.bounds_error = True

        keep_mask = None
        if mask_near_pc:
            self.update_vis_vars()
            pcs_cam = geometry.transform.backproject_pointclouds(
                self.gt_depth_vis, self.fx_vis, self.fy_vis,
                self.cx_vis, self.cy_vis)
            pc, _ = draw3D.draw_pc(
                len(self.frames),
                pcs_cam,
                self.frames.T_WC_batch_np,
            )
            tree = KDTree(pc)
            sparse_grid = sdf_grid_pc[::10, ::10, ::10, :3]
            dists, _ = tree.query(sparse_grid.reshape(-1, 3), k=1)
            dists = dists.reshape(sparse_grid.shape[:-1])
            keep_mask = dists < self.crop_dist
            keep_mask = keep_mask.repeat(10, axis=0).repeat(10, axis=1).repeat(10, axis=2)

        return sdf_grid_pc, keep_mask

    def view_sdf(self):
        show_mesh = False if self.gt_scene else True
        scene = self.draw_3D(
            show_pc=True,
            show_mesh=show_mesh,
            draw_cameras=True,
            show_gt_mesh=self.gt_scene,
            camera_view=True,
        )
        sdf_grid_pc, _ = self.get_sdf_grid_pc(include_gt=False)
        sdf_grid_pc = np.transpose(sdf_grid_pc, (2, 1, 0, 3))
        # sdf_grid_pc = sdf_grid_pc[:, :, ::-1]  # for replica
        visualisation.sdf_viewer.SDFViewer(
            scene=scene, sdf_grid_pc=sdf_grid_pc,
            colormap=True, surface_cutoff=0.01
        )

    def mesh_rec(self, crop_mesh_with_pc=True):
        """
        Generate mesh reconstruction.
        """
        self.update_vis_vars()
        pcs_cam = geometry.transform.backproject_pointclouds(
            self.gt_depth_vis, self.fx_vis, self.fy_vis,
            self.cx_vis, self.cy_vis)
        pc, _ = draw3D.draw_pc(
            len(self.frames),
            pcs_cam,
            self.frames.T_WC_batch_np,
        )

        if self.gt_scene is False and self.incremental:
            pc_tm = trimesh.PointCloud(pc)
            self.set_scene_properties(pc_tm)

        sdf = self.get_sdf_grid()

        sdf_mesh = draw3D.draw_mesh(
            sdf,
            self.scene_scale_np,
            self.bounds_transform_np,
            color_by="none",
        )

        if crop_mesh_with_pc:
            tree = KDTree(pc)
            dists, _ = tree.query(sdf_mesh.vertices, k=1)
            keep_ixs = dists < self.crop_dist
            face_mask = keep_ixs[sdf_mesh.faces].any(axis=1)
            sdf_mesh.update_faces(face_mask)
            sdf_mesh.remove_unreferenced_vertices()
            # sdf_mesh.visual.vertex_colors[~keep_ixs, 3] = 10

        if self.new_grid_dim is not None:
            self.grid_dim = self.new_grid_dim
            self.grid_pc = self.new_grid_pc
            self.new_grid_dim = None
            self.new_grid_pc = None

        return sdf_mesh

    def write_mesh(self, filename, im_pose=None):
        mesh = self.mesh_rec()

        data = trimesh.exchange.ply.export_ply(mesh)
        out = open(filename, "wb+")
        out.write(data)
        out.close()

        if im_pose is not None:
            scene = trimesh.Scene(mesh)
            im = draw3D.capture_scene_im(
                scene, im_pose, tm_pose=True)
            cv2.imwrite(filename[:-4] + ".png", im[..., :3][..., ::-1])

    def compute_slices(
        self, z_ixs=None, n_slices=6,
        include_gt=False, include_diff=False, include_chomp=False,
        draw_cams=False, sdf_range=[-2, 2],
    ):
        # Compute points to query
        if z_ixs is None:
            z_ixs = torch.linspace(30, self.grid_dim - 30, n_slices)
            z_ixs = torch.round(z_ixs).long()
        z_ixs = z_ixs.to(self.device)

        pc = self.grid_pc.reshape(
            self.grid_dim, self.grid_dim, self.grid_dim, 3)
        pc = torch.index_select(pc, self.up_ix, z_ixs)

        if not self.up_aligned:
            indices = np.arange(len(z_ixs))[::-1]
            indices = torch.from_numpy(indices.copy()).to(self.device)
            pc = torch.index_select(pc, self.up_ix, indices)

        cmap = sdf_util.get_colormap(sdf_range=sdf_range)
        grid_shape = pc.shape[:-1]
        n_slices = grid_shape[self.up_ix]
        pc = pc.reshape(-1, 3)

        scales = torch.cat([
            self.scene_scale[:self.up_ix], self.scene_scale[self.up_ix + 1:]])
        im_size = 256 * scales / scales.min()
        im_size = im_size.int().cpu().numpy()

        slices = {}

        with torch.set_grad_enabled(False):
            sdf = fc_map.chunks(pc, self.chunk_size, self.sdf_map)
            sdf = sdf.detach().cpu().numpy()
        sdf_viz = cmap.to_rgba(sdf.flatten(), alpha=1., bytes=False)
        sdf_viz = (sdf_viz * 255).astype(np.uint8)[..., :3]
        sdf_viz = sdf_viz.reshape(*grid_shape, 3)
        sdf_viz = [
            cv2.resize(np.take(sdf_viz, i, self.up_ix), im_size[::-1])
            for i in range(n_slices)
        ]
        slices["pred_sdf"] = sdf_viz

        if include_chomp:
            cost = metrics.chomp_cost(sdf, epsilon=2.)
            cost_viz = imgviz.depth2rgb(
                cost.reshape(self.grid_dim, -1), min_value=0., max_value=1.5)
            cost_viz = cost_viz.reshape(*grid_shape, 3)
            cost_viz = [
                cv2.resize(np.take(cost_viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["pred_cost"] = cost_viz

        pc = pc.reshape(*grid_shape, 3)
        pc = pc.detach().cpu().numpy()

        if include_gt:
            gt_sdf = sdf_util.eval_sdf_interp(
                self.gt_sdf_interp, pc, handle_oob='fill')
            gt_sdf_viz = cmap.to_rgba(gt_sdf.flatten(), alpha=1., bytes=False)
            gt_sdf_viz = gt_sdf_viz.reshape(*grid_shape, 4)
            gt_sdf_viz = (gt_sdf_viz * 255).astype(np.uint8)[..., :3]
            gt_sdf_viz = [
                cv2.resize(np.take(gt_sdf_viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["gt_sdf"] = gt_sdf_viz

            if include_chomp:
                gt_costs = metrics.chomp_cost(gt_sdf, epsilon=2.)
                gt_cost_viz = imgviz.depth2rgb(
                    gt_costs.reshape(self.grid_dim, -1),
                    min_value=0., max_value=1.5)
                gt_cost_viz = gt_cost_viz.reshape(*grid_shape, 3)
                gt_cost_viz = [
                    cv2.resize(
                        np.take(gt_cost_viz, i, self.up_ix), im_size[::-1])
                    for i in range(n_slices)
                ]
                slices["gt_cost"] = gt_cost_viz

        if include_diff:
            sdf = sdf.reshape(*grid_shape)
            diff = np.abs(gt_sdf - sdf)
            diff = diff.reshape(self.grid_dim, -1)
            diff_viz = imgviz.depth2rgb(diff, min_value=0., max_value=0.5)
            diff_viz = diff_viz.reshape(-1, 3)
            viz = np.full(diff_viz.shape, 255, dtype=np.uint8)

            # mask invisible region
            # if self.incremental:
            #     frame_ixs = np.arange(int(self.tot_step_time * self.fps))
            #     sample = self.cached_dataset[frame_ixs]
            # else:
            #     sample = self.cached_dataset.get_all()
            # depth_batch = sample["depth"]
            # T_WC_batch = sample["T"]
            # for i in range(len(frames_data.T_WC_batch)):
            #     T_WC = frames_data.T_WC_batch[i].cpu().numpy()
            #     depth = frames_data.depth_batch_np[i]
            #     visible = geometry.frustum.is_visible(
            #         pc.reshape(-1, 3), T_WC, depth,
            #         self.H, self.W, self.fx, self.fy, self.cx, self.cy,
            #         trunc=self.trunc_distance)
            #     viz[visible] = diff_viz[visible]

            viz = viz.reshape(*grid_shape, 3)
            viz = [
                cv2.resize(np.take(viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["diff"] = viz

        if draw_cams:  # Compute camera markers
            cam_xyz = self.frames.T_WC_batch[:, :3, 3].cpu()
            cam_td = self.to_topdown(cam_xyz, im_size)

            cam_rots = self.frames.T_WC_batch[:, :3, :3].cpu().numpy()
            angs = []
            for rot in cam_rots:
                ang = np.arctan2(rot[0, 2], rot[0, 0])
                # y = - np.sign(range_dim0) * rot[axis_dim0, 2]
                # x = - np.sign(range_dim1) * rot[axis_dim1, 2]
                # ang = np.arctan2(x, y)
                angs.append(ang)

            # Add cam markers to predicted sdf slices
            for i, im in enumerate(slices["pred_sdf"]):
                if self.incremental:
                    trajectory_gt = self.frames.T_WC_batch_np[:, :3, 3]
                    if self.frames.T_WC_gt is not None:
                        trajectory_gt = self.frames.T_WC_gt[:, :3, 3]
                    traj_td = self.to_topdown(trajectory_gt, im_size)
                    for j in range(len(traj_td) - 1):
                        if not (traj_td[j] == traj_td[j + 1]).all():
                            im = im.astype(np.uint8) / 255
                            im = cv2.line(
                                im,
                                traj_td[j][::-1],
                                traj_td[j + 1][::-1],
                                [1., 0., 0.], 2)
                            im = (im * 255).astype(np.uint8)
                for (p, ang) in zip(cam_td, angs):
                    draw.draw_agent(
                        im, p, agent_rotation=ang, agent_radius_px=12)
                slices["pred_sdf"][i] = im

        return slices

    def write_slices(
        self, save_path, prefix="", n_slices=6,
        include_gt=False, include_diff=False, include_chomp=False,
        draw_cams=False, sdf_range=[-2, 2],
    ):
        slices = self.compute_slices(
            z_ixs=None,
            n_slices=n_slices,
            include_gt=include_gt,
            include_diff=include_diff,
            include_chomp=include_chomp,
            draw_cams=draw_cams,
            sdf_range=sdf_range,
        )

        for s in range(n_slices):
            cv2.imwrite(
                os.path.join(save_path, prefix + f"pred_{s}.png"),
                slices["pred_sdf"][s][..., ::-1])
            if include_gt:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"gt_{s}.png"),
                    slices["gt_sdf"][s][..., ::-1])
            if include_diff:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"diff_{s}.png"),
                    slices["diff"][s][..., ::-1])
            if include_chomp:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"pred_cost_{s}.png"),
                    slices["pred_cost"][s][..., ::-1])
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"gt_cost_{s}.png"),
                    slices["gt_cost"][s][..., ::-1])

    def slices_vis(self, n_slices=6):
        slices = self.compute_slices(
            z_ixs=None,
            n_slices=n_slices,
            include_gt=True,
            include_diff=True,
            include_chomp=False,
            draw_cams=True,
        )

        gt_sdf = np.hstack((slices["gt_sdf"]))
        pred_sdf = np.hstack((slices["pred_sdf"]))
        diff = np.hstack((slices["diff"]))

        viz = np.vstack((gt_sdf, pred_sdf, diff))
        return viz

    def to_topdown(self, pts, im_size):
        cam_homog = np.concatenate(
            [pts, np.ones([pts.shape[0], 1])], axis=-1)
        inv_bt = np.linalg.inv(self.bounds_transform_np)
        cam_td = np.matmul(cam_homog, inv_bt.T)
        cam_td = cam_td[:, :3] / self.scene_scale.cpu().numpy()
        cam_td = cam_td / 2 + 0.5  # [-1, 1] -> [0, 1]
        cam_td = np.concatenate((
            cam_td[:, :self.up_ix], cam_td[:, self.up_ix + 1:]), axis=1)
        cam_td = cam_td * im_size
        cam_td = cam_td.astype(int)

        return cam_td

    def obj_slices_vis(self, n_slices=6):
        if self.obj_bounds_file is not None:
            up_ix = 1
            obj_bounds = metrics.get_obj_eval_bounds(
                self.obj_bounds_file, up_ix)

            cmap = sdf_util.get_colormap(sdf_range=[-0.5, 0.5])
            all_slices = []

            for bounds in obj_bounds:
                dims = [256, 256, 256]
                dims[up_ix] = n_slices
                x = torch.linspace(bounds[0, 0], bounds[1, 0], dims[0])
                y = torch.linspace(bounds[0, 1], bounds[1, 1], dims[1])
                z = torch.linspace(bounds[0, 2], bounds[1, 2], dims[2])
                xx, yy, zz = torch.meshgrid(x, y, z)
                pc = torch.cat(
                    (xx[..., None], yy[..., None], zz[..., None]), dim=3
                ).to(self.device)

                sdf = self.sdf_map(pc)
                col = cmap.to_rgba(
                    sdf.detach().cpu().numpy().flatten(),
                    alpha=1., bytes=False)
                col = (col * 255).astype(np.uint8)[..., :3]
                col = col.reshape(*pc.shape[:-1], 3)
                col = np.hstack([col[:, i] for i in range(n_slices)])

                gt_sdf = sdf_util.eval_sdf_interp(
                    self.gt_sdf_interp, pc.cpu(), handle_oob='fill')
                gt_col = cmap.to_rgba(gt_sdf.flatten(), alpha=1., bytes=False)
                gt_col = gt_col.reshape(*pc.shape[:-1], 4)
                gt_col = (gt_col * 255).astype(np.uint8)[..., :3]
                gt_col = np.hstack([gt_col[:, i] for i in range(n_slices)])

                slices = np.vstack((col, gt_col))

                all_slices.append(slices)

            return np.vstack((all_slices))
        return None

    # Evaluation methods ------------------------------------

    def eval_sdf(self, samples=200000, visible_region=True):
        """ If visible_region is True then choose random samples along rays
            in the frames. Otherwise choose random samples in the volume
            where the GT sdf is defined.
        """
        # start, end = start_timing()

        if visible_region:
            sdf, eval_pts = self.eval_sdf_visible(samples)
        else:
            sdf, eval_pts = self.eval_sdf_volume(samples)

        gt_sdf, valid_mask = sdf_util.eval_sdf_interp(
            self.gt_sdf_interp, eval_pts.cpu().detach().numpy(),
            handle_oob='mask')
        # gt sdf gives value 0 inside the walls. Don't include this in loss
        valid_mask = np.logical_and(gt_sdf != 0., valid_mask)

        gt_sdf = gt_sdf[valid_mask]
        sdf = sdf[valid_mask]
        gt_sdf = torch.from_numpy(gt_sdf).to(self.device)

        with torch.set_grad_enabled(False):

            sdf_diff = sdf - gt_sdf
            sdf_diff = torch.abs(sdf_diff)
            l1_sdf = sdf_diff.mean()

            bins_loss = metrics.binned_losses(sdf_diff, gt_sdf)

            # chomp cost difference
            epsilons = [1., 1.5, 2.]
            l1_chomp_costs = [
                torch.abs(
                    metrics.chomp_cost(sdf, epsilon=epsilon) -
                    metrics.chomp_cost(gt_sdf, epsilon=epsilon)
                ).mean().item() for epsilon in epsilons
            ]

        # eval_time = end_timing(start, end)

        res = {
            'av_l1': l1_sdf.item(),
            'binned_l1': bins_loss,
            'l1_chomp_costs': l1_chomp_costs,
        }

        return res

    def eval_sdf_visible(self, samples=20000):
        if self.incremental:
            frame_ixs = np.arange(int(self.tot_step_time * self.fps))
            sample = self.cached_dataset[frame_ixs]
        else:
            sample = self.cached_dataset.get_all()
        depth_batch = torch.FloatTensor(sample["depth"]).to(self.device)
        T_WC_batch = torch.FloatTensor(sample["T"]).to(self.device)

        rays_per_frame = samples // depth_batch.shape[0]
        dist_behind_surf = self.dist_behind_surf
        if self.dataset_format == "ScanNet":
            # For scanNet only evaluate in visible region
            dist_behind_surf == 0

        sample_pts = self.sample_points(
            depth_batch, T_WC_batch,
            n_rays=rays_per_frame, dist_behind_surf=dist_behind_surf,
            n_strat_samples=1, n_surf_samples=0)

        pc = sample_pts["pc"]
        with torch.set_grad_enabled(False):
            sdf = self.sdf_map(pc, noise_std=0)

        sdf = sdf.flatten()
        eval_pts = pc.squeeze()

        # # Check evaluation points look correct
        # scene = self.draw_3D(
        #     False, False,
        #     self.frames.T_WC_batch_np,
        #     draw_cameras=True,
        #     show_gt_mesh=True)
        # pc = trimesh.PointCloud(eval_pts.cpu().numpy(), [128, 128, 128, 255])
        # scene.add_geometry(pc)
        # scene.show()

        return sdf, eval_pts

    def eval_sdf_volume(self, samples=20000):
        """ Sample random points in gt sdf grid volume
        """
        eval_pts = torch.rand(samples, 3)
        eval_pts = eval_pts * (self.sdf_dims - 1)
        eval_pts = eval_pts * self.sdf_transform[0, 0]
        eval_pts = eval_pts + self.sdf_transform[:3, 3]

        # Filter points for evaluation
        if self.dataset_format == "replicaCAD":
            if self.stage_sdf_interp is None:
                stage_sdf = np.load(self.stage_sdf_file)
                transf = np.loadtxt(self.sdf_transf_file)
                self.stage_sdf_interp = sdf_util.sdf_interpolator(
                    stage_sdf, transf)

            eval_stage_sdf = self.stage_sdf_interp(eval_pts)

            # discard_pts = eval_pts[eval_stage_sdf <= 0]
            eval_pts = eval_pts[eval_stage_sdf > 0]

            min_xy = np.loadtxt(self.seq_dir + 'bounds.txt')
            islands = np.loadtxt(self.seq_dir + 'unnavigable.txt')
            px = torch.floor((eval_pts[:, 0] - min_xy[0]) / min_xy[2])
            py = torch.floor((eval_pts[:, 2] - min_xy[1]) / min_xy[2])
            px = torch.clamp(px, min=0, max=islands.shape[1] - 1).int()
            py = torch.clamp(py, min=0, max=islands.shape[0] - 1).int()

            # discard2_pts = eval_pts[islands[py, px] == 1]
            eval_pts = eval_pts[islands[py, px] == 0]

        with torch.set_grad_enabled(False):
            eval_pts = eval_pts.float().to(self.device)

            sdf = self.sdf_map(eval_pts)
            sdf = torch.squeeze(sdf)

        # # Vis evaluation points
        # mesh_gt = trimesh.load(self.scene_file)
        # scene = trimesh.Scene(mesh_gt)
        # pc = trimesh.PointCloud(eval_pts.cpu().numpy(), [0, 255, 0, 255])
        # pc1 = trimesh.PointCloud(discard_pts.cpu().numpy(), [255, 0, 0, 255])
        # pc2 = trimesh.PointCloud(discard2_pts.cpu().numpy(), [255, 0, 0, 255])
        # scene.add_geometry([pc, pc1, pc2])
        # scene.show()

        return sdf, eval_pts

    def eval_object_sdf(self, samples=10000):
        # Evaluate SDF around object if object centric sequence
        errors = None
        if self.obj_bounds_file is not None:
            obj_bounds = metrics.get_obj_eval_bounds(
                self.obj_bounds_file, self.up_ix)
            obj_bounds = torch.FloatTensor(obj_bounds).to(self.device)

            # is object visible yet
            offsets = torch.rand(100, 3).to(self.device)
            extents = obj_bounds[:, 1] - obj_bounds[:, 0]
            pts = obj_bounds[:, 0] + offsets[:, None] * extents

            if self.incremental:
                frame_ixs = np.arange(int(self.tot_step_time * self.fps))
                sample = self.cached_dataset[frame_ixs]
            else:
                sample = self.cached_dataset.get_all()
            depth_batch = torch.FloatTensor(sample["depth"]).to(self.device)
            T_WC_batch = torch.FloatTensor(sample["T"]).to(self.device)

            visible = geometry.frustum.is_visible_torch(
                pts.view(-1, 3), T_WC_batch, depth_batch,
                self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                trunc=0.05)
            visible = visible.detach().cpu().numpy().sum(axis=0) > 0
            visible = visible.reshape(100, len(obj_bounds))
            visible_prop = visible.sum(axis=0) / 100
            visible = visible_prop > 0.5

            errors = []
            for i in range(len(obj_bounds)):
                if visible[i]:
                    offsets = torch.rand(samples, 3).to(self.device)
                    bounds = obj_bounds[i]
                    extents = bounds[1] - bounds[0]
                    pts = bounds[0] + offsets * extents[None, :]

                    gt_sdf, valid_mask = sdf_util.eval_sdf_interp(
                        self.gt_sdf_interp, pts.cpu().numpy(),
                        handle_oob='mask')

                    with torch.set_grad_enabled(False):
                        sdf = self.sdf_map(pts)
                        sdf = torch.squeeze(sdf)

                    gt_sdf = gt_sdf[valid_mask]
                    sdf = sdf[valid_mask].cpu().numpy()
                    l1 = np.abs(gt_sdf - sdf).mean()
                    errors.append(l1)
                else:
                    errors.append(np.nan)

        return errors

    def eval_traj_cost(self, t_ahead=5.):
        """
        Evaluate the SDF along the future trajectory of the sequence.
        """
        if self.traj_file:
            traj = np.loadtxt(self.traj_file)

            traj_start_ix = self.tot_step_time * 30
            traj_end_ix = (self.tot_step_time + t_ahead) * 30
            traj_end_ix = min(len(traj) - 1, traj_end_ix)

            traj_section = traj[int(traj_start_ix): int(traj_end_ix)]
            eval_pts = traj_section[:, [3, 7, 11]]

            # trimesh.PointCloud(eval_pts).show()

            gt_sdf, valid = sdf_util.eval_sdf_interp(
                self.gt_sdf_interp, eval_pts,
                handle_oob='mask')
            valid = np.logical_and(gt_sdf != 0., valid)
            if valid.sum() < (0.9 * valid.shape[0]) or len(traj_section) < 30:
                return np.nan, np.nan

            # Evaluate SDF value at points and sum
            with torch.set_grad_enabled(False):
                eval_pts = torch.FloatTensor(eval_pts).to(self.device)
                sdf = self.sdf_map(eval_pts)
                sdf = sdf.squeeze()

            gt_sdf = gt_sdf[valid]
            sdf = sdf[valid]

            epsilons = [1., 1.5, 2.]
            pred_chomp_costs = [
                metrics.chomp_cost(sdf, epsilon=epsilon).sum().item()
                for epsilon in epsilons
            ]
            gt_chomp_costs = [
                metrics.chomp_cost(gt_sdf, epsilon=epsilon).sum()
                for epsilon in epsilons
            ]

            return pred_chomp_costs, gt_chomp_costs

    def eval_mesh(self, samples=200000):
        # start, end = start_timing()

        mesh_gt = trimesh.load(self.scene_file)
        sdf_mesh = self.mesh_rec()

        acc, comp = metrics.accuracy_comp(
            mesh_gt, sdf_mesh, samples=samples)

        # eval_time = end_timing(start, end)
        return acc, comp

    def sdf_fn(self, pts):
        with torch.set_grad_enabled(False):
            pts = torch.FloatTensor(pts).to(self.device)
            sdf = self.sdf_map(pts)
        return sdf.detach().cpu().numpy()

    def grad_fn(self, pts):
        pts = torch.FloatTensor(pts).to(self.device)
        pts.requires_grad_()
        sdf = self.sdf_map(pts)
        sdf_grad = fc_map.gradient(pts, sdf)

        return sdf_grad.detach().cpu().numpy()

    def eval_fixed(self):
        t = self.eval_times.pop(0)
        return eval_pts.fixed_pts_eval(
            self.sdf_fn, t, self.eval_pts_dir,
            self.seq_dir, self.dataset_format,
            self.cached_dataset, self.dirs_C.cpu(),
            self.gt_sdf_interp, self.eval_pts_root,
            len(self.scene_dataset), grad_fn=self.grad_fn,
        )
