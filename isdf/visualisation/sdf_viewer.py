# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import trimesh
import trimesh.viewer
import pyglet
import threading
import matplotlib.pylab as plt
import torch

from isdf.datasets import sdf_util
from isdf.visualisation import draw3D
from isdf.eval import plot_utils


class PoseViewer(trimesh.viewer.SceneViewer):
    def __init__(self, scene):
        super().__init__(scene, resolution=(1080, 720))

    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """
        magnitude = 10
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()
        elif symbol == pyglet.window.key.A:
            self.toggle_axis()
        elif symbol == pyglet.window.key.G:
            self.toggle_grid()
        elif symbol == pyglet.window.key.Q:
            self.on_close()
        elif symbol == pyglet.window.key.M:
            self.maximize()
        elif symbol == pyglet.window.key.F:
            self.toggle_fullscreen()
        elif symbol == pyglet.window.key.P:
            print(self.scene.camera_transform)

        if symbol in [
                pyglet.window.key.LEFT,
                pyglet.window.key.RIGHT,
                pyglet.window.key.DOWN,
                pyglet.window.key.UP]:
            self.view['ball'].down([0, 0])
            if symbol == pyglet.window.key.LEFT:
                self.view['ball'].drag([-magnitude, 0])
            elif symbol == pyglet.window.key.RIGHT:
                self.view['ball'].drag([magnitude, 0])
            elif symbol == pyglet.window.key.DOWN:
                self.view['ball'].drag([0, -magnitude])
            elif symbol == pyglet.window.key.UP:
                self.view['ball'].drag([0, magnitude])
            self.scene.camera_transform[...] = self.view['ball'].pose


class SDFViewer(trimesh.viewer.SceneViewer):
    def __init__(self,
                 sdf_pc=None,
                 sdf_grid=None, grid2world=None,
                 sdf_grid_pc=None,
                 mesh=None,
                 scene=None,
                 colormap=True,
                 sdf_range=None,
                 open_window=True,
                 surface_cutoff=0.05,
                 save_dir=None,
                 wireframe=False,
                 poses=None,
                 checkpts_dir=None,
                 gt_mesh=None,
                 ray_lens=None,
                 ray_origins=None,
                 ray_dirs_W=None):
        """
            Class for visualisating SDFs.

            Can view mutliple SDFs side by side if input with sdf_grid_pc.

            If colormap is True then displays sdf colormap,
            otherwise green for free space and red for occupied.
        """
        if sdf_pc is not None:
            assert sdf_pc.ndim == 2, "SDF pointcloud must have shape n x 4"
            assert sdf_pc.shape[1] == 4, "SDF pointcloud must have shape n x 4"
            self.sdf_format = 'pc'
            self.sdf_pc = sdf_pc
            self.zs = np.unique(sdf_pc[:, 2])
            if len(self.zs) > 500:
                self.zs = np.linspace(
                    sdf_pc[:, 2].min(), sdf_pc[:, 2].max(), 40)
                step_size = self.zs[1] - self.zs[0]
                sdf_pc[:, 2] -= (sdf_pc[:, 2] - self.zs[0]) % step_size
                self.zs = np.unique(sdf_pc[:, 2])

        elif sdf_grid_pc is not None:
            assert sdf_grid_pc.ndim == 4, \
                "SDF grid pointcloud must have shape (m x n x p x 4+)"
            assert sdf_grid_pc.shape[-1] >= 4, \
                "SDF grid pointcloud must have shape (m x n x p x 4+)"
            self.sdf_format = 'grid_pc'
            self.sdf_grid_pc = sdf_grid_pc
            self.dims = sdf_grid_pc.shape[:-1]
            self.zs = np.arange(sdf_grid_pc.shape[2])
            self.n_grids = sdf_grid_pc.shape[-1] - 3

        elif sdf_grid is not None and grid2world is not None:
            assert sdf_grid.ndim == 3, \
                "SDF grid must have 3 dims"
            self.sdf_format = 'grid'
            self.sdf_grid = sdf_grid
            self.grid2world = grid2world
            self.dims = sdf_grid.shape

            x = np.arange(sdf_grid.shape[0])
            y = np.arange(sdf_grid.shape[1])
            z = np.arange(sdf_grid.shape[2])
            x = x * grid2world[0, 0] + grid2world[0, 3]
            y = y * grid2world[1, 1] + grid2world[1, 3]
            self.zs = z * grid2world[2, 2] + grid2world[2, 3]
            xx, yy = np.meshgrid(x, y, indexing='ij')
            self.xx = xx
            self.yy = yy

        else:
            assert False, "Must provide either \
                    (1) SDF pointcloud (n x 4), \
                    (2) grid pointcloud (m x n x p x 4) or \
                    (3) voxel grid (m x n x p) with a transform."

        self.z_ix = 0
        self.lock = threading.Lock()

        # Colormap
        self.colormap = colormap
        if self.colormap:
            if sdf_range is None:
                if self.sdf_format == 'pc':
                    sdf_range = [sdf_pc[..., 3].min(), sdf_pc[..., 3].max()]
                elif self.sdf_format == 'grid_pc':
                    sdf_range = [sdf_grid_pc[..., 3].min(),
                                 sdf_grid_pc[..., 3].max()]
                else:
                    sdf_range = [sdf_grid.min(), sdf_grid.max()]
            print("sdf_range:", sdf_range)
            self.colormap_fn = sdf_util.get_colormap(sdf_range, surface_cutoff)

        # self.lim_ix = 0
        # self.limits = np.concatenate((
        #     np.linspace(self.sdf_grid_pc.min(), sdf_range[0], 50),
        #     np.linspace(sdf_range[0], sdf_range[1], 200)
        # ))
        self.pose_ix = 0
        self.poses = poses
        self.checkpts_dir = checkpts_dir
        self.gt_mesh = gt_mesh
        # self.limits = np.linspace(sdf_range[0], sdf_range[1], 200)
        self.ray_origins = ray_origins
        self.ray_dirs_W = ray_dirs_W
        self.ray_lens = ray_lens

        # Scene
        if scene is None:
            if mesh is not None:
                if isinstance(mesh, trimesh.Scene):
                    scene = mesh
                else:
                    # mesh.visual.face_colors[:, 3] = 130  # reduce alpha
                    scene = trimesh.Scene(mesh)
            else:
                scene = trimesh.Scene(trimesh.creation.axis())

        if self.sdf_format == "grid_pc":
            if self.n_grids > 1:
                T_extent, extents = trimesh.bounds.oriented_bounds(scene)
                ix = np.argsort(extents)[1]  # Along second shortest axis
                self.offset = T_extent[ix, :3] * extents[ix] * 1.5

                for i in range(1, self.n_grids):
                    shift = i * self.offset
                    offset_meshes = []
                    for g in scene.geometry:
                        m = scene.geometry[g].copy().apply_translation(shift)
                        offset_meshes.append(m)
                    scene.add_geometry(offset_meshes)

        # callback = None
        # if save_dir is not None:
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     self.save_dir = save_dir
        #     callback = self.update_view

        callback = self.next_slice
        self.play_slices = True
        self.z_step = 1
        self.depth_pc = scene.geometry['depth_pc']
        scene.delete_geometry('depth_pc')

        if open_window:
            print(
                "\n\n"
                "Opening iSDF 3D visualization window."
                "\nUse the following keys to change the 3D visualization:"
                "\n- SPACE , pauses / plays the cycling of the SDF slices. "
                "\n- l , toggles the SDF slices."
                "\n- m , toggles the surface mesh (obtained by running marching cubes on the zero level set of the reconstructed SDF)."
                "\n- p , toggles the point cloud of the scene (obtained by backprojecting the depth image)."
                "\n\n"
            )
            super().__init__(
                scene,
                callback=callback,
                resolution=(1080, 720)
            )
            if wireframe:
                self.toggle_wireframe()

    def get_slice_pc(self):
        if self.sdf_format == 'pc':
            sdf_slice_pc = self.sdf_pc[self.sdf_pc[:, 2] == self.zs[self.z_ix]]
        elif self.sdf_format == 'grid_pc':
            full_slice = self.sdf_grid_pc[:, :, self.z_ix]
            sdf_slice_pc = np.repeat(
                full_slice[None, ..., :4], self.n_grids, axis=0)
            sdf_slice_pc[:, :, :, 3] = full_slice[
                ..., -self.n_grids:].transpose(2, 0, 1)
            for i in range(1, self.n_grids):
                shift = i * self.offset
                sdf_slice_pc[i, :, :, :3] += shift
            sdf_slice_pc = sdf_slice_pc.reshape(-1, 4)
        else:
            zz = np.full(self.xx.shape, self.zs[self.z_ix])
            sdf_slice = self.sdf_grid[..., self.z_ix]
            sdf_slice_pc = np.concatenate(
                (self.xx[..., None], self.yy[..., None],
                 zz[..., None], sdf_slice[..., None]),
                axis=-1)
            sdf_slice_pc = sdf_slice_pc.reshape([-1, 4])

        self.z_ix += self.z_step

        if not self.colormap:
            # red for inside, green for outside
            col = np.full(sdf_slice_pc.shape, np.array([1., 0., 0., 1.]))
            # col[sdf_slice_pc[:, 3] > 0] = np.array([0., 1., 0., 1.])
            col[sdf_slice_pc[:, 3] == 0] = np.array([0., 1., 0., 1.])
            col[sdf_slice_pc[:, 3] > 0] = np.array([0., 0., 1., 1.])

        else:
            col = self.colormap_fn.to_rgba(
                sdf_slice_pc[:, 3], alpha=1., bytes=False)

        return sdf_slice_pc, col

    def add_slice_pc(self):
        with self.lock:
            sdf_slice_pc, col = self.get_slice_pc()

            pc = trimesh.PointCloud(sdf_slice_pc[:, :3], col)
            self.scene.add_geometry(pc, geom_name="pc")
            self._update_vertex_list()

    def next_slice(self, scene):
        if self.play_slices:
            if self.z_ix == len(self.zs) - 1:
                self.z_step = -1
            if self.z_ix == 0:
                self.z_step = 1

            with self.lock:
                self.scene.delete_geometry('pc')
            self.add_slice_pc()

    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """
        magnitude = 10
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()
        elif symbol == pyglet.window.key.A:
            self.toggle_axis()
        elif symbol == pyglet.window.key.G:
            self.toggle_grid()
        elif symbol == pyglet.window.key.Q:
            self.on_close()
        elif symbol == pyglet.window.key.M:
            # self.maximize()
            # toggle mesh
            if 'rec_mesh' in self.scene.geometry:
                self.rec_mesh = self.scene.geometry['rec_mesh']
                self.scene.delete_geometry('rec_mesh')
            else:
                self.scene.add_geometry(self.rec_mesh, geom_name='rec_mesh')
        elif symbol == pyglet.window.key.L:
            # toggle sdf slice
            if 'pc' in self.scene.geometry:
                self.scene.delete_geometry('pc')
            else:
                self.add_slice_pc()
        elif symbol == pyglet.window.key.F:
            self.toggle_fullscreen()
        elif symbol == pyglet.window.key.S:
            if self.z_ix < len(self.zs):
                print("Swapping pc")
                with self.lock:
                    self.scene.delete_geometry('pc')
                self.add_slice_pc()
            else:
                print("No more slices")
        elif symbol == pyglet.window.key.O:
            self.save_image(self.save_dir + f'/{self.z_ix:04d}.png')
            print("saved im at z_ix", self.z_ix)
        elif symbol == pyglet.window.key.T:
            print(self.scene.camera_transform)
        elif symbol == pyglet.window.key.P:
            # toggle point cloud
            if 'depth_pc' in self.scene.geometry:
                self.scene.delete_geometry('depth_pc')
            else:
                self.scene.add_geometry(self.depth_pc, geom_name='depth_pc')
        elif symbol == pyglet.window.key.SPACE:
            self.play_slices = not self.play_slices

        if symbol in [
                pyglet.window.key.LEFT,
                pyglet.window.key.RIGHT,
                pyglet.window.key.DOWN,
                pyglet.window.key.UP]:
            self.view['ball'].down([0, 0])
            if symbol == pyglet.window.key.LEFT:
                self.view['ball'].drag([-magnitude, 0])
            elif symbol == pyglet.window.key.RIGHT:
                self.view['ball'].drag([magnitude, 0])
            elif symbol == pyglet.window.key.DOWN:
                self.view['ball'].drag([0, -magnitude])
            elif symbol == pyglet.window.key.UP:
                self.view['ball'].drag([0, magnitude])
            self.scene.camera_transform[...] = self.view['ball'].pose

    def save_slice_imgs(self, direc):
        if not os.path.exists(direc):
            os.makedirs(direc)

        if self.sdf_format == 'grid' or self.sdf_format == 'grid_pc':
            self.z_ix = 0
            print("Num slices", len(self.zs))
            for i in range(len(self.zs)):
                pc, col = self.get_slice_pc()

                pc = pc.reshape(self.dims[0], self.dims[1], 4)
                col = col.reshape(self.dims[0], self.dims[1], 4)

                fname = os.path.join(direc, f'{i:03d}.png')
                save_slice(fname, pc, col, self.colormap_fn)

    def pose_interp(self, scene):
        if self.pose_ix < len(self.poses):
            self.save_image(self.save_dir + f'/{self.pose_ix:04d}.png')
            print("saved im ", self.pose_ix, " / ", len(self.poses))

            with self.lock:
                self.scene.camera_transform = self.poses[self.pose_ix]
            self.pose_ix += 1

    def save_horizontal_slice(self, scene):
        if self.z_ix < len(self.zs):
            self.save_image(self.save_dir + f'/{self.z_ix:04d}.png')
            print("saved im ", self.z_ix, " / ", len(self.zs))
            with self.lock:
                self.scene.delete_geometry('pc')
            self.add_slice_pc()

    def save_plane(self, scene):
        if self.z_ix < len(self.zs):
            self.save_image(self.save_dir + f'/{self.z_ix:04d}.png')
            print("saved im ", self.z_ix, " / ", len(self.zs))

            pc = self.sdf_grid_pc[:, :, self.z_ix, :3]
            vertices = np.array([pc[0,0], pc[0, -1], pc[-1, 0], pc[-1, -1]])
            faces = np.array([[0,1,2], [1,2,3], [0,2,1], [1,3,2]])
            plane = trimesh.Trimesh(vertices=vertices, faces=faces)
            plane.visual.face_colors[:] = [170, 170, 170, 110]
            self.z_ix += 1

            with self.lock:
                self.scene.delete_geometry('pc')
                self.scene.delete_geometry('plane')
                self.scene.add_geometry(plane, geom_name='plane')

    def project_rays(self, scene):
        if self.z_ix < len(self.ray_lens):
            self.save_image(self.save_dir + f'/{self.z_ix:04d}.png')
            print("saved im ", self.z_ix, " / ", len(self.ray_lens))

            ends = self.ray_origins + self.ray_lens[self.z_ix] * self.ray_dirs_W
            lines = torch.cat(
                (self.ray_origins[:, None, :], ends[:, None, :]), dim=1)
            rays = trimesh.load_path(lines.cpu())
            rays.colors = [[100, 100, 100, 100]] * len(rays.entities)

            self.z_ix += 1

            with self.lock:
                self.scene.delete_geometry('rays')
                self.scene.add_geometry(rays, geom_name='rays')

    def update_view(self, scene):

        if self.z_ix < len(self.poses):
            self.save_image(self.save_dir + f'/{self.z_ix:04d}.png')
            print("saved im", self.z_ix, " / ", len(self.poses))

            with self.lock:
                scene.camera_transform = self.poses[self.z_ix]

            self.z_ix += 1

    def save_level_sets(self, scene):

        if self.lim_ix < len(self.limits):
            self.save_image(self.save_dir + f'/{self.lim_ix:04d}.png')
            print("saved im for limit", self.limits[self.lim_ix],
                  self.lim_ix, " / ", len(self.limits))
            keep = self.sdf_grid_pc[..., 3] > self.limits[self.lim_ix]
            pts = self.sdf_grid_pc[keep]
            with self.lock:
                self.scene.delete_geometry('pc')
                if len(pts) > 0:
                    col = self.colormap_fn.to_rgba(
                        pts[:, 3], alpha=1., bytes=False)
                    pc = trimesh.PointCloud(pts[:, :3], col)
                    self.scene.add_geometry(pc, geom_name="pc")
                self._update_vertex_list()

            self.lim_ix += 1

    def save_seq(self, scene):
        if self.pose_ix < len(self.poses):
            self.save_image(self.save_dir + f'/{self.pose_ix:04d}.png')
            print("saved im for t", self.pose_ix, " / ", len(self.poses))
            with self.lock:
                # new camera
                self.scene.delete_geometry('cam')
                camera = trimesh.scene.Camera(
                    fov=self.scene.camera.fov,
                    resolution=self.scene.camera.resolution)
                marker = draw3D.draw_camera(
                    camera, self.poses[self.pose_ix],
                    color=(0., 1., 0., 0.8), marker_height=0.5)
                self.scene.add_geometry(marker[1], geom_name='cam')

                # new sdf slices
                # t = self.pose_ix / 30
                # chkpt_path = self.checkpts_dir + f"/step_{t:.3f}.pth"
                # if os.path.exists(chkpt_path) and self.pose_ix > 410:
                #     print("replacing sdf slice at t =", t)
                #     sdf_map = plot_utils.load_model(
                #         chkpt_path, self.gt_mesh, "cuda")
                #     pc = torch.FloatTensor(self.sdf_grid_pc[..., :3]).cuda()
                #     pc = pc.reshape(-1, 3)
                #     with torch.set_grad_enabled(False):
                #         print(pc.shape)
                #         sdf = fc_map.chunks(pc, 200000, sdf_map)

                #     col = self.colormap_fn.to_rgba(
                #         sdf.cpu().numpy(), alpha=1., bytes=False)
                #     col[:, 3] = 0.7
                #     pc = trimesh.PointCloud(pc.cpu().numpy(), col)
                #     self.scene.add_geometry(pc, geom_name="pc")

            self.pose_ix += 1

def save_slice(fname, pc, col, colormap_fn):
    col = col.transpose(1, 0, 2)

    extent = [0, np.linalg.norm(pc[0, 0, :2] - pc[-1, 0, :2]),
              0, np.linalg.norm(pc[0, 0, :2] - pc[0, -1, :2])]

    plt.imshow(col, extent=extent)
    plt.colorbar(colormap_fn)
    plt.title(f"z = {pc[0, 0, 2]:.3f}")
    plt.savefig(fname)
    plt.close()
