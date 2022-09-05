import numpy as np
import threading
import time
import imgviz
import cv2
import skimage.measure
import matplotlib.pylab as plt

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from isdf import geometry
from isdf.datasets import sdf_util


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


class iSDFWindow:

    def __init__(self, trainer, optim_iter, font_id):
        self.trainer = trainer
        self.optim_iter = optim_iter

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            trainer.W, trainer.H, trainer.fx, trainer.fy, trainer.cx, trainer.cy)

        self.window = gui.Application.instance.create_window('iSDF viewer', 1480, 900)

        if trainer.live:
            mode = "live"
        elif trainer.incremental:
            mode = "incremental"
        else:
            mode = "batch"

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Application control
        button_play_pause = gui.ToggleSwitch('Resume/Pause')
        button_play_pause.set_on_clicked(self._on_switch)

        self.training_iters_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, em, 0))
        steps_label = gui.Label('Training iters per step')
        self.training_iters_slider = gui.Slider(gui.Slider.INT)
        self.training_iters_slider.set_limits(1, 50)
        self.training_iters_slider.int_value = 5
        self.training_iters_grid.add_child(steps_label)
        self.training_iters_grid.add_child(self.training_iters_slider)

        button_clear_kf = None
        if trainer.incremental or trainer.live:
            button_clear_kf = gui.Button('Clear Keyframes')
            button_clear_kf.horizontal_padding_em = 0
            button_clear_kf.vertical_padding_em = 0.1
            button_clear_kf.set_on_clicked(self._clear_keyframes)

        self.button_compute_mesh = gui.Button('Recompute mesh')
        self.button_compute_mesh.set_on_clicked(self._recompute_mesh)
        self.button_compute_mesh.horizontal_padding_em = 0
        self.button_compute_mesh.vertical_padding_em = 0.1
        self.button_compute_mesh.enabled = False

        self.button_compute_slices = gui.Button('Recompute SDF slices')
        self.button_compute_slices.set_on_clicked(self._recompute_slices)
        self.button_compute_slices.horizontal_padding_em = 0
        self.button_compute_slices.vertical_padding_em = 0.1
        self.button_compute_slices.enabled = False

        self.button_compute_renders = gui.Button('Recompute renders')
        self.button_compute_renders.set_on_clicked(self._recompute_renders)
        self.button_compute_renders.horizontal_padding_em = 0
        self.button_compute_renders.vertical_padding_em = 0.1
        self.button_compute_renders.enabled = False

        ### Info panel
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id

        ## Items in vis props
        self.vis_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))

        mesh_label = gui.Label('Mesh reconstruction')
        self.mesh_box = gui.Checkbox('')
        self.mesh_box.checked = True
        self.mesh_box.set_on_checked(self._toggle_mesh)
        self.vis_prop_grid.add_child(mesh_label)
        self.vis_prop_grid.add_child(self.mesh_box)

        interval_label = gui.Label('    Meshing interval steps')
        self.interval_slider = gui.Slider(gui.Slider.INT)
        self.interval_slider.set_limits(20, 500)
        self.interval_slider.int_value = 200
        self.vis_prop_grid.add_child(interval_label)
        self.vis_prop_grid.add_child(self.interval_slider)

        voxel_dim_label = gui.Label('    Meshing voxel grid dim')
        self.voxel_dim_slider = gui.Slider(gui.Slider.INT)
        self.voxel_dim_slider.set_limits(20, 256)
        self.voxel_dim_slider.int_value = self.trainer.grid_dim
        self.voxel_dim_slider.set_on_value_changed(self._change_voxel_dim)
        self.vis_prop_grid.add_child(voxel_dim_label)
        self.vis_prop_grid.add_child(self.voxel_dim_slider)

        crop_label = gui.Label('    Crop near pc')
        self.crop_box = gui.Checkbox('')
        if mode == 'batch':
            self.crop_box.checked = False
        else:
            self.crop_box.checked = True
        self.vis_prop_grid.add_child(crop_label)
        self.vis_prop_grid.add_child(self.crop_box)

        normal_col_label = gui.Label('    Color by normals')
        self.normal_col_box = gui.Checkbox('')
        self.normal_col_box.checked = False
        self.vis_prop_grid.add_child(normal_col_label)
        self.vis_prop_grid.add_child(self.normal_col_box)

        slices_label = gui.Label('SDF slices')
        self.slices_box = gui.Checkbox('')
        self.slices_box.checked = False
        self.slices_box.set_on_checked(self._toggle_slices)
        self.vis_prop_grid.add_child(slices_label)
        self.vis_prop_grid.add_child(self.slices_box)

        slices_interval_label = gui.Label('    Compute interval steps')
        self.slices_interval_slider = gui.Slider(gui.Slider.INT)
        self.slices_interval_slider.set_limits(20, 500)
        self.slices_interval_slider.int_value = 200
        self.vis_prop_grid.add_child(slices_interval_label)
        self.vis_prop_grid.add_child(self.slices_interval_slider)

        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        origin_label = gui.Label('Origin')
        self.origin_box = gui.Checkbox('')
        self.origin_box.checked = True
        self.vis_prop_grid.add_child(origin_label)
        self.vis_prop_grid.add_child(self.origin_box)

        kf_label = gui.Label('Keyframes')
        self.kf_box = gui.Checkbox('')
        self.kf_box.checked = True
        self.vis_prop_grid.add_child(kf_label)
        self.vis_prop_grid.add_child(self.kf_box)

        pc_label = gui.Label('Depth point cloud')
        self.pc_box = gui.Checkbox('')
        self.pc_box.checked = False
        self.vis_prop_grid.add_child(pc_label)
        self.vis_prop_grid.add_child(self.pc_box)

        self.gt_mesh = None
        if self.trainer.gt_scene:
            # mesh_trimesh = trimesh.load(self.trainer.scene_file, process=False)
            self.gt_mesh = o3d.io.read_triangle_mesh(self.trainer.scene_file)
            self.gt_mesh.compute_vertex_normals()
            gt_mesh_label = gui.Label('Ground truth mesh')
            self.gt_mesh_box = gui.Checkbox('')
            self.gt_mesh_box.checked = False
            self.vis_prop_grid.add_child(gt_mesh_label)
            self.vis_prop_grid.add_child(self.gt_mesh_box)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), em, em)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_rgb_depth = gui.ImageWidget()
        self.render_normals_depth = gui.ImageWidget()
        tab1.add_child(gui.Label('Input rgb and depth'))
        tab1.add_child(self.input_rgb_depth)
        tab1.add_fixed(vspacing)

        black_vis = np.full(
            [self.trainer.H, 2 * self.trainer.W, 3], 0, dtype=np.uint8)
        self.no_render = o3d.geometry.Image(black_vis)
        render_label = gui.Label('Rendered normals and depth')
        self.render_box = gui.Checkbox('')
        self.render_box.checked = False
        self.render_box.set_on_checked(self._toggle_render)
        render_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
        render_grid.add_child(render_label)
        render_grid.add_child(self.render_box)
        render_interval_label = gui.Label('    Render interval')
        self.render_interval_slider = gui.Slider(gui.Slider.INT)
        self.render_interval_slider.set_limits(5, 100)
        self.render_interval_slider.int_value = 20
        render_grid.add_child(render_interval_label)
        render_grid.add_child(self.render_interval_slider)
        tab1.add_child(render_grid)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.render_normals_depth)
        tab1.add_fixed(vspacing)
        tab1.add_fixed(vspacing)
        tabs.add_tab('Latest frames', tab1)

        ### Keyframes image tab
        tab2 = gui.Vert(0, tab_margins)
        self.n_panels = 10
        self.keyframe_panels = []
        for _ in range(self.n_panels):
            kf_panel = gui.ImageWidget()
            tab2.add_child(kf_panel)
            self.keyframe_panels.append(kf_panel)
        tabs.add_tab('Keyframes', tab2)

        set_enabled(self.vis_prop_grid, True)
        self.interval_slider.enabled = self.mesh_box.checked
        self.voxel_dim_slider.enabled = self.mesh_box.checked
        self.render_interval_slider.enabled = self.render_box.checked
        self.slices_interval_slider.enabled = self.slices_box.checked

        self.panel.add_child(gui.Label('iSDF controls'))
        self.panel.add_child(gui.Label(f'Operation mode: {mode}'))
        self.panel.add_child(button_play_pause)
        self.panel.add_child(self.training_iters_grid)
        if button_clear_kf is not None:
            self.panel.add_child(button_clear_kf)
        self.panel.add_child(self.button_compute_mesh)
        self.panel.add_child(self.button_compute_slices)
        self.panel.add_child(self.button_compute_renders)
        self.panel.add_fixed(vspacing)
        # self.panel.add_child(gui.Label('Info'))
        self.panel.add_child(self.output_info)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(gui.Label('3D visualisation settings'))
        self.panel.add_child(self.vis_prop_grid)
        self.panel.add_child(tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()

        # timings panel
        self.timings_panel = gui.Vert(spacing, gui.Margins(em, 0.5 * em, em, em))
        self.timings = gui.Label('Compute balance in last 20s:')
        self.timings_panel.add_child(self.timings)

        # colorbar panel
        self.colorbar_panel = gui.Vert(spacing, gui.Margins(0, 0, 0, 0))
        self.sdf_colorbar = gui.ImageWidget()
        self.colorbar_panel.add_child(self.sdf_colorbar)

        self.dialog_panel = gui.Vert(spacing, gui.Margins(em, em, em, em))
        self.dialog = gui.Dialog("Tracking lost!")
        self.dialog.add_child(gui.Label('Tracking lost!'))
        self.dialog_panel.add_child(self.dialog)
        self.dialog_panel.visible = False

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.colorbar_panel)
        w.add_child(self.timings_panel)
        w.add_child(self.dialog_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.is_done = False

        self.is_started = True
        self.is_running = True
        self.is_surface_updated = False
        if self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)

        self.kfs = []
        if mode == 'batch':
            h = int(trainer.frames.im_batch_np[-1].shape[0] / 6)
            w = int(trainer.frames.im_batch_np[-1].shape[1] / 6)
            self.kfs = [
                cv2.resize(kf, (w, h)) for kf in trainer.frames.im_batch_np
            ]
        self.max_points = 100000
        self.kf_panel_size = 3
        self.steps_before_meshing = 49
        self.clear_kf_frustums = False
        self.sdf_grid_pc = None
        self.slice_ix = 0
        self.slice_step = 1
        self.colormap_fn = None
        self.vis_times = []
        self.optim_times = []
        self.prepend_text = ""
        self.latest_mesh = None
        self.latest_pcd = None
        self.latest_frustums = []
        self.T_WC_latest = None

        self.lit_mat = rendering.MaterialRecord()
        self.lit_mat.shader = "defaultLit"
        self.unlit_mat = rendering.MaterialRecord()
        self.unlit_mat.shader = "unlitLine"
        self.unlit_mat.line_width = 5.0

        self.cam_scale = 0.1 if "franka" in trainer.dataset_format else 0.2

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 23 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y, rect.get_right() - x, rect.height)

        timings_panel_width = 15 * em
        timings_panel_height = 3 * em
        self.timings_panel.frame = gui.Rect(
            rect.get_right() - timings_panel_width,
            rect.y,
            timings_panel_width,
            timings_panel_height
        )

        colorbar_panel_width = 25 * em
        colorbar_panel_height = 4 * em
        self.colorbar_panel.frame = gui.Rect(
            rect.get_right() - colorbar_panel_width,
            rect.get_bottom() - colorbar_panel_height,
            colorbar_panel_width,
            colorbar_panel_height
        )

        dialog_panel_width = 16 * em
        dialog_panel_height = 4 * em
        self.dialog_panel.frame = gui.Rect(
            rect.get_right() // 2 - dialog_panel_width + panel_width // 2,
            rect.get_bottom() // 2 - dialog_panel_height,
            dialog_panel_width,
            dialog_panel_height
        )

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running
        self.button_compute_slices.enabled = not self.button_compute_slices.enabled
        self.button_compute_mesh.enabled = not self.button_compute_mesh.enabled
        self.button_compute_renders.enabled = not self.button_compute_renders.enabled

    def _recompute_mesh(self):
        self.reconstruct_mesh()
        self.output_info.text = "Recomputed mesh\n\n\n"

    def _recompute_slices(self):
        self.compute_sdf_slices()
        self.output_info.text = "Recomputed slices\n\n\n"

    def _recompute_renders(self):
        self.update_latest_frames(True)
        self.output_info.text = "Recomputed renders\n\n\n"

    def _clear_keyframes(self):
        self.is_started = False
        time.sleep(0.3)
        self.output_info.text = "Clearing keyframes"
        self.trainer.clear_keyframes()
        self.iter = 0
        info, new_kf, end = self.optim_iter(self.trainer, self.iter)
        self.iter += 1
        self.kfs = [new_kf]
        self.output_info.text = f"Iteration {self.iter}\n" + info
        self.clear_kf_frustums = True
        self.is_started = True

    def _toggle_mesh(self, is_on):
        if self.mesh_box.checked:
            self.interval_slider.enabled = True
            self.voxel_dim_slider.enabled = True
        else:
            self.widget3d.scene.remove_geometry("rec_mesh")
            self.interval_slider.enabled = False
            self.voxel_dim_slider.enabled = False

    def _toggle_slices(self, is_on):
        if self.slices_box.checked:
            self.slices_interval_slider.enabled = True
            self.sdf_colorbar.update_image(self.colorbar_img)
        else:
            self.widget3d.scene.remove_geometry("sdf_slice")
            self.slices_interval_slider.enabled = False
            self.sdf_colorbar.update_image(self.no_colorbar)

    def _toggle_render(self, is_on):
        if self.render_box.checked is False:
            self.render_normals_depth.update_image(self.no_render)
            self.render_interval_slider.enabled = False
        else:
            self.render_interval_slider.enabled = True

    def _change_voxel_dim(self, val):
        grid_dim = self.voxel_dim_slider.int_value
        grid_pc = geometry.transform.make_3D_grid(
            [-1.0, 1.0],
            grid_dim,
            self.trainer.device,
            transform=self.trainer.bounds_transform,
            scale=self.trainer.scene_scale,
        )
        self.trainer.new_grid_dim = grid_dim
        self.trainer.new_grid_pc = grid_pc.view(-1, 3).to(self.trainer.device)

    # On start: point cloud buffer and model initialization.
    def _on_start(self):

        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((self.max_points, 3), dtype=np.float32)))
        pcd_placeholder.point['colors'] = o3c.Tensor(
            np.zeros((self.max_points, 3), dtype=np.float32))
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)
        self.widget3d.scene.scene.add_geometry('slice_pc', pcd_placeholder, mat)

        if self.trainer.gt_scene:
            if self.gt_mesh_box.checked:
                self.widget3d.scene.add_geometry(
                    'gt_mesh', self.gt_mesh, self.lit_mat)

        if self.origin_box.checked:
            self.widget3d.scene.add_geometry('origin', self.origin, self.unlit_mat)

        self.is_started = True

    def _on_close(self):
        self.is_done = True
        print('Finished.')
        return True

    def init_render(self):
        self.output_info.text = "\n\n\n"

        blank = np.full([80, 500, 3], 255, dtype=np.uint8)
        self.no_colorbar = o3d.geometry.Image(blank)
        self.colorbar_img = self.no_colorbar
        self.sdf_colorbar.update_image(self.no_colorbar)

        blank = np.full([self.trainer.H, self.trainer.W, 3], 255, dtype=np.uint8)

        blank_im = o3d.geometry.Image(np.hstack([blank] * 2))
        self.input_rgb_depth.update_image(blank_im)
        self.render_normals_depth.update_image(self.no_render)

        kf_im = o3d.geometry.Image(np.hstack([blank] * self.kf_panel_size))
        for panel in self.keyframe_panels:
            panel.update_image(kf_im)

        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def toggle_content(self, name, geometry, mat, show):
        if (self.widget3d.scene.has_geometry(name) is False) and show:
            self.widget3d.scene.add_geometry(name, geometry, mat)
        elif self.widget3d.scene.has_geometry(name) and (show is False):
            self.widget3d.scene.remove_geometry(name)

    def update_render(
        self,
        latest_frame,
        render_frame,
        keyframes_vis,
        rec_mesh,
        slice_pcd,
        pcd,
        latest_frustum,
        kf_frustums,
    ):
        self.input_rgb_depth.update_image(latest_frame)
        if render_frame is not None:
            self.render_normals_depth.update_image(render_frame)

        for im, kf_panel in zip(keyframes_vis, self.keyframe_panels):
            kf_panel.update_image(im)

        self.widget3d.scene.scene.update_geometry(
            "points", pcd, rendering.Scene.UPDATE_POINTS_FLAG |
            rendering.Scene.UPDATE_COLORS_FLAG)            

        self.widget3d.scene.scene.update_geometry(
            "slice_pc", slice_pcd, rendering.Scene.UPDATE_POINTS_FLAG |
            rendering.Scene.UPDATE_COLORS_FLAG)   

        if rec_mesh is not None:
            self.widget3d.scene.remove_geometry("rec_mesh")
            self.widget3d.scene.add_geometry("rec_mesh", rec_mesh, self.lit_mat)

        if latest_frustum is not None:
            self.widget3d.scene.remove_geometry("latest_frustum")
            self.widget3d.scene.add_geometry(
                "latest_frustum", latest_frustum, self.unlit_mat)

        if self.clear_kf_frustums:
            for i in range(50):
                self.widget3d.scene.remove_geometry(f"frustum_{i}")
            self.clear_kf_frustums = False

        for i, frustum in enumerate(kf_frustums):
            self.widget3d.scene.remove_geometry(f"frustum_{i}")
            if frustum is not None:
                self.widget3d.scene.add_geometry(f"frustum_{i}", frustum, self.unlit_mat)

        if self.trainer.gt_scene:
            self.toggle_content('gt_mesh', self.gt_mesh, self.lit_mat, self.gt_mesh_box.checked)
        self.toggle_content('origin', self.origin, self.unlit_mat, self.origin_box.checked)


    # Major loop
    def update_main(self):

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render()
        )

        self.iter = 0
        self.step = 0

        while not self.is_done:
            # if not self.is_started or not self.is_running:
            #     time.sleep(0.05)
            #     continue
            t0 = time.time()

            if self.is_running:

                # training steps ------------------------------
                for i in range(self.training_iters_slider.int_value):
                    info, new_kf, end = self.optim_iter(self.trainer, self.iter)
                    if new_kf is not None:
                        self.kfs.append(new_kf)
                    if end and self.trainer.incremental and not self.trainer.live:
                        self.prepend_text = "SEQUENCE ENDED - CONTINUING TRAINING\n"
                    self.output_info.text = self.prepend_text + \
                        f"Step {self.step} -- Iteration {self.iter}\n" + info
                    self.iter += 1

                t1 = time.time()
                self.optim_times.append(t1 - t0)

                # image vis -----------------------------------

                # keyframe vis
                kf_vis = []
                c = 0
                ims = []
                for im in self.kfs:
                    im = cv2.copyMakeBorder(
                        im, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )
                    ims.append(im)
                    c += 1
                    if c == self.kf_panel_size:
                        kf_im = o3d.geometry.Image(np.hstack(ims))
                        kf_vis.append(kf_im)
                        ims = []
                        c = 0
                blank = np.full(im.shape, 255, dtype=np.uint8)
                if len(ims) != 0:
                    for _ in range(c, 3):
                        ims.append(blank)
                    kf_im = o3d.geometry.Image(np.hstack(ims))
                    kf_vis.append(kf_im)

                # latest frame vis (rgbd and render) --------------------------------
                do_render = False
                if (
                    self.render_box.checked and
                    self.step % self.render_interval_slider.int_value == 0
                ):
                    do_render = True
                self.update_latest_frames(do_render)

                # 3D vis --------------------------------------

                # reconstructed mesh from marching cubes on zero level set
                rec_mesh = None
                if (
                    self.step % self.interval_slider.int_value == 0 and
                    self.mesh_box.checked and
                    self.step > self.steps_before_meshing
                ):
                    self.reconstruct_mesh()
                    rec_mesh = self.latest_mesh

                # sdf slices
                if (
                    self.step % self.slices_interval_slider.int_value == 0 and
                    self.slices_box.checked and
                    self.step > self.steps_before_meshing
                ):
                    self.compute_sdf_slices()

                slice_pcd = self.next_slice_pc()

                # point cloud from depth
                pcd = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
                if self.pc_box.checked:
                    self.compute_depth_pcd()
                    pcd = self.latest_pcd

                # keyframes
                if self.kf_box.checked:
                    self.update_kf_frustums()
                    kf_frustums = self.latest_frustums
                else:
                    kf_frustums = [None] * len(self.latest_frustums)

                # latest frame
                latest_frustum = None
                if self.trainer.incremental or self.trainer.live:
                    latest_frustum = o3d.geometry.LineSet.create_camera_visualization(
                        self.intrinsic.width,
                        self.intrinsic.height,
                        self.intrinsic.intrinsic_matrix,
                        np.linalg.inv(self.T_WC_latest),
                        scale=self.cam_scale,
                    )
                    latest_frustum.paint_uniform_color([0.961, 0.475, 0.000])
                self.latest_frustum = latest_frustum

            else:

                kf_vis = []

                rec_mesh = None
                if self.mesh_box.checked:
                    rec_mesh = self.latest_mesh

                slice_pcd = self.next_slice_pc()

                pcd = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
                if self.pc_box.checked:
                    if self.latest_pcd is None:
                        self.compute_depth_pcd()
                    pcd = self.latest_pcd

                if self.kf_box.checked:
                    kf_frustums = self.latest_frustums
                else:
                    kf_frustums = [None] * len(self.latest_frustums)

                time.sleep(0.05)

            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_render(
                    self.latest_frame,
                    self.render_frame,
                    kf_vis,
                    rec_mesh,
                    slice_pcd,
                    pcd,
                    self.latest_frustum,
                    kf_frustums,
                )
            )

            if self.is_running:
                self.vis_times.append(time.time() - t1)

                t_vis = np.sum(self.vis_times)
                t_optim = np.sum(self.optim_times)
                t_tot = t_vis + t_optim
                prop_vis = int(np.round(100 * t_vis / t_tot))
                prop_optim = int(np.round(100 * t_optim / t_tot))
                while t_tot > 20:
                    self.vis_times.pop(0)
                    self.optim_times.pop(0)
                    t_tot = np.sum(self.vis_times) + np.sum(self.optim_times)

                self.timings.text = "Compute balance in last 20s:\n" +\
                    f"training {prop_optim}% : visualisation {prop_vis}%"

                self.step += 1

        time.sleep(0.5)

    def reconstruct_mesh(self):
        self.output_info.text = "Computing mesh reconstruction with marching cubes\n\n\n"
        rec_mesh = self.trainer.mesh_rec(self.crop_box.checked).as_open3d
        rec_mesh.compute_vertex_normals()
        if self.normal_col_box.checked:
            rec_mesh.vertex_colors = rec_mesh.vertex_normals
        self.latest_mesh = rec_mesh

    def next_slice_pc(self):
        if self.slices_box.checked and self.sdf_grid_pc is not None:
            slice_pc = self.sdf_grid_pc[:, :, self.slice_ix].reshape(-1, 4)
            slice_pcd = o3d.t.geometry.PointCloud(o3c.Tensor(slice_pc[:, :3]))
            slice_cols = self.colormap_fn.to_rgba(slice_pc[:, 3], bytes=False)
            slice_cols = slice_cols[:, :3].astype(np.float32)
            slice_pcd.point['colors'] = o3c.Tensor(slice_cols)
            # next slice
            if self.slice_ix == self.sdf_grid_pc.shape[2] - 1:
                self.slice_step = -1
            if self.slice_ix == 0:
                self.slice_step = 1
            self.slice_ix += self.slice_step
            return slice_pcd
        else:
            return o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))

    def compute_sdf_slices(self):
        self.output_info.text = "Computing SDF slices\n\n\n"
        sdf_grid_pc, _ = self.trainer.get_sdf_grid_pc()
        sdf_grid_pc = np.transpose(sdf_grid_pc, (2, 1, 0, 3))
        self.sdf_grid_pc = sdf_grid_pc
        sdf_range = [self.sdf_grid_pc[..., 3].min(), self.sdf_grid_pc[..., 3].max()]

        self.colormap_fn = sdf_util.get_colormap(sdf_range, 0.02)

        fig, ax = plt.subplots(figsize=(5, 2), tight_layout=True)
        plt.colorbar(self.colormap_fn, ax=ax, orientation='horizontal')
        ax.remove()
        fig.set_tight_layout(True)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data[120:]
        self.colorbar_img = o3d.geometry.Image(data)
        self.sdf_colorbar.update_image(self.colorbar_img)

    def compute_depth_pcd(self):
        T_WC_batch = self.trainer.frames.T_WC_batch_np
        self.trainer.update_vis_vars()
        pcs_cam = geometry.transform.backproject_pointclouds(
            self.trainer.gt_depth_vis, self.trainer.fx_vis, self.trainer.fy_vis,
            self.trainer.cx_vis, self.trainer.cy_vis)
        pcs_cam = np.einsum('Bij,Bkj->Bki', T_WC_batch[:, :3, :3], pcs_cam)
        pcs_world = pcs_cam + T_WC_batch[:, None, :3, 3]
        pcs_world = pcs_world.reshape(-1, 3).astype(np.float32)
        cols = self.trainer.gt_im_vis.reshape(-1, 3)
        cols = cols.astype(np.float32) / 255
        if len(pcs_world) > self.max_points:
            ixs = np.random.choice(
                np.arange(len(pcs_world)), self.max_points, replace=False)
            pcs_world = pcs_world[ixs].astype(np.float32)
            cols = cols[ixs].astype(np.float32)
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(pcs_world))
        pcd.point['colors'] = o3c.Tensor(cols)
        self.latest_pcd = pcd

    def update_kf_frustums(self):
        kf_frustums = []
        T_WC_batch = self.trainer.frames.T_WC_batch_np
        for T_WC in T_WC_batch[:-1]:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.intrinsic.width,
                self.intrinsic.height,
                self.intrinsic.intrinsic_matrix,
                np.linalg.inv(T_WC),
                scale=self.cam_scale,
            )
            frustum.paint_uniform_color([0.000, 0.475, 0.900])
            kf_frustums.append(frustum)
        self.latest_frustums = kf_frustums

    def update_latest_frames(self, do_render):
        rgbd_vis, render_vis, T_WC = self.trainer.latest_frame_vis(do_render)
        self.T_WC_latest = T_WC
        if (T_WC == np.eye(4)).all():
            self.dialog_panel.visible = True
        else:
            self.dialog_panel.visible = False
        latest_frame = o3d.geometry.Image(rgbd_vis.astype(np.uint8))
        render_frame = None
        if render_vis is not None:
            render_frame = o3d.geometry.Image(render_vis.astype(np.uint8))
        self.latest_frame = latest_frame
        self.render_frame = render_frame
