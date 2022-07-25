# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import types

import numpy as np
import pyglet
import trimesh
import trimesh.transformations as tf
import trimesh.viewer
import io
import PIL.Image


def from_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(180), [1, 0, 0]
    )


def numpy_to_image(arr):
    with io.BytesIO() as f:
        PIL.Image.fromarray(arr).save(f, format="PNG")
        return pyglet.image.load(filename=None, file=f)


def _get_tile_shape(num, hw_ratio=1):
    r_num = int(round(math.sqrt(num / hw_ratio)))  # weighted by wh_ratio
    c_num = 0
    while r_num * c_num < num:
        c_num += 1
    while (r_num - 1) * c_num >= num:
        r_num -= 1
    return r_num, c_num


def display_scenes(
    data,
    height=480,
    width=640,
    tile=None,
    caption=None,
    rotate=False,
    rotation_scaling=1,
):
    import glooey

    scenes = None
    scenes_group = None
    scenes_ggroup = None
    if isinstance(data, types.GeneratorType):
        next_data = next(data)
        if isinstance(next_data, types.GeneratorType):
            scenes_ggroup = data
            scenes_group = next_data
            scenes = next(next_data)
        else:
            scenes_group = data
            scenes = next_data
    else:
        scenes = data

    scenes.pop("__clear__", False)

    if tile is None:
        nrow, ncol = _get_tile_shape(len(scenes), hw_ratio=height / width)
    else:
        nrow, ncol = tile

    configs = [
        pyglet.gl.Config(
            sample_buffers=1, samples=4, depth_size=24, double_buffer=True
        ),
        pyglet.gl.Config(double_buffer=True),
    ]
    HEIGHT_LABEL_WIDGET = 19
    PADDING_GRID = 1
    for config in configs:
        try:
            window = pyglet.window.Window(
                height=(height + HEIGHT_LABEL_WIDGET) * nrow,
                width=(width + PADDING_GRID * 2) * ncol,
                caption=caption,
                config=config,
            )
            break
        except pyglet.window.NoSuchConfigException:
            pass
    window.rotate = rotate

    window._clear = False
    if scenes_group:
        window.play = True
        window.next = False
    window.scenes_group = scenes_group
    window.scenes_ggroup = scenes_ggroup

    def usage():
        return """\
Usage:
  q: quit
  s: play / pause
  z: reset view
  n: next
  r: rotate view (clockwise)
  R: rotate view (anti-clockwise)\
"""

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            window.on_close()
        elif window.scenes_group and symbol == pyglet.window.key.S:
            window.play = not window.play
        elif symbol == pyglet.window.key.Z:
            for name in scenes:
                if isinstance(widgets[name], trimesh.viewer.SceneWidget):
                    widgets[name].reset_view()
        elif symbol == pyglet.window.key.N:
            if (
                window.scenes_ggroup
                and modifiers == pyglet.window.key.MOD_SHIFT
            ):
                try:
                    window.scenes_group = next(window.scenes_ggroup)
                    window.next = True
                    window._clear = True
                except StopIteration:
                    return
            else:
                window.next = True
        elif symbol == pyglet.window.key.C:
            camera_transform_ids = set()
            for key, widget in widgets.items():
                if isinstance(widget, trimesh.viewer.SceneWidget):
                    camera_transform_id = id(widget.scene.camera_transform)
                    if camera_transform_id in camera_transform_ids:
                        continue
                    camera_transform_ids.add(camera_transform_id)
                    print(f"{key}:")
                    camera_transform = widget.scene.camera_transform
                    print(repr(from_opengl_transform(camera_transform)))
        elif symbol == pyglet.window.key.R:
            # rotate camera
            window.rotate = not window.rotate  # 0/1
            if modifiers == pyglet.window.key.MOD_SHIFT:
                window.rotate *= -1
        elif symbol == pyglet.window.key.H:
            print(usage())
        elif symbol == pyglet.window.key.V:
            print(usage())

    def callback(dt):
        if window.rotate:
            for widget in widgets.values():
                if isinstance(widget, trimesh.viewer.SceneWidget):
                    axis = tf.transform_points(
                        [[0, 1, 0]],
                        widget.scene.camera_transform,
                        translate=False,
                    )[0]
                    widget.scene.camera_transform[...] = (
                        tf.rotation_matrix(
                            np.deg2rad(window.rotate * rotation_scaling),
                            axis,
                            point=widget.scene.centroid,
                        )
                        @ widget.scene.camera_transform
                    )
                    # print(widget.scene.camera_transform)
                    widget.view["ball"]._n_pose = widget.scene.camera_transform
            return

        if window.scenes_group and (window.next or window.play):
            try:
                scenes = next(window.scenes_group)
                clear = scenes.get("__clear__", False) or window._clear
                window._clear = False
                for key, widget in widgets.items():
                    scene = scenes[key]
                    if isinstance(widget, trimesh.viewer.SceneWidget):
                        assert isinstance(scene, trimesh.Scene)
                        if clear:
                            widget.clear()
                            widget.scene = scene
                        else:
                            widget.scene.geometry.update(scene.geometry)
                            widget.scene.graph.load(scene.graph.to_edgelist())
                        widget.scene.camera_transform[
                            ...
                        ] = scene.camera_transform
                        widget.view[
                            "ball"
                        ]._n_pose = widget.scene.camera_transform
                        widget._draw()
                    elif isinstance(widget, glooey.Image):
                        widget.set_image(numpy_to_image(scene))
            except StopIteration:
                print("Reached the end of the scenes")
                window.play = False
            window.next = False

    gui = glooey.Gui(window)
    grid = glooey.Grid()
    grid.set_padding(PADDING_GRID)

    widgets = {}
    trackball = None
    for i, (name, scene) in enumerate(scenes.items()):
        vbox = glooey.VBox()
        vbox.add(glooey.Label(text=name, color=(255,) * 3), size=0)
        if isinstance(scene, trimesh.Scene):
            widgets[name] = trimesh.viewer.SceneWidget(scene)
            if trackball is None:
                trackball = widgets[name].view["ball"]
            else:
                widgets[name].view["ball"] = trackball
        elif isinstance(scene, np.ndarray):
            widgets[name] = glooey.Image(
                numpy_to_image(scene), responsive=True
            )
        else:
            raise TypeError(f"unsupported type of scene: {scene}")
        vbox.add(widgets[name])
        grid[i // ncol, i % ncol] = vbox

    gui.add(grid)

    pyglet.clock.schedule_interval(callback, 1 / 30)
    pyglet.app.run()
    pyglet.clock.unschedule(callback)
