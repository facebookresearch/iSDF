# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
import imageio
import os
import scipy


def to_topdown(
    pt_xyz, axis_dim0, axis_dim1, ranges, start_dim01, im_size
):
    pt_dim0 = pt_xyz[:, axis_dim0]
    pt_dim1 = pt_xyz[:, axis_dim1]
    pt_td = np.concatenate((pt_dim0[:, None], pt_dim1[:, None]), axis=-1)
    pt_td = (pt_td - start_dim01) / ranges
    pt_td = pt_td * im_size

    return pt_td


def draw_agent(c_im, agent_position, agent_rotation, agent_radius_px):
    sprite_file = "/".join([x for x in os.getcwd().split('/')][:-1]) + \
        "/visualisation/100x100.png"
    AGENT_SPRITE = imageio.imread(sprite_file)

    AGENT_SPRITE = AGENT_SPRITE.reshape(-1, 4)
    sums = AGENT_SPRITE[:, :3].sum(axis=1)
    ixs = sums > 600
    AGENT_SPRITE[ixs, 3] = 0
    AGENT_SPRITE[~ixs, :3] = [0, 255, 0]
    AGENT_SPRITE = AGENT_SPRITE.reshape(100, 100, 4)

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_SPRITE, agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = AGENT_SPRITE.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )

    c_im = paste_overlapping_image(c_im, resized_agent, agent_position)

    return c_im


def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location,
    mask=None,
):
    """
    https://github.com/facebookresearch/habitat-lab/blob/786a5eec68cf3b4cf7134af615394c981d365a89/habitat/utils/visualizations/utils.py
    Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]): (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]): (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0]: foreground.shape[0] - max_pad[0],
        min_pad[1]: foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0]: foreground.shape[0] - max_pad[0],
            min_pad[1]: foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def add_im_to_vis(vis, im, reduce_factor=4):
    h = int(im.shape[0] / reduce_factor)
    w = int(im.shape[1] / reduce_factor)
    im = cv2.resize(im, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.copyMakeBorder(
        im, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    if vis is None:
        vis = im
    else:
        vis = np.hstack((vis, im))
    return vis
