#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import json
import os
from datetime import datetime
import argparse
import cv2

import open3d.visualization.gui as gui
from isdf.visualisation import isdf_window
from isdf.modules import trainer


def optim_iter(trainer, t):
    # get/add data---------------------------------------------------------
    new_kf = None
    end = False
    finish_optim = trainer.steps_since_frame == trainer.optim_frames
    if trainer.incremental and (finish_optim or t == 0):
        # After n steps with new frame, check whether to add it to kf set.
        if t == 0:
            add_new_frame = True
        else:
            add_new_frame = trainer.check_keyframe_latest()

        if add_new_frame:
            new_frame_id = trainer.get_latest_frame_id()
            size_dataset = len(trainer.scene_dataset)
            if new_frame_id >= size_dataset:
                end = True
                print("**************************************",
                      "End of sequence",
                      "**************************************")
            else:
                print("Total step time", trainer.tot_step_time)
                print("frame______________________", new_frame_id)

                frame_data = trainer.get_data([new_frame_id])
                trainer.add_frame(frame_data)

                if t == 0:
                    trainer.last_is_keyframe = True
                    trainer.optim_frames = 200

        if t == 0 or (isdf_trainer.last_is_keyframe and not add_new_frame):
            new_kf = isdf_trainer.frames.im_batch_np[-1]
            h = int(new_kf.shape[0] / 6)
            w = int(new_kf.shape[1] / 6)
            new_kf = cv2.resize(new_kf, (w, h))

    # optimisation step---------------------------------------------
    losses, step_time = isdf_trainer.step()
    status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
    status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)

    return status, new_kf, end


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description="iSDF.")
    parser.add_argument("--config", type=str, required=True, help="input json config")
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    args, _ = parser.parse_known_args()  # ROS adds extra unrecongised args
    config_file = args.config
    incremental = args.no_incremental

    # init trainer-------------------------------------------------------------
    isdf_trainer = trainer.Trainer(
        device,
        config_file,
        incremental=incremental,
    )

    # open3d vis window --------------------------------------------------------
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = isdf_window.iSDFWindow(
        isdf_trainer,
        optim_iter,
        mono,
    )
    app.run()
