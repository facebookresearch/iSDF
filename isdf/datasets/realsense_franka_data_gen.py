# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# script subscribes to live franka + realsense data and saves for offline eval 

import numpy as np
import json
import os
import glob
import torch
import cv2 

from isdf.ros_utils import node
from isdf.datasets.data_util import save_trajectory

# switch to current path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(os.path.join(dname, '..'))

def save_rgbd(rgb_folder : str, depth_folder : str, rgb : np.ndarray, depth : np.ndarray):
    # store the images and depth
    img_count = len(glob.glob(rgb_folder + "/*.jpg"))
    rgb_viz = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if img_count % 10 == 0:
        node.show_rgbd(rgb_viz, depth, img_count)

    cv2.imwrite(rgb_folder + "/{:05d}.jpg".format(img_count), rgb)
    np.save(depth_folder + "/{:05d}".format(img_count), depth)
    print(f"Saved {img_count:05d}.jpg")

if __name__ == "__main__":
    config_file = "train/configs/realsense_franka.json"
    with open(config_file) as json_file:
        config = json.load(json_file)

    data_root = "../data/realsense_franka"
    dataset_path = os.path.join(data_root, str(len(glob.glob(data_root + "/*"))))
    rgb_folder = os.path.join(dataset_path, "rgb")
    depth_folder = os.path.join(dataset_path, "depth")
    pose_file = os.path.join(dataset_path, "traj.txt")
    os.makedirs(rgb_folder)
    os.makedirs(depth_folder)

    torch.multiprocessing.set_start_method('spawn', force=True)
    queue = torch.multiprocessing.Queue(maxsize=1)

    process = torch.multiprocessing.Process(
        target=node.iSDFFrankaNode,
        args=(queue, False,  config["ext_calib"]),
    ) # subscribe to franka poses 

    process.start()


    print(f"Saving data to: {dataset_path}")
    Ts = []
    try:
        while True:
            data = node.get_latest_frame(queue)
            if data is not None:
                image, depth, Twc = data
                Ts.append(Twc)
                save_rgbd(rgb_folder, depth_folder, image, depth)
    except KeyboardInterrupt:
        Ts = np.dstack(Ts)
        Ts = np.rollaxis(Ts, -1)
        print(f"Saving traj to: {pose_file}")
        save_trajectory(Ts, pose_file, format="realsense_franka", timestamps=np.arange(Ts.shape[0]))


