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

import open3d as o3d
from isdf.modules import trainer

import pickle
from isdf.visualisation import draw, draw3D
from scipy.spatial import KDTree

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass

pc = []
robot_T_camera = []

with open("dump1.pkl", "rb") as f:
    for i, data in enumerate(pickleLoader(f)): 
        print(data.keys())
        pc.append(data['pc'][:,:3])
        robot_T_camera.append(data['robot_T_camera'])

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

    for i in range(len(pc)):  

        tcw = np.linalg.inv(robot_T_camera[i])
        pc_w = np.ones((pc[i].shape[0], 4))
        pc_w[:, :3] = pc[i]
        pc_c = (tcw @ pc_w.T).T 


        data = {
            'pc' : pc_c[:, :3],
            'twc' : robot_T_camera[i]
        }

        for x in range(100):
            losses, step_time = isdf_trainer.step(data)
            status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
            status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
            print("STATUS :", status)

        sdf = isdf_trainer.get_sdf_grid()
        
        sdf_mesh = draw3D.draw_mesh(
            sdf,
            isdf_trainer.scene_scale_np,
            isdf_trainer.bounds_transform_np,
            color_by="none",
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[i])
        o3d.visualization.draw_geometries([pcd])

        tree = KDTree(pc[i])
        dists, _ = tree.query(sdf_mesh.vertices, k=1)
        keep_ixs = dists < isdf_trainer.crop_dist
        face_mask = keep_ixs[sdf_mesh.faces].any(axis=1)
        sdf_mesh.update_faces(face_mask)
        sdf_mesh.remove_unreferenced_vertices()
        mesh = sdf_mesh.as_open3d
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])