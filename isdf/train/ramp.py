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
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

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

def plot_costmap(pos, cost):

    xyz = pos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # norm = plt.Normalize(vmin=norm_min, vmax=norm_max)
    norm = plt.Normalize(vmin = np.min(cost), vmax = np.max(cost))
    # norm = cost /(np.max(cost)-np.min(cost))
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = cmap(norm(cost))
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    # pcd.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([pcd])

def plot_costmap1(pos, cost):

    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    norm = plt.Normalize(vmin = np.min(cost), vmax = np.max(cost))
    
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = cmap(norm(cost))
    img = ax.scatter(pos[:,0], pos[:,1], pos[:,2], marker='s',
                 s=10, c = colors, alpha=0.05)
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    #plt.colorbar()
    plt.show()

     

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
        pc_c = pc_c[:, :3]

        dirs_C =  pc_c/np.expand_dims(np.linalg.norm(pc_c, axis = 1), axis = -1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_c)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


        data = {
            'pcd' : pcd,
            'twc' : robot_T_camera[i],
            'dirs_C': dirs_C,
            'depth': pc_c[:,2]
        }

        for x in range(100):
            losses, step_time = isdf_trainer.step(data)
            status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
            status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
            print("STATUS :", status)

        x_min = np.min(pc[i][:, 0])
        x_max = np.max(pc[i][:, 0])

        y_min = np.min(pc[i][:, 1])
        y_max = np.max(pc[i][:, 1])

        z_min = np.min(pc[i][:, 2])
        z_max = np.max(pc[i][:, 2])
        dim = 100
        x = torch.linspace(x_min,x_max, steps=dim, device='cpu')
        y = torch.linspace(y_min,y_max, steps=dim, device='cpu')
        z = torch.linspace(z_min-0.1,z_max+0.6, steps=dim, device='cpu')

        grid = torch.meshgrid(x, y, z)
        
        grid_3d = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
        )

        print(grid_3d.shape)
        sdf = isdf_trainer.get_sdf_grid(grid_3d, dim)
        print(sdf.shape)
        
        
        isdf_pc, _ = isdf_trainer.get_sdf_grid_pc(grid_pc = grid_3d, dim = dim)
        #isdf_pc = isdf_pc[::2,::2,::2,:]
        isdf_pc = np.reshape(isdf_pc, (-1, 4))
        

        print(z_max)

        x_idx = np.logical_and(isdf_pc[:, 0]<x_max,  isdf_pc[:, 0]>x_min) 
        y_idx = np.logical_and(isdf_pc[:, 1]<y_max,  isdf_pc[:, 1]>y_min)
        z_idx = np.logical_and(isdf_pc[:, 2]>z_min,  isdf_pc[:, 2]<z_max)
        keep_idx = np.logical_and(np.logical_and(x_idx, y_idx), z_idx)
        #keep_idx = np.logical_and(x_idx, y_idx)
        
        if i == 4:
            plot_costmap(isdf_pc[keep_idx,:3], isdf_pc[keep_idx,3])

        sdf_mesh = draw3D.draw_mesh(
            sdf,
            isdf_trainer.scene_scale_np,
            isdf_trainer.bounds_transform_np,
            color_by="none",
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[i])
        o3d.visualization.draw_geometries([pcd])
        
        '''
        tree = KDTree(pc[i])
        dists, _ = tree.query(sdf_mesh.vertices, k=1)
        keep_ixs = dists < isdf_trainer.crop_dist
        face_mask = keep_ixs[sdf_mesh.faces].any(axis=1)
        sdf_mesh.update_faces(face_mask)
        sdf_mesh.remove_unreferenced_vertices()
        '''
        
        mesh = sdf_mesh.as_open3d
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])