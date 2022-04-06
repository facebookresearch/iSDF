# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import trimesh
import json
import os
import glob
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from isdf.datasets import sdf_util


def get_transf_and_scale(conf):
    transform = np.eye(4)
    if "translation" in conf.keys():
        transform[:3, 3] = conf["translation"]

    if "rotation" in conf.keys():
        # Rotation is stored with different quaterion convention to scipy
        q = np.roll(conf["rotation"], -1)
        r = R.from_quat(q)
        transform[:3, :3] = r.as_matrix()

    scale = 1.
    if "uniform_scale" in conf.keys():
        scale = conf["uniform_scale"]

    return transform, scale


def load_mesh(conf, dataset_path, dump=True):
    fname = os.path.join(dataset_path, conf["template_name"] + ".glb")
    mesh = trimesh.load(fname)

    if isinstance(mesh, trimesh.Scene) and dump:
        mesh = mesh.dump().sum()

    transform, scale = get_transf_and_scale(conf)

    mesh.apply_scale(scale)
    mesh.apply_transform(transform)

    return mesh


def load_articulated_meshes(conf, dataset_path, joint_cfg=None):
    urdf_file = os.path.join(
        dataset_path, "*", conf["template_name"] + ".urdf")
    urdf_file = glob.glob(urdf_file)[0]

    obj = URDF.load(urdf_file)

    lfk = obj.link_fk(cfg=joint_cfg)
    meshes = []
    for link in lfk:
        for visual in link.visuals:
            for mesh in visual.geometry.meshes:
                pose = lfk[link].dot(visual.origin)
                if visual.geometry.mesh is not None:
                    if visual.geometry.mesh.scale is not None:
                        S = np.eye(4)
                        # S[:3,:3] = np.diag(visual.geometry.mesh.scale)
                        np.fill_diagonal(S[:3, :3], visual.geometry.mesh.scale)
                        pose = pose.dot(S)
                mesh.apply_transform(pose)
                meshes.append(mesh)

    transform, scale = get_transf_and_scale(conf)

    for m in meshes:
        m.apply_scale(scale)
        m.apply_transform(transform)

    return meshes


def load_replicaCAD(scene_config, dataset_path, stage_sdf_dir=None,
                    joint_cfg={}, verbose=True):
    """ Load ReplicaCAD scene from config.
        Compute SDF for the scene if stage_sdf_dir is not None.
            Stage SDF is given and loaded in as our sfd_from_mesh fn
            may not be able to deal with complex topology.
            SDF for each object is computed in the stage SDF voxel
            grid and then combined into the stage SDF.

        Returns:
        Trimesh scene
        SDF
        SDF voxel grid transform
    """
    do_sdf = stage_sdf_dir is not None

    with open(scene_config, 'r') as f:
        conf = json.load(f)
    if verbose:
        print(conf.keys())

    scene = trimesh.Scene()

    stage = load_mesh(conf["stage_instance"], dataset_path, dump=False)
    scene.add_geometry(stage)
    if do_sdf:
        stage_sdf, stage_transform = sdf_util.read_sdf_txt(stage_sdf_dir)
        full_sdf = stage_sdf.copy()

    # Objects
    for obj_conf in conf["object_instances"]:
        if verbose:
            print("Adding object:", obj_conf["template_name"])
        mesh = load_mesh(obj_conf, dataset_path)
        scene.add_geometry(mesh)

        if do_sdf:
            obj_sdf, transform = sdf_util.sdf_from_mesh_gridgiven(
                mesh, transform=stage_transform, dims=full_sdf.shape)
            full_sdf = np.minimum(full_sdf, obj_sdf)

    # Articulated objects
    path = os.path.join(dataset_path, "urdf")
    for art_obj_conf in conf["articulated_object_instances"]:
        if verbose:
            print("Adding articulated object:", art_obj_conf["template_name"])
        cfg = None
        if art_obj_conf["template_name"] in joint_cfg:
            cfg = joint_cfg[art_obj_conf["template_name"]]
        obj_meshes = load_articulated_meshes(art_obj_conf, path, cfg)
        for m in obj_meshes:
            scene.add_geometry(m)

        if do_sdf:
            joined_mesh = trimesh.util.concatenate(obj_meshes)
            obj_sdf, transform = sdf_util.sdf_from_mesh_gridgiven(
                joined_mesh, transform=stage_transform,
                dims=full_sdf.shape)
            full_sdf = np.minimum(full_sdf, obj_sdf)

    if do_sdf:
        return scene, stage_sdf, full_sdf, stage_transform
    else:
        return scene


if __name__ == "__main__":
    from isdf.visualisation import sdf_viewer

    scene_name = "apt_2_v1"
    replicacad_path = "/mnt/sda/ReplicaCAD/replica_cad/"
    scene_config = f"{replicacad_path}/configs/scenes/{scene_name}.scene_instance.json"
    stage_sdf_dir = f"/home/joe/projects/incSDF/data/habitat_sdfs/frl_apartment_stage_1cm/"
    output_dir = f"/home/joe/projects/incSDF/data/gt_sdfs/{scene_name}/1cm/"
    mesh_out_dir = f"/home/joe/projects/incSDF/data/gt_sdfs/{scene_name}/"

    joint_cfg = {}
    if scene_name == "apt_2_v1":
        joint_cfg = {"fridge": {"top_door_hinge": np.pi / 2.}}
    elif scene_name == "apt_3_v1":
        joint_cfg = {"kitchen_counter": {"middle_slide_top": 0.38}}

    scene, stage_sdf, full_sdf, transform = load_replicaCAD(
        scene_config, replicacad_path,
        stage_sdf_dir=stage_sdf_dir,
        joint_cfg=joint_cfg)

    # scene = load_replicaCAD(
    #     scene_config, replicacad_path, joint_cfg=joint_cfg)
    # scene.show()

    # save sdfs and transform
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'sdf.npy'), full_sdf)
    np.save(os.path.join(output_dir, 'stage_sdf.npy'), stage_sdf)
    np.savetxt(os.path.join(output_dir, 'transform.txt'), transform)

    # save mesh
    meshes = scene.dump()
    m = trimesh.util.concatenate(meshes)
    trimesh.exchange.export.export_mesh(
        m, f'{mesh_out_dir}/mesh.obj', 'obj')

    viewer = sdf_viewer.SDFViewer(
        scene=scene,
        sdf_grid=full_sdf, grid2world=transform,
        colormap=True, tm_viewer=True,
        surface_cutoff=0.05)
