# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
import git

from isdf.train.batch_train import batch_utils


project_dir = git.Repo(
    '.', search_parent_directories=True).working_tree_dir + "/"

save_root = project_dir + "/results/iSDF/"
data_dir = project_dir + "/data/"
base_config_file = project_dir + "/isdf/train/configs/replicaCAD.json"
scannet_root = "/mnt/sda/ScanNet/"


exp_settings = {
    # "model": {
    #     # "n_embed_funcs": [11, 18],
    #     "frac_time_perception": [1.0],
    # },
    # "loss": {
    #     "loss_type": ["L1", "L2"],
    #     "trunc_weight": [2., 4, 8.0, 12.],
    #     "trunc_distance": [0.1],
    #     "eik_weight": 0.3,
    #     "surf_norm_weight": 0.2
    # },
    # "dataset": {
    #     "noisy_depth": [1],
    # },
}


# config_files, save_paths = batch_utils.create_configs_vary1param(
#     exp_settings,
#     base_config_file,
#     data_dir,
#     scannet_root,
#     save_root,
#     runs_per_seq=1,
#     save_slices=False
# )

config_files, save_paths = batch_utils.create_configs_nruns(
    base_config_file,
    data_dir,
    scannet_root,
    save_root,
    runs_per_seq=10,
    save_slices=False
)


print("Config files:\n:", config_files)
print("Save paths:\n:", save_paths)
print("Num experiments:\n", len(config_files))


for i in range(len(config_files)):
    print("Running", config_files[i], save_paths[i])
    batch_utils.run(config_files[i], save_paths[i])
